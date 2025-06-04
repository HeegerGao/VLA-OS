import warnings
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F

import einops
import clip # pip install git+https://github.com/openai/CLIP.git
import numpy as np
from clip.model import ModifiedResNet
from torchvision.ops import FeaturePyramidNetwork
from pytorch3d.ops import sample_farthest_points # pip install git+https://github.com/YanjieZe/3D-Diffusion-Policy.git#subdirectory=third_party/pytorch3d_simplified

from utils.robot_utils import pointcloud_from_depth_and_camera_params

class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


def load_clip():
    clip_model, clip_transforms = clip.load("RN50")
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = ModifiedResNetFeatures(layers, output_dim, heads)
    backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    return backbone, normalize


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


class AdaLN(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
             nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (N, B, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x


class FeedforwardLayer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0,
                 use_adaln=False):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, diff_ts=None):
        if diff_ts is not None:
            x = self.adaln(x, diff_ts)
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.0, use_adaln=False):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        if use_adaln:
            self.adaln = AdaLN(embedding_dim)

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None, pad_mask=None):
        if diff_ts is not None:
            adaln_query = self.adaln(query, diff_ts)
        else:
            adaln_query = query
        attn_output, _ = self.multihead_attn(
            query=adaln_query,
            key=value,
            value=value,
            rotary_pe=None if query_pos is None else (query_pos, value_pos),
            key_padding_mask=pad_mask
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class MultiheadCustomAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, slot_competition=False, return_kv=False, gate_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        ##### Custom
        self.slot_competition = slot_competition
        self.return_kv = return_kv
        self.gate_attn = None
        if gate_attn:
            self.gate_attn = Parameter(torch.randn(num_heads))  # randn
        #####

        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, k_mem=None, v_mem=None, mem_mask=None, rotary_pe=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        return multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, slot_competition=self.slot_competition,
            return_kv=self.return_kv, k_mem=k_mem, v_mem=v_mem,
            gate_attn=self.gate_attn, mem_mask=mem_mask,
            rotary_pe=rotary_pe)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 slot_competition=False,
                                 rotary_pe=None,
                                 return_kv=False,
                                 k_mem=None,
                                 v_mem=None,
                                 gate_attn=None,
                                 mem_mask=None
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if qkv_same:
        # self-attention
        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif kv_same:
        # encoder-decoder attention
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        if key is None:
            assert value is None
            k = None
            v = None
        else:

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

    else:
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = F.linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(value, _w, _b)

    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    if rotary_pe is not None:  # rotary pe ROPE disentangeld
        qp, kvp = rotary_pe
        q_cos, q_sin = qp[..., 0], qp[..., 1]
        k_cos, k_sin = kvp[..., 0], kvp[..., 1]
        q = RotaryPositionEncoding.embed_rotary(q.transpose(0, 1), q_cos, q_sin).transpose(0, 1)
        k = RotaryPositionEncoding.embed_rotary(k.transpose(0, 1), k_cos, k_sin).transpose(0, 1)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    if slot_competition:
        attn_output_weights = F.softmax(attn_output_weights, dim=-2) + 1e-8
        attn_output_weights = attn_output_weights / attn_output_weights.sum(dim=-1, keepdim=True)
    else:
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)

    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    # do memorizing transformer gating
    if (gate_attn is not None) and (k_mem is not None) and (v_mem is not None):
        k_mem = k_mem.permute((2, 0, 1))
        key_mem_len = k_mem.shape[0]
        k_mem = k_mem.contiguous().view(key_mem_len, bsz * num_heads, head_dim).transpose(0, 1)
        v_mem = v_mem.permute((2, 0, 1))
        v_mem = v_mem.contiguous().view(key_mem_len, bsz * num_heads, head_dim).transpose(0, 1)
        #         if True:
        #             k_mem = F.normalize(k_mem, dim = -1)

        attn_output_weights_mem = torch.bmm(q, k_mem.transpose(1, 2))  # [24, 16, 110]
        # bcz correspondance b/w key key is good not query, key visually
        #         attn_output_weights_mem = torch.bmm(k, k_mem.transpose(1, 2))
        attn_output_weights_mem = F.softmax(attn_output_weights_mem, dim=-1)
        if mem_mask is not None:
            mem_mask = mem_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, key_mem_len]
            attn_output_weights_mem = attn_output_weights_mem.reshape(bsz, num_heads, tgt_len, key_mem_len)
            attn_output_weights_mem = attn_output_weights_mem * mem_mask
            attn_output_weights_mem = attn_output_weights_mem.reshape(bsz * num_heads, tgt_len, key_mem_len)

        attn_output_weights_mem = F.dropout(attn_output_weights_mem, p=dropout_p, training=training)
        attn_output_mem = torch.bmm(attn_output_weights_mem, v_mem)  # [bsz * num_heads, tgt_len, head_dim]

        # gated learnable attention like memorizing transformers
        print("gate_attn ", torch.sigmoid(gate_attn))
        gate = torch.sigmoid(gate_attn).reshape(-1, 1, 1, 1)  # (n_head, 1, 1, 1)
        attn_output_mem = attn_output_mem.view(bsz, num_heads, tgt_len, head_dim).transpose(0,
                                                                                            1)  # [num_heads, bsz, tgt_len, head_dim]
        attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim).transpose(0,
                                                                                    1)  # [num_heads, bsz, tgt_len, head_dim]
        attn_output = gate * attn_output_mem + (1. - gate) * attn_output
        attn_output = attn_output.transpose(1, 0).view(bsz * num_heads, tgt_len, head_dim)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if return_kv:
        return attn_output, q, k, v
    elif need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #         return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class FFWRelativeCrossAttentionModule(nn.Module):

    def __init__(self, embedding_dim, num_attn_heads, num_layers,
                 use_adaln=True):
        super().__init__()

        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(RelativeCrossAttentionLayer(
                embedding_dim, num_attn_heads, use_adaln=use_adaln
            ))
            self.ffw_layers.append(FeedforwardLayer(
                embedding_dim, embedding_dim, use_adaln=use_adaln
            ))

    def forward(self, query, value, diff_ts=None,
                query_pos=None, value_pos=None):
        output = []
        for i in range(self.num_layers):
            query = self.attn_layers[i](
                query, value, diff_ts, query_pos, value_pos
            )
            query = self.ffw_layers[i](query, diff_ts)
            output.append(query)
        return output


class Encoder(nn.Module):
    DEPTH_DENORMALIZE = {
        'left_shoulder_camera_near': 0.009999999776482582,
        'left_shoulder_camera_far': 3.200000047683716,
        'right_shoulder_camera_near': 0.009999999776482582,
        'right_shoulder_camera_far': 3.200000047683716,
        'wrist_camera_near': 0.009999999776482582,
        'wrist_camera_far': 3.5,
        'front_camera_near': 0.009999999776482582,
        'front_camera_far': 4.5
    }

    def __init__(self,
                 backbone="clip",
                 image_size=(224, 224),
                 embedding_dim=96,
                 num_sampling_level=3,
                 nhist=3,
                 num_attn_heads=8,
                 fps_subsampling_factor=5,
                 quaternion_format='xyzw',):
        super().__init__()
        assert backbone in ["clip"]
        assert image_size in [(128, 128), (224, 224), (256, 256)]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor

        self._rotation_parametrization = '6D'
        self._quaternion_format = quaternion_format

        # Frozen backbone
        if backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        if self.image_size == (128, 128):
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == (256, 256):
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]
        elif self.image_size == (224, 224):
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [7, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def forward(self, multi_RGB_dict, multi_depth_dict, extrinsics_dict, intrinsics_dict, eef_pose_history):
        """
        ### Example Inputs:
        ```python
        multi_RGB_dict = {
            "front_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 3), dtype=torch.uint8,), # (B, H, W, 3)
            "left_shoulder_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 3), dtype=torch.uint8,),
            "right_shoulder_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 3), dtype=torch.uint8,),
            "wrist_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 3), dtype=torch.uint8,),
        }
        multi_depth_dict = {
            "front_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 1), dtype=torch.float32,),
            "left_shoulder_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 1), dtype=torch.float32,),
            "right_shoulder_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 1), dtype=torch.float32,),
            "wrist_camera": torch.randint(low=0, high=255, size=(2, 224, 224, 1), dtype=torch.float32,),
        }
        extrinsics_dict = {
            "front_camera": torch.randn(2, 4, 4),
            "left_shoulder_camera": torch.randn(2, 4, 4),
            "right_shoulder_camera": torch.randn(2, 4, 4),
            "wrist_camera": torch.randn(2, 4, 4),
        }
        intrinsics_dict = {
            "front_camera": torch.randn(2, 3, 3),
            "left_shoulder_camera": torch.randn(2, 3, 3),
            "right_shoulder_camera": torch.randn(2, 3, 3),
            "wrist_camera": torch.randn(2, 3, 3),
        }
        eef_pose_history = torch.randn(2, 10, 8)
        ```
        
        - `multi_RGB_dict`: a dict of tensor with shape (B, H, W, 3)
        - `multi_depth_dict`: a dict of tensor with shape (B, H, W, 1)
        - `extrinsics_dict`: a dict of tensor with shape (B, 4, 4)
        - `intrinsics_dict`: a dict of tensor with shape (B, 3, 3)
        
        ***IMPORTANT***: all dict should have the same name keys:
        - front_camera
        - left_shoulder_camera
        - right_shoulder_camera
        - wrist_camera
        
        Outputs (H, W are size of patchified image encoded by CLIP):
        - `adaln_gripper_feats`: (B, nhist, F), gripper history features
        - `fps_feats`: (B, N, F), sampled visual features, N = ncam*H*W//downsample_ratio
        """

        # step 1: permute to the desired type
        multi_RGB_dict = {
            key: value / 255
            for key, value in multi_RGB_dict.items() if value.shape[-1] == 3 and len(value.shape) == 4
        }
        multi_depth_dict = {
            key: value * (self.DEPTH_DENORMALIZE[f'{key}_far']-self.DEPTH_DENORMALIZE[f'{key}_near']) + self.DEPTH_DENORMALIZE[f'{key}_near']
            for key, value in multi_depth_dict.items() if value.shape[-1] == 1 and len(value.shape) == 4
        }

        multi_RGB_dict = {
            key: value.detach().cpu().numpy()
            for key, value in multi_RGB_dict.items() if type(value) == torch.Tensor
        }
        multi_depth_dict = {
            key: value.detach().cpu().numpy()
            for key, value in multi_depth_dict.items() if type(value) == torch.Tensor
        }

        pixel_positions = [pointcloud_from_depth_and_camera_params(
                depth,
                extrinsics_dict[viewname].detach().cpu().numpy(),
                intrinsics_dict[viewname].detach().cpu().numpy(),
            ) for viewname, depth in multi_depth_dict.items()
        ]
        pixel_rgb = [rgb for _, rgb in multi_RGB_dict.items()]

        device = next(self.backbone.parameters()).device
        
        pixel_positions = torch.tensor(np.stack(pixel_positions, axis=1)).permute(0,1,4,2,3).to(device)  # (B, ncam, 3, H, W)
        pixel_rgb = torch.tensor(np.stack(pixel_rgb, axis=1)).permute(0,1,4,2,3).to(device)  # (B, ncam, 3, H, W)

        # Convert rotation parametrization
        eef_pose_history = self.convert_rot(eef_pose_history).to(device).to(torch.bfloat16)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encode_images(
            pixel_rgb, pixel_positions,
        ) # [(B, ncam, F, H_i, W_i)], [(B, ncam * H_i * W_i, 3)]

        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0].to(torch.bfloat16)

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encode_curr_gripper(
            eef_pose_history, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats = self.run_fps(
            context_feats.transpose(0, 1),
        )

        return torch.cat([adaln_gripper_feats, fps_feats], dim=1)  # concat in the seq dim
    
    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.
        Args:
            - curr_gripper: (B, nhist, 3+)
        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.
        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)
        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.
        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions
        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)

        return rgb_feats_pyramid, pcd_pyramid

    def run_fps(self, context_features,):
        # context_features (Np, B, F)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_points, _ = sample_farthest_points(
            points=einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            K=max(npts // self.fps_subsampling_factor, 1),
        )

        return sampled_points


def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_ortho6d_from_rotation_matrix(matrix):
    # The orhto6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
    return ortho6d


