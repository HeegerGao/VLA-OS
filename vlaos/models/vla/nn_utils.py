# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""
Various utilities for neural networks.
Taken from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
"""

from typing import Dict, Optional, Tuple, Literal, Union, List
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


def timestep_embedding(timesteps, dim, max_period=10000, bf16=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding if not bf16 else embedding.to(torch.bfloat16)

def apply_rotary_embedding_offset(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_q: torch.Tensor,
    sin_q: torch.Tensor,
    cos_k: torch.Tensor,
    sin_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to q and k with explicit offsets for each.

    Args:
        q: Query tensor, shape [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor,   shape [batch_size, num_heads, seq_len_k, head_dim]
        cos_q, sin_q: positional embeddings for q, shape [seq_len_q, head_dim]
        cos_k, sin_k: positional embeddings for k, shape [seq_len_k, head_dim]

    Returns:
        q_rot, k_rot with rotary embedding applied.
    """
    # Rotate q
    q_cos = cos_q.unsqueeze(0).unsqueeze(0)
    q_sin = sin_q.unsqueeze(0).unsqueeze(0)
    q_rot = q * q_cos + torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1) * q_sin

    # Rotate k
    k_cos = cos_k.unsqueeze(0).unsqueeze(0)
    k_sin = sin_k.unsqueeze(0).unsqueeze(0)
    k_rot = k * k_cos + torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1) * k_sin

    return q_rot, k_rot

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(0, max_seq_len).float()
        angles = torch.einsum("i,j->ij", positions, inv_freq)

        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

    def forward(
        self,
        q: torch.Tensor,       # [B, H, seq_len_q, D]
        k: torch.Tensor,       # [B, H, seq_len_k, D]
        offset_q: int = 0,
        offset_k: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, seq_len_q, D = q.shape
        _, _, seq_len_k, _ = k.shape
        device = q.device

        # 构造全局位置索引，并对 max_seq_len 取模
        idx_q = (torch.arange(seq_len_q, device=device) + offset_q) % self.max_seq_len
        idx_k = (torch.arange(seq_len_k, device=device) + offset_k) % self.max_seq_len

        # 按索引取出每个位置的 cos/sin
        cos_q = self.cos[idx_q]    # [seq_len_q, D]
        sin_q = self.sin[idx_q]
        cos_k = self.cos[idx_k]    # [seq_len_k, D]
        sin_k = self.sin[idx_k]

        # 再 unsqueeze 到 [1,1,seq_len, D] 以便广播
        cos_q = cos_q.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len_q,D]
        sin_q = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k = cos_k.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len_k,D]
        sin_k = sin_k.unsqueeze(0).unsqueeze(0)

        # 最终应用 rotary
        q_rot = q * cos_q + torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1) * sin_q
        k_rot = k * cos_k + torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1) * sin_k

        return q_rot, k_rot
    
def generate_blockwise_causal_attention_mask(mask_config: List[Dict[str, Union[Literal["causal","bidirectional"], int, str]]],
                                             batch_size: int,
                                             num_head:int) -> torch.Tensor:
    """
    All blockwise attention is causal, which means current block can only attend to previous blocks.
    Only the attention mechanism inside the block can be configured to be causal or bidirectional.
    
    Args:
        mask_config: A list of dictionaries, each dictionary contains the mask configuration for a block.
        batch_size: The batch size.
        num_head: The number of heads.
        
    Returns:
        mask: The blockwise attention mask.
        
    
    An example mask_config:
    [
        {
            "block_name": "vlm",
            "mask_type": "causal", # inside the block, the attention is causal
            "seq_len": 512,
        },
        {
            "block_name": "action",
            "mask_type": "bidirectional", # inside the block, the attention is bidirectional
            "seq_len": 4,
        },
        {
            "block_name": "proprio",
            "mask_type": "causal",
            "seq_len": 4,
        },
    ]
    
    The example ouptut mask:
        output.shape = [batch_size, num_heads, seq_len, seq_len], where seq_len = 512+4+4 = 520
        output.dtype = torch.bool
        When output[:,:,i,j] is True, it means the i-th token can attend to the j-th token.
    """
    # Compute total sequence length
    total_seq_len = sum(block["seq_len"] for block in mask_config)
    
    # Initialize mask to all False (disallow everything initially)
    # We'll fill True in positions that are allowed.
    mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool)
    
    start = 0
    # Build blockwise mask
    for block in mask_config:
        block_type = block["mask_type"]
        block_len = block["seq_len"]
        end = start + block_len
        
        # Allow attending all tokens from *previous* blocks
        # so that each block can attend blocks 0..(i-1)
        mask[start:end, :start] = True
        
        # Within this block:
        if block_type == "causal":
            # Lower-triangular mask within the block (including diagonal)
            # i.e., token i can attend token j only if j <= i (in the block)
            block_mask = torch.tril(torch.ones((block_len, block_len), dtype=torch.bool))
            mask[start:end, start:end] = block_mask
        else:
            # 'bidirectional': every token in the block can attend
            # every other token in the block
            mask[start:end, start:end] = True
        
        start = end  # move to next block
    
    # Expand to [batch_size, num_head, seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape [1, 1, total_seq_len, total_seq_len]
    mask = mask.expand(batch_size, num_head, total_seq_len, total_seq_len)
    
    return mask

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class RawImagePatchifier(nn.Module):
    """
    A conv2d image patchifier that converts a 5-D image tensor into patches (tokens).

    Input size: [batch, video_horizon, channels, height, width]
    Output size: [batch, seq_len, hidden_size]

    Args:
        hidden_size (int): Projected patch embedding size.
        image_size (int): Height/width of the input images (assumed square).
        patch_size (int): Size of the patch (assumed square).
        num_channels (int): Number of input channels.
    """
    def __init__(self,
                 hidden_size: int,
                 image_size: int,
                 patch_size: int,
                 num_channels: int
                 ):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        
        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size."

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

    def forward(self, pixel_values: torch.FloatTensor, batch_size: int) -> torch.Tensor:
        """
        Patchifies a 5-D tensor [B, T, C, H, W] into a 3-D tensor [B, T * num_patches, embed_dim].
        
        Args:
            pixel_values (torch.FloatTensor): 5-D input of shape (batch, video_horizon, channels, height, width).

        Returns:
            torch.Tensor: 3-D patch embeddings of shape (batch, seq_len, embed_dim),
                          where seq_len = video_horizon * (image_size // patch_size) ** 2.
        """
        B, T, C, H, W = pixel_values.shape
        
        # Fold the batch and temporal dimensions for convolution
        # Shape -> [B*T, C, H, W]
        pixel_values_4d = pixel_values.view(B * T, C, H, W)

        # Apply the Conv2d patch embedding
        # patch_embeds shape -> [B*T, embed_dim, num_patches_H, num_patches_W]
        patch_embeds = self.patch_embedding(pixel_values_4d)

        # Flatten patches along spatial dims
        # [B*T, embed_dim, num_patches_H, num_patches_W] -> [B*T, embed_dim, num_patches_H * num_patches_W]
        patch_embeds = patch_embeds.flatten(2)

        # Transpose to [B*T, num_patches, embed_dim]
        patch_embeds = patch_embeds.transpose(1, 2)

        # Reshape back to [B, T, num_patches, embed_dim]
        # Then flatten T * num_patches into a single sequence dimension
        patch_embeds = rearrange(patch_embeds, "(b t) n d -> b (t n) d", b=batch_size)

        return patch_embeds
    
def pad_and_make_mask(seqs):
    """
    Args:
        seqs: List[Tensor], 每个 Tensor 形状 [seq_len_i, D]
    Returns:
        padded: Tensor of shape [B, max_len, D]
        mask:   BoolTensor of shape  [B, max_len], True 表示有效，False 表示 padding
    """
    # 1) 先计算每条序列的长度
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)  # [B]

    # 2) pad_sequence 默认在 时间 维度右侧 pad=0，batch_first=True 得到 [B, max_len, D]
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)  
    
    # 3) 构造 mask：对每个序列，前 length[i] 个位置为 True，其余为 False
    B, max_len, _ = padded.shape
    # arange: [0,1,2,...,max_len-1] -> [B, max_len] 环绕展开
    idxs = torch.arange(max_len).unsqueeze(0).expand(B, max_len)  
    mask = idxs < lengths.unsqueeze(1)  # [B, max_len] bool
    
    return padded, mask