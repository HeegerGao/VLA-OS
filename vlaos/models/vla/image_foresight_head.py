import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import DropPath

from utils.infinity_utils import *

class ImageForesightPlanningLayer(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        cache=True,
    ):
        super(ImageForesightPlanningLayer, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.cache = cache
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.sa.kv_caching(cache)
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only
    

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        with torch.amp.autocast(device_type="cuda", enabled=False):    # disable half precision
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        
        x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
        x_sa, k, v = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind)
        x = x + self.drop_path(x_sa.mul_(gamma1))
        x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        
        return x, k, v


class ImageGenerationHead(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        scale_schedule=None,
        d_vlm=None,
        return_k_and_v=False,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = False
        self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.d_vlm = d_vlm
        self.scale_schedule = scale_schedule
        self.customized_flash_attn = customized_flash_attn
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        self.cache=return_k_and_v

        
        super().__init__()
        self.rng = torch.Generator(device="cuda") # NOTE: should be dist.get_device(), hard code for now
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        if self.d_vlm is not None:
            self.vlm_to_kv_compact = nn.Sequential(
                nn.Linear(self.d_vlm, self.Ct5),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.Ct5, self.Ct5),
            )
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5 ** 0.5
            self.register_buffer('cfg_uncond', cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = ImageForesightPlanningLayer(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                cache=self.cache,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))

        self.blocks = nn.ModuleList(self.unregistered_blocks)
        self.block_chunks = len(self.blocks)

        # map kv to vlm dim
        self.kv_to_vlm_dim = nn.Linear(embed_dim, d_vlm)  # vlm_dim = 128

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "infinity_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.to(torch.bfloat16), cond_BD.to(torch.bfloat16)).to(torch.bfloat16))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.infer(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.to(torch.bfloat16)       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT # kv_compact: [num_layer, bsz*seq_len, 2, dim]
            assert self.depth == kv_compact.shape[0], f'kv_compact has shape {kv_compact.shape} which means [num_layer, bsz*seq_len, 2, dim], to do attn layerwise, the depth of VD head must match the num_layer of kv_compact'
            
            if self.d_vlm is not None:
                kv_compact = self.vlm_to_kv_compact(kv_compact).contiguous() # float32
            
            kv_compact = self.text_norm(kv_compact).contiguous()

            sos = cond_BD = self.text_proj_for_sos((kv_compact[-1], cu_seqlens_k, max_seqlen_k)).contiguous()
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        ImageForesightPlanningLayer.forward
        k_list = []; v_list = []
        for i, b in enumerate(self.blocks):
            ca_kv = kv_compact[i], cu_seqlens_k, max_seqlen_k # take out the kv of the corresponding VLM layer
            if self.add_lvl_embeding_only_first_block and i == 0:
                x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
            if not self.add_lvl_embeding_only_first_block:
                x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
            x_BLC, k, v = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
            
            k_list.append(k); v_list.append(v)

        # [3. unpad the seqlen dim, and then get logits]
        k_list = torch.stack(k_list, dim=0) # [num_layer, bsz, h, L, d]
        v_list = torch.stack(v_list, dim=0) # [num_layer, bsz, h, L, d]
        k_list = rearrange(k_list, 'n b h l d -> n b l (h d)') # [num_layer, bsz, L, h*d]
        v_list = rearrange(v_list, 'n b h l d -> n b l (h d)') # [num_layer, bsz, L, h*d]
        
        k_list = self.kv_to_vlm_dim(k_list) # [num_layer, bsz, L, 128]
        v_list = self.kv_to_vlm_dim(v_list) # [num_layer, bsz, L, 128]
        
        return self.get_logits(x_BLC[:, :l_end], cond_BD), k_list, v_list    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def infer(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        returns_vemb=0,
        cfg_insertion_layer=[-5],
        vae_type=0, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert inference_mode and (not self.use_flex_attn), "you can't use flex attn in inference mode"

        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in scale_schedule]
        scale_schedule = [(pt, ph, pw) for pt, ph, pw in scale_schedule]
        
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            if self.d_vlm is not None:
                kv_compact = self.vlm_to_kv_compact(kv_compact).contiguous() # float32

            bs = B

            kv_compact = self.text_norm(kv_compact)
            sos = cond_BD = self.text_proj_for_sos((kv_compact[-1], cu_seqlens_k, max_seqlen_k)).contiguous() # the sos comes from the last v of the VLM
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous() # NOTE: cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        cur_L, ret = 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, ImageForesightPlanningLayer) else b.attn).kv_caching(True)
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cur_L += np.array(pn).prod()

            for block_idx, b in enumerate(self.unregistered_blocks):
                ca_kv = kv_compact[block_idx], cu_seqlens_k, max_seqlen_k
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                last_stage,_,_ = b(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)

            logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

            if si < gt_leak:
                idx_Bld = gt_ls_Bl[si]
            else:
                assert pn[0] == 1
                idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]

                idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

            idx_Bld_list.append(idx_Bld)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            if si != num_stages_minus_1:
                summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up) # mode: trilinear
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
            else:
                summed_codes += codes
            
            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        image_foresight_planning_keys = []; image_foresight_planning_values = []
        if inference_mode:
            for b in self.unregistered_blocks:
                image_foresight_planning_keys.append((b.sa if isinstance(b, ImageForesightPlanningLayer) else b.attn).cached_k)
                image_foresight_planning_values.append((b.sa if isinstance(b, ImageForesightPlanningLayer) else b.attn).cached_v)
                (b.sa if isinstance(b, ImageForesightPlanningLayer) else b.attn).kv_caching(False)
        
        img = vae.decode(summed_codes.squeeze(-3))
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
        return ret, idx_Bl_list, img, image_foresight_planning_keys, image_foresight_planning_values

    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: ImageForesightPlanningLayer
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift

class ImageForesightPlanningHead(nn.Module):
    def __init__(self, vae_cfg, generation_head_cfg, bsc_cfg):
        super().__init__()
        self.vae_cfg = vae_cfg
        self.generation_head_cfg = generation_head_cfg
        self.bsc_cfg = bsc_cfg

        self.videohead_vae = load_visual_tokenizer(self.vae_cfg)
        self.bitwise_self_correction = BitwiseSelfCorrection(self.videohead_vae, self.bsc_cfg)
        self.image_generation_head = ImageGenerationHead(**self.generation_head_cfg, vae_local=self.videohead_vae)
        
        # freeze the vae
        for param in self.videohead_vae.parameters():
            param.requires_grad = False

    def forward(self, vlm_key, vlm_value, future_images,):
        
        x_BLC_wo_prefix, gt_ms_idx_Bl = self.tokenize_image_with_videohead_vae(future_images)
        kv_compact, lens, cu_seqlens_k, max_len = self.prepare_condition_input_for_videogen(vlm_key, vlm_value)
        
        logits_BLV, k, v = self.image_generation_head(
            label_B_or_BLT=(kv_compact, lens, cu_seqlens_k.to("cuda"), max_len), # kv_compact: [num_layer, bsz*seq_len, 2, dim]
            x_BLC_wo_prefix=x_BLC_wo_prefix,
            scale_schedule=[(pt, ph, pw) for pt, ph, pw in self.generation_head_cfg.scale_schedule],
            cfg_infer=False,
        )
        video_planning_loss = self.calculate_loss(logits_BLV, gt_ms_idx_Bl)
        
        return video_planning_loss, k, v

    def generate_image_during_training(self, logits_BLV):

        tmp_bs, tmp_seq_len = logits_BLV.shape[:2]
        logits_BLV = logits_BLV.reshape(logits_BLV.shape[0], -1, 2)
        idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
            logits_BLV,
            rng=self.image_generation_head.rng,
            top_k=self.image_generation_head.top_k,
            top_p=self.image_generation_head.top_p,
            num_samples=1
        )[:, :, 0]
        idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

        scale_schedule = [(1, h, w) for (t, h, w) in self.generation_head_cfg.scale_schedule]
        prev = 0; cur = 0
        summed_codes = 0
        for si,pn in enumerate(scale_schedule):
            cur += np.array(pn).prod().item()
            idx_Bld_this_scale = idx_Bld[:, prev:cur, :]
            idx_Bld_this_scale = idx_Bld_this_scale.reshape(tmp_bs, pn[1], pn[2], -1).unsqueeze(1)

            code_this_scale = self.videohead_vae.quantizer.lfq.indices_to_codes(idx_Bld_this_scale, label_type='bit_label')

            if si != len(scale_schedule) - 1:
                summed_codes += F.interpolate(code_this_scale, size=scale_schedule[-1], mode="trilinear")
            else:
                summed_codes += code_this_scale
            prev = cur

        img = self.videohead_vae.decode(summed_codes.squeeze(-3))
        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)

        return img
    
    def tokenize_image_with_videohead_vae(self, next_frame):
        vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in self.generation_head_cfg.scale_schedule]
            
        raw_features, _, _ = self.videohead_vae.encode_for_raw_features(next_frame, scale_schedule=vae_scale_schedule)
        x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, next_frame, raw_features, "cuda")

        return x_BLC_wo_prefix, gt_ms_idx_Bl

    def prepare_condition_input_for_videogen(self, vlm_key, vlm_value):
        """
        The video generation model only conditions on the v of the last layer of the VLM model.
        vlm_key: [num_layer, bsz, seq_len, d]
        
        Output compact_kv: [num_layer, bsz*seq_len, 2, d]
        """
        bsz = vlm_value.shape[1]
        lens: List[int] = [vlm_value[0].shape[1]] * bsz
        max_len: int = max(lens)
        cu_seqlens_k = torch.arange(0, bsz+1) * max_len
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        
        # NOTE: concat all batch sequences together to deal with dynamic seq_len within a batch
        # cu_seqlens_k is used to capture the start index of each sequence in the batch
        # here we have same seq_len within a batch, but we follow the original code which can handle dynamic length
        vlm_key = vlm_key.reshape(vlm_key.shape[0], -1, vlm_key.shape[-1]) # [num_layer, bsz, seq_len, d] -> [num_layer, bsz*seq_len, d]
        vlm_value = vlm_value.reshape(vlm_value.shape[0], -1, vlm_value.shape[-1]) # [num_layer, bsz, seq_len, d] -> [num_layer, bsz*seq_len, d]
        compact_kv = torch.stack((vlm_key, vlm_value), dim=2) # [num_layer, bsz*seq_len, 2, d]
        
        return (compact_kv, lens, cu_seqlens_k.to("cuda"), max_len)

    def calculate_loss(self, logits_BLV, gt_ms_idx_Bl):
        scale_schedule = self.image_generation_head.scale_schedule
        V = self.videohead_vae.vocab_size
        bsz = logits_BLV.shape[0]
        train_loss = nn.CrossEntropyLoss(reduction='none')
        
        seq_len = logits_BLV.shape[1]
        seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
        training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
        gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
        
        tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
        loss = train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
        loss = loss.mean(dim=-1)
    
        lw = []
        last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
        for (pt, ph, pw) in scale_schedule:
            this_scale_area = np.sqrt(pt * ph * pw)
            lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
        lw = torch.tensor(lw, device=loss.device)[None, ...]
        lw = lw / lw.sum()

        loss = loss.mul(lw).sum(dim=-1).mean()
        
        return loss

    def generate(
        self,
        vlm_key,
        vlm_value,
        B=1,
        cfg_list=[],
        tau_list=[],
        scale_schedule=None,
        top_k=900,
        top_p=0.97,
        cfg_sc=3,
        cfg_exp_k=0.0,
        cfg_insertion_layer=-5,
        vae_type=0,
        gumbel=0,
        softmax_merge_topk=-1,
        gt_leak=-1,
        gt_ls_Bl=None,
        g_seed=None,
        sampling_per_bits=1,
    ):
        sstt = time.time()
        if not isinstance(cfg_list, list):
            cfg_list = [cfg_list] * len(scale_schedule)
        if not isinstance(tau_list, list):
            tau_list = [tau_list] * len(scale_schedule)
        
        stt = time.time()
        kv_compact, lens, cu_seqlens_k, max_len = self.prepare_condition_input_for_videogen(vlm_key, vlm_value)
        cond_tuple = (kv_compact, lens, cu_seqlens_k.to("cuda"), max_len)
        _, _, img_list, image_foresight_planning_keys, image_foresight_planning_values = self.image_generation_head.infer(
            vae=self.videohead_vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=cond_tuple, g_seed=g_seed,
            B=B, negative_label_B_or_BLT=None,
            cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
        )
        # print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")

        image_foresight_planning_keys = torch.stack(image_foresight_planning_keys)
        image_foresight_planning_keys = image_foresight_planning_keys.reshape(
            image_foresight_planning_keys.shape[0], image_foresight_planning_keys.shape[1], image_foresight_planning_keys.shape[-2], -1
        )
        image_foresight_planning_values = torch.stack(image_foresight_planning_values)
        image_foresight_planning_values = image_foresight_planning_values.reshape(
            image_foresight_planning_values.shape[0], image_foresight_planning_values.shape[1], image_foresight_planning_values.shape[-2], -1
        )
        
        return img_list, image_foresight_planning_keys, image_foresight_planning_values
