import torch
from torch import nn
from typing import Callable
from vlaos.models.vla.nn_utils import (
    timestep_embedding,
    RotaryEmbedding,
    GemmaRMSNorm,
)
from einops import rearrange

class LowLevelActionHeadLayer(nn.Module):
    def __init__(self, llm_emb_dim, dim_head=64, heads=8, out_dim=512, gated=True, dropout=0.1, max_seq_len=5000):
        super().__init__()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        # Projection layers for state keys, values. In case that the state from llm is different from the dim here,
        # this layer will project the state to the dim here.
        self.to_state_kv = nn.Linear(llm_emb_dim, dim_inner, bias=True)

        self.to_qkv = nn.Linear(dim_inner, 3 * dim_inner, bias=True)
        
        # Gating projection for action tokens
        self.to_action_gate = nn.Linear(dim_inner, dim_inner, bias=True) if gated else None

        # Output projections
        self.to_out = nn.Linear(dim_inner, dim_inner, bias=True)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(dim_head, max_seq_len=max_seq_len)
        
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, out_dim),
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # normalization
        self.norm1 = GemmaRMSNorm(dim_inner)
        self.norm2 = GemmaRMSNorm(dim_inner)

    def forward(self, vlm_keys, vlm_values, planning_tokens, t_tokens, proprio_tokens, action_tokens, given_attention_mask=None, three_d_tokens=None):
        assert t_tokens.shape[1] == 1, "The time should be single timestep!"
        
        proprio_len = proprio_tokens.shape[1]
        original_proprio_token_len = proprio_len
        
        if three_d_tokens is None:
            original_all_tokens = torch.cat([planning_tokens, t_tokens, proprio_tokens, action_tokens], dim=1)
        else:
            original_all_tokens = torch.cat([planning_tokens, t_tokens, proprio_tokens, three_d_tokens, action_tokens], dim=1)
            proprio_token_len += three_d_tokens.shape[1]

        # Project state queries, keys, and values
        state_kv = self.to_state_kv(torch.cat([vlm_keys, vlm_values], dim=1))
        state_keys, state_values = state_kv.chunk(2, dim=1)
        # Reshape to multi-head format
        state_keys = rearrange(state_keys, 'b n (h d) -> b h n d', h=self.heads)
        state_values = rearrange(state_values, 'b n (h d) -> b h n d', h=self.heads)

        all_tokens = self.norm1(original_all_tokens)

        token_qkv = self.to_qkv(all_tokens)
        queries, keys, values = token_qkv.chunk(3, dim=-1)
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.heads)
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.heads)
        
        keys = torch.cat([state_keys, keys], dim=2)
        values = torch.cat([state_values, values], dim=2)
        
        b, n, d = planning_tokens.shape
        _, m, _ = action_tokens.shape

        n += state_keys.shape[2]  # n is the length of planning tokens + state keys
        planning_token_len = planning_tokens.shape[1]

        # Apply rotary embeddings
        queries, keys = self.rotary_emb(queries, keys)
        
        # causal mask
        attn_mask = torch.zeros((planning_token_len+m+1+proprio_len, n+m+1+proprio_len), dtype=torch.bool, device=planning_tokens.device)
        
        attn_mask[:planning_token_len, :n] = 1  # allow vl -> vl + kv
        attn_mask[planning_token_len:planning_token_len+1+proprio_len, :n+1+proprio_len] = 1  # allow time+proprio -> vl+time+proprio
        attn_mask[planning_token_len+1+proprio_len:, :] = 1  # allow action -> vl+time+proprio+action

        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).repeat(b, 1, 1, 1)  # [b, 1, planning_token_len+m+1+proprio_len, n+m+1+proprio_len]
        if given_attention_mask is not None:
            assert given_attention_mask.shape[1] == n+m+1+proprio_len, f"The given attention mask should be of shape [b, n+m+1+proprio_len({n},{m},{proprio_len}):], got {given_attention_mask.shape}."
            # the given attention mask is [b, n+m+1+proprio_len], which is for masking the state tokens.
            # thus, for every row, it has the same attn mask
            attn_mask = attn_mask.to(torch.bool) & given_attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)  # [b, 1, 1, n+m+1+proprio_len]

        # Compute scaled dot-product attention (only for action_queries)
        sim = torch.einsum('b h i d, b h j d -> b h i j', queries, keys) * self.scale
        sim = sim.masked_fill(~attn_mask, float('-inf'))  # Apply attention mask
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, values)

        # Merge heads back
        all_out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply gating to action output if gated
        if self.to_action_gate:
            gate = self.to_action_gate(all_out).sigmoid()
            all_out = all_out * gate

        # Final projection
        all_out = self.to_out(all_out)
  
        # dropout + residual
        all_out = original_all_tokens + self.dropout(all_out)
        
        # ffn
        ffn_output = self.ffn(self.norm2(all_out))
        all_out = all_out + self.dropout(ffn_output)

        planning_tokens_output = all_out[:, 0:planning_token_len, :]
        timestep_token_output = all_out[:, planning_token_len:planning_token_len+1, :]
        proprio_token_output = all_out[:, planning_token_len+1:planning_token_len+1+original_proprio_token_len, :]
        if three_d_tokens is not None:
            three_d_token_output = all_out[:, planning_token_len+1+original_proprio_token_len:planning_token_len+1:planning_token_len+1+proprio_len, :]
        else:
            three_d_token_output = None
        action_tokens_output = all_out[:, planning_token_len+1+proprio_len:, :]      

        return planning_tokens_output, timestep_token_output, proprio_token_output, action_tokens_output, three_d_token_output

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H/P, W/P]
        x = x.flatten(2)  # [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, emb_dim]
        return x


class LowLevelActionHead(nn.Module):
    ''' A separate low-level action head with a ViT and a transformer.
        It takes the proprioceptive state and the action tokens as input and predicts the next action tokens.
    '''    
    def __init__(self, cfg, llm_emb_dim, num_layers, language_planning=False, visual_planning=False, image_foresight_planning=False):
        super(LowLevelActionHead, self).__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.hidden_size = cfg.hidden_size
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.hidden_size)
        self.action_decoder = nn.Linear(cfg.hidden_size, cfg.action_dim)
        
        self.language_planning = language_planning
        self.visual_planning = visual_planning
        self.image_foresight_planning = image_foresight_planning
        
        self.proprio_encoder = nn.Linear(cfg.proprio_dim, cfg.hidden_size)
        self.time_encoder = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        
        if self.language_planning:
            self.language_proj = nn.Linear(cfg.language_embedding_dim, cfg.hidden_size)

        if self.visual_planning:
            self.visual_plannign_patch_embedding = PatchEmbedding(3+3, cfg.patch_size, cfg.hidden_size) # the visual planning results are 3-channel images
            self.visual_planning_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, cfg.hidden_size))
            self.vision_proj = nn.Linear(cfg.hidden_size+cfg.extra_vision_embedding_dim, cfg.hidden_size)
        
        if self.image_foresight_planning:
            self.goal_image_patch_embedding = PatchEmbedding(3, 16, cfg.hidden_size)    # goal image always be 256x256, so the patch size is always 16
            self.goal_image_pos_embedding = nn.Parameter(torch.randn(1, (256 // 16) ** 2, cfg.hidden_size)) # the goal image size must be 256x256
        
        self.net = nn.ModuleList([
            LowLevelActionHeadLayer(llm_emb_dim, cfg.hidden_size//cfg.num_heads, cfg.num_heads, cfg.hidden_size, gated=cfg.gated)
            for _ in range(num_layers)
        ])

    def forward(
        self, 
        vlm_keys,
        vlm_values,
        proprio, 
        noisy_actions, 
        t, 
        language_planning_features=None, 
        visual_planning_imgs=None, 
        image_foresight_img=None,
        language_planning_feature_attention_mask=None,
        bf16=False,
        three_d_tokens=None,
    ):
        '''
        The action head only take in the vlm kv and planning res (lang reasoning, visual reasoning, image foresight) as input.
        '''

        batch_size = vlm_keys.shape[1]  # dim 0 is layer, dim 1 is batch size, dim 2 is sequence length
        assert vlm_keys.shape[0] == vlm_values.shape[0] == self.num_layers, "Invalid number of layers!"
        device = vlm_keys.device

        vlm_kv_len = vlm_keys.shape[2]
        attention_mask = torch.ones((batch_size, vlm_kv_len), device=device, dtype=torch.bool)

        planning_tokens = None
        
        if self.language_planning and language_planning_features is not None:
            if language_planning_feature_attention_mask is None:
                language_planning_feature_attention_mask = torch.ones((batch_size, language_planning_features.shape[1]), device=device, dtype=torch.bool)    
            lang_tokens = self.language_proj(language_planning_features)
            if planning_tokens is None:
                planning_tokens = lang_tokens
            attention_mask = torch.cat([attention_mask, language_planning_feature_attention_mask], dim=1)

        if self.visual_planning and visual_planning_imgs is not None:
            visual_planning_img_tokens = self.visual_plannign_patch_embedding(visual_planning_imgs)
            visual_planning_img_tokens = visual_planning_img_tokens + self.visual_planning_pos_embedding
            if planning_tokens is None:
                planning_tokens = visual_planning_img_tokens
            else:
                planning_tokens = torch.cat([planning_tokens, visual_planning_img_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, visual_planning_img_tokens.shape[1]), device=device)], dim=1)
            
        if self.image_foresight_planning and image_foresight_img is not None:
            goal_image_tokens = self.goal_image_patch_embedding(image_foresight_img.reshape(-1, 3, image_foresight_img.shape[-2], image_foresight_img.shape[-1]))
            goal_image_tokens = goal_image_tokens + self.goal_image_pos_embedding
            if planning_tokens is None:
                planning_tokens = goal_image_tokens
            else:
                planning_tokens = torch.cat([planning_tokens, goal_image_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, goal_image_tokens.shape[1]), device=device)], dim=1)
        
        if planning_tokens is None:
            # use one zero token as the planning token
            planning_tokens = torch.zeros((batch_size, 1, self.hidden_size), device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=1)  # [b, n+1], where n is the length of vlm_keys

        action_token = self.action_encoder(noisy_actions)
        proprio_token = self.proprio_encoder(proprio)
        time_emb = self.time_encoder(timestep_embedding(t, self.hidden_size, bf16=bf16)).unsqueeze(1)  # [b, 1, hidden_size]
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1+proprio_token.shape[1]+action_token.shape[1]), device=device)], dim=1)  # [b, n+m+1+proprio_len], action is m
        
        network_3d_tokens = three_d_tokens.clone() if three_d_tokens is not None else None
        for layer in range(self.num_layers):
            planning_tokens, time_emb, proprio_token, action_token, network_3d_tokens = self.net[layer](vlm_keys[layer], vlm_values[layer], planning_tokens, time_emb, proprio_token, action_token, given_attention_mask=attention_mask, three_d_tokens=network_3d_tokens)
            
        action_token = self.action_decoder(action_token)
        return action_token
    