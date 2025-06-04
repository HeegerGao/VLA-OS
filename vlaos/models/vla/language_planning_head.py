from typing import Dict, List, Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from vlaos.models.vla.nn_utils import (
    RotaryEmbedding,
    GemmaRMSNorm,
)
from einops import rearrange
from transformers.modeling_outputs import CausalLMOutputWithPast

class LanguagePlanningHeadLayer(nn.Module):
    def __init__(self, llm_emb_dim, dim_head=64, heads=8, out_dim=512, dropout=0.1, max_seq_len=10000):
        super().__init__()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        
        # Projection layers for state keys, values. In case that the state from llm is different from the dim here,
        # this layer will project the state to the dim here.
        self.to_state_kv = nn.Linear(llm_emb_dim, dim_inner, bias=True)
        
        # Projection layers for planning queries, keys, values
        self.to_planning_qkv = nn.Linear(dim_inner, 3 * dim_inner, bias=True)
        
        # Output projections
        self.to_out = nn.Linear(dim_inner, dim_inner, bias=True)
        
        self.attn_dropout = nn.Dropout(dropout)

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

    def forward(
        self, 
        planning_tokens,
        vlm_keys=None, 
        vlm_values=None, 
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_planning_kv=False,
    ):
        assert vlm_keys is not None, "vlm_keys and vlm_values must be provided."
        
        original_tokens = planning_tokens

        normed_states = self.norm1(planning_tokens)
        # Project action queries, keys, and values
        planning_qkv = self.to_planning_qkv(normed_states)
        planning_queries, planning_keys, planning_values = planning_qkv.chunk(3, dim=-1)

        planning_queries = rearrange(planning_queries, 'b n (h d) -> b h n d', h=self.heads)
        planning_keys = rearrange(planning_keys, 'b n (h d) -> b h n d', h=self.heads)
        planning_values = rearrange(planning_values, 'b n (h d) -> b h n d', h=self.heads)

        if past_key_value is not None:
            # here we assume that the past_key_value is a tuple of (keys, values)
            planning_keys = torch.cat([past_key_value[0], planning_keys], dim=-2)
            planning_values = torch.cat([past_key_value[1], planning_values], dim=-2)
        
        # we only cache the planning keys and queries
        present_key_value = (planning_keys, planning_values) if use_cache else None

        # Apply rotary embeddings
        # during inference, we need to give the planning queries the i-th rotary embedding, not the first one
        if planning_queries.shape[2] < planning_keys.shape[2]:
            # inference
            delta = planning_keys.shape[2] - planning_queries.shape[2]
            # we use planning_keys to pad planning queries to keep the device and pricision the same
            planning_queries, planning_keys = self.rotary_emb(planning_queries, planning_keys, offset_q=delta, offset_k=0)
        else:
            planning_queries, planning_keys = self.rotary_emb(planning_queries, planning_keys)

        # Project state queries, keys, and values
        state_kv = self.to_state_kv(torch.cat([vlm_keys, vlm_values], dim=1))
        state_keys, state_values = state_kv.chunk(2, dim=1)

        # Reshape to multi-head format
        state_keys = rearrange(state_keys, 'b n (h d) -> b h n d', h=self.heads)
        state_values = rearrange(state_values, 'b n (h d) -> b h n d', h=self.heads)

        state_len = state_keys.shape[2]
            
        keys = torch.cat([state_keys, planning_keys], dim=-2)
        values = torch.cat([state_values, planning_values], dim=-2)

        # Compute scaled dot-product attention (only for planning_queries)
        sim = torch.einsum('b h i d, b h j d -> b h i j', planning_queries, keys) * self.scale

        # make a causal mask for the planning qkv, and make the vlm part all true
        # sim: [B, H, Q, K]
        q_len, k_len = sim.size(2), sim.size(3)
        if past_key_value is not None:
            # incremental / use_cache 路径：单条 query，要能看到所有 keys
            diag = k_len - 1
        else:
            # full-run 路径：每行 i 只能看到 j <= i + state_len
            diag = state_len
        causal_mask = torch.tril(
            torch.ones(q_len, k_len, device=sim.device),
            diagonal=diag
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).repeat(
            sim.size(0), sim.size(1), 1, 1,
        )
        
        #  this is for the padding tokens for the planning tokens. we assume the given atten mask is [B, T_q]
        if attention_mask is not None:
            # the length of attention mask equals to q, so we first expand the mask to the same length as keys
            assert attention_mask.shape[1] == sim.size(2), f"Attention mask length {attention_mask.shape[1]} does not match query length {sim.size(2)}"
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 形状变为 (B, 1, 1, T_q)
            attention_mask = attention_mask.repeat(1, sim.size(1), sim.size(2), 1)  # 变为 (B, num_heads, T_q, T_q)
            # use an "and" operation to combine the attention mask and the causal mask
            causal_mask[:, :, :, state_len:] = (attention_mask.bool() & causal_mask[:, :, :, state_len:].clone().bool())

        causal_mask = causal_mask.bool()
        sim = sim.masked_fill(~causal_mask, float('-inf'))
        attn = sim.softmax(dim=-1)
        
        attn = self.attn_dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, values)

        # Merge heads back
        all_out = rearrange(out, 'b h n d -> b n (h d)')

        # Final projection
        all_out = self.to_out(all_out)
  
        # dropout + residual
        all_out = original_tokens + self.dropout(all_out)
        
        attentions = attn if output_attentions else None
        
        # ffn
        ffn_output = self.ffn(self.norm2(all_out))
        all_out = all_out + self.dropout(ffn_output)
        
        hidden_states_for_output = all_out if output_hidden_states else None 

        if return_planning_kv:
            # Return the planning keys and values for the next layer
            return all_out, present_key_value, attentions, hidden_states_for_output, rearrange(planning_keys, 'b h n d -> b n (h d)'), rearrange(planning_values, 'b h n d -> b n (h d)')
        else:
            return all_out, present_key_value, attentions, hidden_states_for_output, None, None

class LanguagePlanningHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        llm_emb_dim: int = 768,
        num_heads: int = 8,
        planning_mode: str = "implicit"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        assert planning_mode in ["implicit", "explicit"], f"Invalid planning mode: {planning_mode}"
        self.planning_mode = planning_mode
        
        if self.planning_mode == "explicit":
            self.planning_kv_to_llm_emb_dim = nn.Linear(hidden_size, llm_emb_dim, bias=True)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            LanguagePlanningHeadLayer(llm_emb_dim, hidden_size // num_heads, num_heads, hidden_size) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # 共享嵌入权重

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        vlm_key, 
        vlm_value,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict = True,
    ) -> CausalLMOutputWithPast:
        
        assert vlm_key is not None, "vlm_key and vlm_vaules must be provided."
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("need input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds * self.hidden_size ** 0.5

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        present_key_values = [] if use_cache else None

        past_key_values = past_key_values if past_key_values is not None else [None] * self.num_layers
        if self.planning_mode == "explicit":
            planning_keys = []
            planning_values = []

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(        
                hidden_states,
                vlm_keys=vlm_key[idx], 
                vlm_values=vlm_value[idx],
                attention_mask=attention_mask,
                past_key_value=past_key_values[idx],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_planning_kv=(self.planning_mode == "explicit"),
            )
            
            if self.planning_mode == "explicit":
                planning_keys.append(self.planning_kv_to_llm_emb_dim(layer_outputs[4]))
                planning_values.append(self.planning_kv_to_llm_emb_dim(layer_outputs[5]))
            
            hidden_states = layer_outputs[0]
            if use_cache:
                present_key_values.append(layer_outputs[1])
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],) if len(layer_outputs) > 2 else all_attentions

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        planning_keys = torch.stack(planning_keys, dim=0) if self.planning_mode == "explicit" else None
        planning_values = torch.stack(planning_values, dim=0) if self.planning_mode == "explicit" else None

        # 计算 logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将 logits 重塑为 (batch_size * seq_length, vocab_size)
            # 1) 把 logits 去掉最后一个时刻
            shift_logits = logits[:, :-1, :].contiguous()
            # 2) 把 labels 去掉第一个 token
            shift_labels = labels[:, 1:].contiguous()
            # 3) 计算 loss
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
                
        if not return_dict:
            output = (logits, present_key_values)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_attentions,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        ), planning_keys, planning_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        vlm_keys,
        vlm_values,
        eos_token_id: Optional[int] = None,
        max_length: int = 300,
        attention_mask = None,
        **kwargs,
    ) -> torch.LongTensor:
        self.eval()
        generated = input_ids
        past_key_values = None

        batch_size, cur_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, cur_len), device=input_ids.device, dtype=torch.long)
        
        # 第一次调用 forward，传入整个 input_ids 以构建初始缓存
        outputs, _, _ = self.forward(
            input_ids=generated,
            vlm_key=vlm_keys,
            vlm_value=vlm_values,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        next_token_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)        
        past_key_values = outputs.past_key_values
        
        cur_len = generated.shape[1]
        # 用于标记哪些序列已经生成了 eos_token_id
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)

        # 生成循环，从当前长度开始，每次只传入最新的 token
        while cur_len < max_length:
            # print("generating ", cur_len)
            outputs, _, _ = self.forward(
                vlm_key=vlm_keys,
                vlm_value=vlm_values,
                input_ids=generated[:, -1:],  # 只处理最新一个 token
                # attention_mask=torch.ones((batch_size,1), device=attention_mask.device, dtype=torch.long),
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **kwargs,
            )
            next_token_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            
            if eos_token_id is not None:
                # 把已完成序列的 logits 全部置为 -inf，只保留 eos_token_id
                next_token_logits = next_token_logits.masked_fill(
                    finished.unsqueeze(-1),
                    float("-inf")
                )
                next_token_logits[:, eos_token_id] = 0.0  # 确保 eos 维度最大
            
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            past_key_values = outputs.past_key_values

            cur_len += 1
            # 如果指定了 eos_token_id，则检测是否有生成停止符
            if eos_token_id is not None:
                # 更新 finished 状态：只要该批次中某个序列当前生成的 token 等于 eos_token_id，就认为该序列结束
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                # 如果所有序列都生成了 eos_token_id，则提前结束生成
                if finished.all():
                    break
        
        # planning keys and values are past_key_values - vlm_key_values
        planning_keys = []
        planning_values = []

        for layer_num in range(len(past_key_values)):
            planning_keys.append(past_key_values[layer_num][0])
            planning_values.append(past_key_values[layer_num][1])

        planning_keys = torch.stack(planning_keys, dim=0)
        planning_values = torch.stack(planning_values, dim=0)
        
        planning_keys = rearrange(planning_keys, 'l b h n d -> l b n (h d)')    # merge heads
        planning_values = rearrange(planning_values, 'l b h n d -> l b n (h d)')
        
        if self.planning_mode == "explicit":
            planning_keys = self.planning_kv_to_llm_emb_dim(planning_keys)
            planning_values = self.planning_kv_to_llm_emb_dim(planning_values)
        else:
            planning_keys, planning_values = None, None

        return generated, planning_keys, planning_values