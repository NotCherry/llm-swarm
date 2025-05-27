import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer
import os

import global_vars

# Updated config
QWEN3_1_7B_CONFIG = {
    "vocab_size": 151936, 
    "hidden_size": 2048, 
    "num_attention_heads": 16,
    "num_key_value_heads": 8, # For Grouped Query Attention (GQA)
    "intermediate_size": 6144, 
    "rope_theta": 10000.0, 
    "rms_norm_eps": 1e-6, 
    "dtype": torch.bfloat16, 
    "max_context_length": 32768 
}

class Qwen3Model(nn.Module):
    def __init__(self, shard=None, model_size="1.7B", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        
        self.config = QWEN3_1_7B_CONFIG
        self.shard = shard if shard else type('Shard', (), {'is_first_layer': lambda: True, 'is_last_layer': lambda: True, 'start_layer': 0, 'end_layer': 24})()
        self.loaded_keys = []
        self.device = device
        self.model = nn.ModuleDict()

        if self.shard.is_first_layer():
            self.tokenizer =  AutoTokenizer.from_pretrained(
            global_vars.SELECTED_MODEL,
            use_fast=False,
            token=os.getenv('HF_TOKEN')
        )
            self.model["embed_tokens"] = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"], dtype=self.config["dtype"])
        
        # Create only the layers within the shard's range, with correct indices
        self.model["layers"] = nn.ModuleDict({
            str(i): self._transformer_block() for i in range(self.shard.start_layer, self.shard.end_layer + 1)
        })
        
        if self.shard.is_last_layer():
            self.model["norm"] = nn.RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"], dtype=self.config["dtype"])
            self.lm_head = nn.Linear(self.config["hidden_size"], self.config["vocab_size"], bias=False, dtype=self.config["dtype"])
        
        # # Initialize tokenizer if on the first shard
        # if self.shard.is_first_layer():
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         "Qwen/Qwen3-1.7B",
        #         use_fast=False,
        #         token=os.getenv('HF_TOKEN')
        #     )
        #     # Create the model structure to match Qwen3 state dict
        #     self.model = nn.Module()
        #     self.model.embed_tokens = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        #     if self.config["dtype"] != torch.float32:
        #         self.model.embed_tokens = self.model.embed_tokens.to(dtype=self.config["dtype"])
        
        # # Create transformer layers within the shard's range
        # if not hasattr(self, 'model'):
        #     self.model = nn.Module()
        
        # self.model.layers = nn.ModuleList([
        #     self._transformer_block() for i in range(self.shard.start_layer, self.shard.end_layer + 1)
        # ])
        
        # # Initialize final layers if on the last shard
        # if self.shard.is_last_layer():
        #     self.model.norm = nn.RMSNorm(self.config["hidden_size"], eps=self.config["rms_norm_eps"])
        #     if self.config["dtype"] != torch.float32:
        #         self.model.norm = self.model.norm.to(dtype=self.config["dtype"])
        #     self.lm_head = nn.Linear(self.config["hidden_size"], self.config["vocab_size"], bias=False)
        #     if self.config["dtype"] != torch.float32:
        #         self.lm_head = self.lm_head.to(dtype=self.config["dtype"])

    def _transformer_block(self):
        return TransformerBlock(
            hidden_size=self.config["hidden_size"],
            num_attention_heads=self.config["num_attention_heads"],
            intermediate_size=self.config["intermediate_size"],
            num_key_value_heads=self.config["num_key_value_heads"],
            rope_theta=self.config["rope_theta"],
            rms_norm_eps=self.config["rms_norm_eps"],
            dtype=self.config["dtype"]
        )
    
    def forward(self, input_ids=None, hidden_states=None, position_ids=None, attention_mask=None):
        if self.shard.is_first_layer():
            hidden_states = self.model.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            
            # Create causal attention mask
            if attention_mask is None:
                attention_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Process layers in this shard
        for i, layer in enumerate(self.model.layers):
            hidden_states = layer(hidden_states, position_ids, attention_mask)
        
        # Final processing if last shard
        if self.shard.is_last_layer():
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            return hidden_states, position_ids, attention_mask

    def load_state_dict(self, state_dict, strict=False, assign=False):
        super().load_state_dict(state_dict, strict, assign)
        self.loaded_keys.extend(state_dict.keys())

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, rope_theta, rms_norm_eps, dtype):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_size, num_attention_heads, num_key_value_heads, rope_theta, rms_norm_eps, dtype=dtype
        )
        self.mlp = FeedForward(hidden_size, intermediate_size, dtype)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        if dtype != torch.float32:
            self.input_layernorm = self.input_layernorm.to(dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        if dtype != torch.float32:
            self.post_attention_layernorm = self.post_attention_layernorm.to(dtype=dtype)

    def forward(self, hidden_states, position_ids, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, rope_theta, rms_norm_eps, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.key_value_dim = self.num_key_value_heads * self.head_dim
        self.rope_theta = rope_theta
        self.dtype = dtype

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.key_value_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.key_value_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        
        # Add the missing normalization layers
        self.q_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        if dtype != torch.float32:
            self.q_norm = self.q_norm.to(dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=rms_norm_eps)
        if dtype != torch.float32:
            self.k_norm = self.k_norm.to(dtype=dtype)

    def forward(self, hidden_states, position_ids, attention_mask):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to q, k, v
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply normalization to q and k BEFORE RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = self.apply_rope(q, k, position_ids)
        
        # Repeat key and value tensors for GQA (Grouped Query Attention)
        if self.num_key_value_heads != self.num_attention_heads:
            num_groups = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match scores dimensions
            if attention_mask.dim() == 4:  # [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask.expand(batch_size, self.num_attention_heads, seq_len, seq_len)
            scores = scores + attention_mask
            
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original format
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Apply output projection
        return self.o_proj(context)

    def apply_rope(self, q, k, position_ids):
        def apply_rotary_emb(x, positions, theta):
            batch_size, num_heads, seq_len, head_dim = x.shape
            half_dim = head_dim // 2
            
            # Create frequency tensor
            freqs = 1.0 / (theta ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim))
            
            # Expand positions to match expected dimensions
            if positions.dim() == 2:  # [batch_size, seq_len]
                positions = positions.unsqueeze(1)  # [batch_size, 1, seq_len]
            
            # Calculate angles: [batch_size, 1, seq_len, head_dim//2]
            angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
            
            # Create cos and sin tensors: [batch_size, 1, seq_len, head_dim//2]
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            
            # Expand to match num_heads: [batch_size, num_heads, seq_len, head_dim//2]
            cos = cos.expand(batch_size, num_heads, seq_len, half_dim)
            sin = sin.expand(batch_size, num_heads, seq_len, half_dim)
            
            # Split x into two halves
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            
            # Apply rotation
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return rotated.to(x.dtype)
        
        q = apply_rotary_emb(q, position_ids, self.rope_theta)
        k = apply_rotary_emb(k, position_ids, self.rope_theta)
        return q, k

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(gate) * up)