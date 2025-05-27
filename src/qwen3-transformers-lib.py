from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3DecoderLayer, Qwen3RotaryEmbedding, create_causal_mask, create_sliding_window_causal_mask, Qwen3RMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache

from src.structs import Shard
from transformers import AutoTokenizer
import src.global_vars as global_vars
import os

logger = logging.get_logger(__name__)

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


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config, shard: Shard, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.shard = shard
        self.device = device
        self.model = nn.ModuleDict()
        self.loaded_keys = []
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.gradient_checkpointing = False

        # Initialize tokenizer and embedding only for the first shard
        if self.shard.is_first_layer():
            self.tokenizer = AutoTokenizer.from_pretrained(
                global_vars.SELECTED_MODEL,
                use_fast=False,
                token=os.getenv('HF_TOKEN')
            )
            self.model["embed_tokens"] = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                padding_idx=self.padding_idx,
                dtype=config.dtype if hasattr(config, "dtype") else torch.float32
            )

        # Create only the layers within the shard's range, with correct indices
        self.model["layers"] = nn.ModuleDict({
            str(i): Qwen3DecoderLayer(config, layer_idx=i)
            for i in range(self.shard.start_layer, self.shard.end_layer + 1)
        })

        # Initialize norm and lm_head only for the last shard
        if self.shard.is_last_layer():
            self.model["norm"] = Qwen3RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.dtype if hasattr(config, "dtype") else torch.float32
            )
            self.model["lm_head"] = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=config.dtype if hasattr(config, "dtype") else torch.float32
            )

        # Initialize rotary embeddings (not sharded, as it’s typically shared across layers)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids=None,
        hidden_states=None,
        position_ids=None,
        attention_mask=None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Handle gradient checkpointing and cache compatibility
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # Validate past_key_values
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        # Initialize hidden_states and position_ids for the first shard
        if self.shard.is_first_layer():
            if input_ids is None and hidden_states is None:
                raise ValueError("Either input_ids or hidden_states must be provided for the first shard.")
            if input_ids is not None:
                hidden_states = self.model["embed_tokens"](input_ids)
                batch_size, seq_len = input_ids.shape
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        elif hidden_states is None:
            raise ValueError("hidden_states must be provided for non-first shards.")

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Compute cache_position if not provided
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        # Prepare attention masks
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            attention_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                attention_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # Create position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Collect hidden states and attentions if requested
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Process only the layers in this shard
        for i in range(self.shard.start_layer, self.shard.end_layer + 1):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer = self.model["layers"][str(i)]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask[layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final processing for the last shard
        if self.shard.is_last_layer():
            hidden_states = self.model["norm"](hidden_states)
            logits = self.model["lm_head"](hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                logits=logits,
            )
        else:
            # Return intermediate states for non-last shards
            return hidden_states, position_ids, attention_mask, past_key_values