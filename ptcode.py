import json
import os
import pathlib
from typing import Any, Union
import torch
from transformers import AutoTokenizer
from torch import nn
from dataclasses import dataclass
from util import log
import gc
from dotenv import load_dotenv
load_dotenv()

from util import SELECTED_MODEL

# utils

def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    # Create a simple mapping of string names to torch dtypes
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "bf16": torch.bfloat16
    }
    
    # Normalize the input string
    dtype_str = dtype_str.lower().strip()
    
    # Handle "torch." prefix if present
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]
    
    # print(dtype_str)
    # print(dtype_map.get(dtype_str))
    return dtype_map.get(dtype_str)


# Define paths and device
model_path = "./model.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    SELECTED_MODEL,
    use_fast=False,
    token=os.getenv('HF_TOKEN')  # Replace with your Hugging Face token
)
@dataclass
class Shard:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int
    loaded: bool

    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        return self.end_layer == (self.n_layers - 1)

    def get_layer_count(self) -> int:
        return self.end_layer + 1
    def to_dict(self) -> dict:
        return {
        "model_id": self.model_id,
        "start_layer": self.start_layer,
        "end_layer": self.end_layer,
        "n_layers": self.n_layers,
        "loaded": self.loaded
        }

STATE_DATA = {}

def brodcast_state(layer, data):
    global STATE_DATA
    STATE_DATA[layer] = data

def poll_state(layer):
    global STATE_DATA
    return STATE_DATA[layer]

# Define the LLaMA 3.2 1B Instruct model architecture
class LlamaModel(nn.Module):
    def __init__(self, shard: Shard):
        super().__init__()
        self.config = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "num_key_value_heads": 8,
            "dtype": torch.bfloat16
        }
        self.shard = shard
        self.model = nn.ModuleDict()
        if self.shard.is_first_layer():
            self.model["embed_tokens"] = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"], dtype=self.config["dtype"])
        
        # Create only the layers within the shard's range, with correct indices
        self.model["layers"] = nn.ModuleDict({
            str(i): self._transformer_block() for i in range(self.shard.start_layer, self.shard.end_layer + 1)
        })
        
        if self.shard.is_last_layer():
            self.model["norm"] = nn.RMSNorm(self.config["hidden_size"], self.config["rms_norm_eps"], dtype=self.config["dtype"])
            self.lm_head = nn.Linear(self.config["hidden_size"], self.config["vocab_size"], bias=False, dtype=self.config["dtype"])

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
        # minior changes for FSDP arhitecture
        if self.shard.is_first_layer():
            hidden_states = self.model["embed_tokens"](input_ids)
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Process only the layers in this shard
        for i in range(self.shard.start_layer, self.shard.end_layer + 1):
            layer = self.model["layers"][str(i)]
            hidden_states = layer(hidden_states, position_ids, attention_mask)
        
        # Final processing if last shard, otherwise broadcast
        if self.shard.is_last_layer():
            hidden_states = self.model["norm"](hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            # brodcast_state(self.shard.end_layer, hidden_states)
            return hidden_states, position_ids, attention_mask

    def forward2(self, input_ids, attention_mask=None):
        # Version for local testing
        # Get initial hidden states
        if self.shard.is_first_layer():
            hidden_states = self.model["embed_tokens"](input_ids)
        else:
            hidden_states = poll_state(self.shard.start_layer - 1)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Process only the layers in this shard
        for i in range(self.shard.start_layer, self.shard.end_layer + 1):
            layer = self.model["layers"][str(i)]
            hidden_states = layer(hidden_states, position_ids, attention_mask)
        
        # Final processing if last shard, otherwise broadcast
        if self.shard.is_last_layer():
            hidden_states = self.model["norm"](hidden_states)
            logits = self.lm_head(hidden_states)
            return logits
        else:
            brodcast_state(self.shard.end_layer, hidden_states)
            return None # Return None if not the last shard

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, num_key_value_heads, rope_theta, rms_norm_eps, dtype):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            hidden_size, num_attention_heads, num_key_value_heads, rope_theta, dtype=dtype
        )
        self.mlp = FeedForward(hidden_size, intermediate_size, dtype)
        self.input_layernorm = nn.RMSNorm(hidden_size, rms_norm_eps, dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, rms_norm_eps, dtype)

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
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, rope_theta, dtype):
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

    def forward(self, hidden_states, position_ids, attention_mask):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Repeat key and value tensors for GQA
        num_groups = self.num_attention_heads // self.num_key_value_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

        q, k = self.apply_rope(q, k, position_ids)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1, dtype=self.dtype)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)

    def apply_rope(self, q, k, position_ids):
        # Implement RoPE
        def get_sinusoidal_embeddings(positions, dim, theta):
            assert dim % 2 == 0
            positions = positions.float()
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float) / dim))
            angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            return torch.cat([cos, sin], dim=-1)

        def apply_rotary_emb(x, positions, theta):
            batch_size, num_heads, seq_len, head_dim = x.shape
            half_dim = head_dim // 2

            freqs = 1.0 / (theta ** (torch.arange(0, half_dim, device=x.device, dtype=torch.float32) / half_dim))
            angles = positions[..., None].float() * freqs  # [batch, seq_len, half_dim]
            cos = torch.cos(angles).unsqueeze(1)  # [batch, 1, seq_len, half_dim]
            sin = torch.sin(angles).unsqueeze(1)  # [batch, 1, seq_len, half_dim]

            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return rotated



        # Apply RoPE to q and k
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


def remap_dict(state_dict):        
    remapped_state_dict = {}
    for key in state_dict:
        new_key = key
        if key == "tok_embeddings.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "output.weight":
            new_key = "lm_head.weight"
        elif key.startswith("layers."):
            new_key = key.replace("layers.", "model.layers.")
            new_key = new_key.replace("attn.q_proj", "self_attn.q_proj")
            new_key = new_key.replace("attn.k_proj", "self_attn.k_proj")
            new_key = new_key.replace("attn.v_proj", "self_attn.v_proj")
            new_key = new_key.replace("attn.o_proj", "self_attn.o_proj")
        elif key == "norm.weight":
            new_key = "model.norm.weight"
        remapped_state_dict[new_key] = state_dict[key] 
    return remapped_state_dict

# remapped_state_dict = remap_dict(state_dict)


def safe_load_metadata_single(fn: Union[str, pathlib.Path]) -> tuple[torch.Tensor, int, dict[str, Any]]:


    if not ".safetensors" in str(fn):
        fn = f"{fn}/model.safetensors"
    
    print(f"Loading {fn}")  
    if (os.path.exists(fn)):
        f = open(fn, "rb")
        size = f.read(8)
        data_start = int.from_bytes(size, "little")
        raw_data = f.read(data_start)
        data_start += 8
        return f, data_start, json.loads(raw_data.decode('utf-8'))
    
    raise ValueError(f"File {fn} not found")
    # assert "model.safetensors" not in str(t), "Model path should not contain model.safetensors"

def safe_load_metadata_multifile(fn: Union[str, pathlib.Path], l) -> tuple[torch.Tensor, int, dict[str, Any], bool]:

    weight_map = json.load(open(fn, "r"))['weight_map']
    layers_files = []
    for k, v in weight_map.items():
        if l in k:
            layers_files.append(v)
    if len(set(layers_files)) > 1:
        return *safe_load_metadata_single( f"{fn[:-28]}/{layers_files[0]}"), False

    return *safe_load_metadata_single( f"{fn[:-28]}/{layers_files[0]}"), True

def safe_load_by_layer(model_path: str, layer_index: int = -1, l="model.layers.{layer_index}."):
    if layer_index >= 0:
        l = l.format(layer_index=layer_index)


    fn = f"{model_path}/model.safetensors.index.json"
    # assert os.path.exists(fn), "safetensors.index.json not exists"
    layer_weights = {}
    f, data_start, metadata, loaded_all = (*safe_load_metadata_single(model_path), True)  if not os.path.exists(fn) else  safe_load_metadata_multifile(fn, l)
    last_layer = ""
    if not loaded_all:
        last_layer = f"models.layers.{layer_index - 1}"
    for k in metadata.keys():
            if l in k or (not loaded_all and last_layer in k):
                log.info(k)
                layer_data = metadata[k]
                f.seek(data_start + (layer_data['data_offsets'][0]))
                size = layer_data['data_offsets'][1] - \
                    layer_data['data_offsets'][0]
                data = f.read(size)
                t = torch.frombuffer(data, dtype=str_to_torch_dtype(layer_data['dtype'])).reshape(layer_data['shape'])
                layer_weights[k] = t
    f.close()
    dct = remap_dict(layer_weights)

    return dct

def safe_load_layer(layer_name: str, layer_dtype: str, layer_shape, data):
    layer_weights = {
        layer_name: torch.frombuffer(data, dtype=str_to_torch_dtype(layer_dtype)).reshape(layer_shape)
    }
    
    return remap_dict(layer_weights)

def build_transformer(model_path: str, shard: Shard = None, verbose=False):
    model = LlamaModel(shard)
    loaded_keys = []

    if model.shard.start_layer == 0:
        weights = safe_load_by_layer(model_path, l="model.embed_tokens")
        loaded_keys = loaded_keys + list(weights.keys())
        model.load_state_dict(weights, strict=False)

    for i in range(model.shard.start_layer, shard.end_layer + 1):
        weights = safe_load_by_layer(model_path, i)
        loaded_keys = loaded_keys + list(weights.keys())
        model.load_state_dict(weights, strict=False)

    if model.shard.is_last_layer():
        weights = safe_load_by_layer(model_path, l="model.norm")
        loaded_keys = loaded_keys + list(weights.keys())
        model.load_state_dict(weights, strict=False)     
        
        weights = safe_load_by_layer(model_path, l="output.weight")
        if len(weights.keys()) < 1:
            weights = safe_load_by_layer(model_path, l="model.embed_tokens")
            weights['lm_head.weight'] = weights['model.embed_tokens.weight']
            loaded_keys = loaded_keys + ['lm_head.weight']
        model.load_state_dict(weights, strict=False)
    
    expected_keys = set(dict(model.named_parameters()).keys())
    loaded_keys = set(loaded_keys)
    print("Missing:", expected_keys - loaded_keys)
    print("Unexpected:", loaded_keys - expected_keys)
    gc.collect()
    return model


def model_generate_text(
        model,
        input_text = "how to never give up on goal", 
        top_p = 0.9,
        temperature = 0.7, first_layer=False, last_layer=False, h=None, p_ids=None, att =None ):
    
    with torch.no_grad():
        print("Generating...")
        if first_layer:
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            generated_ids = input_ids.clone()
            end_token_id = tokenizer.eos_token_id
        

        
        logits = None
        # IMPORTANT!!!
        if not last_layer:
            return model(generated_ids) # h, p_ids, att = model(generated_ids)
        if first_layer and last_layer:
            logits = model(generated_ids)
        else:     
            logits = model(hidden_states=h, position_ids=p_ids, attention_mask=att)

        next_token_logits = logits[:, -1, :] / temperature

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Mask tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to always keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter the mask back to original ordering
        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(decoded_output)
        if next_token_id.item() == end_token_id:
            # assert False, "implement Brodcast end to masternode "
            return False

        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print("Output:", decoded_output)
        return decoded_output


if __name__ == '__main__':
    model = None
    model2 = None

    try:
        # shard1 = Shard("LLAMA-3.2-1B", 0, 7, 16, True)
        # model = build_transformer(".", shard1)
        # shard2 = Shard("LLAMA-3.2-1B", 8, 15, 16, True)
        # model2 = build_transformer(".", shard2)

        shard1 = Shard("LLAMA-3.2-1B", 0, 15, 16, True)
        model = build_transformer(".", shard1)
    except RuntimeError as e:
        raise "We are cooked"

    # assert model != None and model2 != None

    model.to(device)
    model.eval()

    # model2.to(device)
    # model2.eval()

    # Example inference with top-p sampling
    input_text = "how to never give up on goal"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    print("Generating...")
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        generated_ids = input_ids.clone()
        end_token_id = tokenizer.eos_token_id
        max_length = 50
        top_p = 0.9
        temperature = 0.7

        for _ in range(max_length):
            # h, p_ids, att = model(generated_ids)
            # logits = model2(hidden_states=h, position_ids=p_ids, attention_mask=att)
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift mask right to always keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter the mask back to original ordering
            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )

            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # print(decoded_output)
            if next_token_id.item() == end_token_id:
                break

        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Output:", decoded_output)
