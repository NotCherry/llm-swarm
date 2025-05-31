import io

import safetensors.torch
from src.ptcode import safe_load_metadata_single
from src.model.loading import safe_load_by_layer
import torch
import numpy as np
import safetensors
print(safetensors.__version__)
# print(help(safetensors))

file_a = "Qwen!Qwen3-1.7B.safetensors"
file_b = "../model-00002-of-00002.safetensors"
layer_name = "model.embed_tokens.weight"
l3= "model.layers.11.self_attn.q_norm.weight"
l4 = "model.layers.11.mlp.up_proj.weight"
l2 = "lm_head.weight"
f, d, aa = safe_load_metadata_single(file_a)
sel_lay = l2
a = safe_load_by_layer(file_a, layer_prefix=sel_lay)
b = safe_load_by_layer(file_b, layer_prefix=sel_lay)
print(aa)
# b = safe_load_by_layer(file_b, layer_prefix=layer_name)
print(a[sel_lay][:10])
print(b[sel_lay][:10])
# is_equal = torch.all(torch.eq(a[l2], b[layer_name]))
# assert is_equal, "model.embed_tokens.weight and lm_head.weight are not equal"

# for k in aa.keys():
#     if "dummy" not in k:
#         a = safe_load_by_layer(file_a, layer_prefix=k)
#         b = safe_load_by_layer(file_b, layer_prefix=k)

#         is_equal = torch.all(torch.eq(a[k], b[k]))
#         assert is_equal, k

# def tensor_to_bytes(v, k):
#     t = v.cpu() if v.device.type == "cuda" else v
#     t = t.contiguous()
#     return safetensors.torch._tobytes(t, k)
# import json
# fn = "test.safetensors"
# with open(fn, "wb") as f:
#     new_meta = {"model.layers.0.input_layernorm.weight": {"dtype": "BF16", "shape": [2048], "data_offsets": [0, 4096]}}
#     meta_str = json.dumps(new_meta).encode('utf-8')
#     extra = (8 - len(meta_str) % 8) % 8
#     meta_str += b" " * extra
#     size = len(meta_str)
#     f.write(int.to_bytes(size, byteorder='little', signed=False, length=8))
#     f.write(meta_str)
#     for k, v in b.items():
#         f.write()            

# b2 = safe_load_by_layer(fn, layer_prefix=layer_name)
# print(b2[layer_name][:10])
# print(len(tensor_to_bytes(b[layer_name], "k")))
# print(b[layer_name][:10])
# print(a[layer_name][:10])
# print(safe_load_metadata_single(fn))
# bb = safetensors.torch._tobytes(b[layer_name], "test")
# print(torch.frombuffer(bb, dtype=torch.bfloat16).reshape(2048)[:10])

# torch_tensor_float32 = b.to(torch.float32)

# Step 3: Convert JAX array to NumPy array (may involve copying depending on device)

# Step 4: Serialize NumPy array to bytes using pickle

# Get the byte string
# byte_string = buffer.getvalue()
# byte_string = bytearray(buffer)

# print(torch.frombuffer(byte_string, dtype=torch.bfloat16).reshape(2048)[:10])
# print(torch.from)

# print(b.numpy(dtype=bfloat16))
# print(np.array(b, dtype=bfloat16))

# f, data_start, meta = safe_load_metadata_single(file_a)
# print(data_start)
# print(meta)
# f.seek(data_start)
# size = 525340672 - 525336576
# bb = bytearray(f.read(size))
# bb = torch.frombuffer(bb, dtype=torch.bfloat16).reshape(2048)
# f.close()


# print(bb[:10])
# print(a[layer_name][:10])
# print(b[layer_name][:10])
# # assert is_equal
# with open(file_b, 'rb') as f:
#     f.seek(0)
#     js_size = int.from_bytes(f.read(8),"little")
#     print(js_size)
#     b = f.read(js_size)
#     print(len(b))
#     f.seek(525336576)
#     size = 525340672 - 525336576
#     print(size)
#     b = bytearray(f.read(size))
#     print(torch.frombuffer(b, dtype=torch.bfloat16).reshape(2048))
# import json
# with open("meta/meta-llama-Llama-3.2-1B-Instruct.json", 'r') as f:
#     js = json.loads(f.read())
#     json_bytes = json.dumps(js, separators=(",", ":")).encode('utf-8')
#     extra = (8 - len(json_bytes) % 8) % 8
#     json_bytes += b" " * extra
#     print(len(json_bytes))


    
    