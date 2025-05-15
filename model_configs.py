
import torch


LLAMA_3_2_CONFIGS = {
    "1B": {
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
        },
        "#B": {
            "hidden_size": 3072,
            "num_hidden_layers": 28,
            "num_attention_heads": 24,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "num_key_value_heads": 8,
            "dtype": torch.bfloat16
        },

}