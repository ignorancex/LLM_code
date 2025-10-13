# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


# Dict keys chosen to be compatible with model.config.
MODEL_ID_TO_DIMS = {
    "opt-350M": {
        "num_hidden_layers": 24,  # number of blocks in NN
        "vocab_size": 50272,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,  # context length used at training size
        "has_gate_proj": False,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,  # Grouped Query Attention ratio = num_key_value_heads / num_attention_heads
    },  # This model actually has a linear projection from 512 (embedding dimension) to 1024 (d_model)
    "llama-v3-8B": {
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "has_gate_proj": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    "phi-3-medium": {
        "num_hidden_layers": 40,
        "vocab_size": 32064,
        "hidden_size": 5120,
        "intermediate_size": 17920,
        "max_position_embeddings": 4096,
        "has_gate_proj": True,
        "num_attention_heads": 40,
        "num_key_value_heads": 10,
    },
    "phi-3-mini": {
        "num_hidden_layers": 32,
        "vocab_size": 32064,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "max_position_embeddings": 4096,
        "has_gate_proj": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    },
    "mistral-v01-7B": {
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "has_gate_proj": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    "turbosparse-mistral": {
        "num_hidden_layers": 32,
        "vocab_size": 32064,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 4096,  # the original max context length is 32k, but in Turbosparse finetuning 4k was used instead
        "has_gate_proj": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
    "dummy": {
        "num_hidden_layers": 2,
        "vocab_size": 101,
        "hidden_size": 101,
        "intermediate_size": 101,
        "max_position_embeddings": 2048,
        "has_gate_proj": False,
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
    },
}
