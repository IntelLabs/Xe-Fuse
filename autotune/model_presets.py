"""
xe-fuse model presets — architecture configs for pipeline code generation.

Each preset defines the dimensions and operation types for one transformer layer.
The pipeline generator uses these to produce model-specific C++ test binaries.

Usage:
    from model_presets import MODEL_PRESETS
    config = MODEL_PRESETS["llama3_8b"]

Custom configs can also be loaded from JSON with the same schema.
"""

MODEL_PRESETS = {
    "llama3_8b": {
        "name": "LLaMA 3 8B",
        "H": 4096,
        "H_kv": 1024,       # 8 KV heads * 128 head_dim (GQA)
        "I": 14336,          # intermediate_size
        "num_layers": 32,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "llama2_7b": {
        "name": "LLaMA 2 7B",
        "H": 4096,
        "H_kv": 4096,       # MHA (no GQA)
        "I": 11008,
        "num_layers": 32,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "llama3_70b": {
        "name": "LLaMA 3 70B",
        "H": 8192,
        "H_kv": 1024,       # 8 KV heads * 128 head_dim
        "I": 28672,
        "num_layers": 80,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "gemma2_9b": {
        "name": "Gemma 2 9B",
        "H": 3584,
        "H_kv": 2048,       # 8 KV heads * 256 head_dim
        "I": 14336,
        "num_layers": 42,
        "use_rope": True,
        "ffn_activation": "geglu",
        "gated_ffn": True,
    },
    "gemma2_27b": {
        "name": "Gemma 2 27B",
        "H": 4608,
        "H_kv": 2048,       # 16 KV heads * 128 head_dim
        "I": 36864,
        "num_layers": 46,
        "use_rope": True,
        "ffn_activation": "geglu",
        "gated_ffn": True,
    },
    "mistral_7b": {
        "name": "Mistral 7B",
        "H": 4096,
        "H_kv": 1024,       # 8 KV heads * 128 head_dim
        "I": 14336,
        "num_layers": 32,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "qwen25_7b": {
        "name": "Qwen 2.5 7B",
        "H": 3584,
        "H_kv": 512,        # 4 KV heads * 128 head_dim
        "I": 18944,
        "num_layers": 28,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "qwen25_72b": {
        "name": "Qwen 2.5 72B",
        "H": 8192,
        "H_kv": 1024,       # 8 KV heads * 128 head_dim
        "I": 29568,
        "num_layers": 80,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "phi3_mini": {
        "name": "Phi-3 Mini 3.8B",
        "H": 3072,
        "H_kv": 3072,       # MHA (no GQA)
        "I": 8192,
        "num_layers": 32,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
    "phi3_medium": {
        "name": "Phi-3 Medium 14B",
        "H": 5120,
        "H_kv": 5120,       # MHA (no GQA)
        "I": 17920,
        "num_layers": 40,
        "use_rope": True,
        "ffn_activation": "swiglu",
        "gated_ffn": True,
    },
}


def get_preset(name: str) -> dict:
    if name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return MODEL_PRESETS[name]


def list_presets() -> None:
    print("Available model presets:")
    print(f"  {'Name':<16s} {'Model':<20s} {'H':>6s} {'H_kv':>6s} {'I':>6s} {'FFN':>8s} {'RoPE'}")
    print("  " + "-" * 78)
    for key, cfg in MODEL_PRESETS.items():
        print(f"  {key:<16s} {cfg['name']:<20s} {cfg['H']:>6d} {cfg['H_kv']:>6d} "
              f"{cfg['I']:>6d} {cfg['ffn_activation']:>8s} {'yes' if cfg['use_rope'] else 'no':>4s}")
