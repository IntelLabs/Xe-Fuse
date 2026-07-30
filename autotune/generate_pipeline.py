#!/usr/bin/env python3
"""
xe-fuse pipeline generator — produces model-specific C++ pipeline benchmarks.

Generates a complete C++ test binary that:
  1. Runs a fused GEMM+epilogue pipeline (rstd -> Q -> V -> K0 -> rstd -> FFN)
  2. Verifies correctness against FP32 references
  3. Benchmarks fused vs unfused (bare GEMM + standalone ops)
  4. Prints structured output for LLM agent parsing

Usage:
    uv run generate_pipeline.py --preset llama3_8b --output test_llama3.cpp
    uv run generate_pipeline.py --config custom.json --output test_custom.cpp
    uv run generate_pipeline.py --list-presets
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from model_presets import MODEL_PRESETS, list_presets


def generate_pipeline_cpp(config: dict, preset_name: str = "custom",
                          seq_len: int = 2048, autotune: bool = False) -> str:
    template_path = Path(__file__).parent / "pipeline_template.cpp.j2"

    if not template_path.exists():
        print(f"ERROR: Template not found at {template_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from jinja2 import Template
    except ImportError:
        print("ERROR: jinja2 required. Install with: uv pip install jinja2", file=sys.stderr)
        sys.exit(1)

    with open(template_path) as f:
        tmpl = Template(f.read())

    pipeline_parts = []
    pipeline_parts.append(f"Q({'K4' if config['use_rope'] else 'K1'})")
    pipeline_parts.append("V(K1)")
    pipeline_parts.append("O(K0)")
    pipeline_parts.append(f"FFN({config['ffn_activation']})")

    # Tile selection — single tile for the whole pipeline (safest)
    # Picks based on the most constrained kernel (K4 RoPE at the Q projection shape)
    H = config["H"]
    H_kv = config["H_kv"]
    N_ffn = 2 * config["I"] if config["gated_ffn"] else config["I"]
    tile_vars = {}

    if autotune:
        from tile_selector import select_tile
        k = "k4" if config["use_rope"] else "k1"
        tile_vars["tile_shape"] = select_tile(seq_len, H, H, k)
    # else: template defaults to _256, _256, _32

    return tmpl.render(
        model_name=config["name"],
        preset_name=preset_name,
        timestamp=datetime.now().isoformat(),
        default_H=config["H"],
        default_H_kv=config["H_kv"],
        default_I=config["I"],
        use_rope=config["use_rope"],
        ffn_activation=config["ffn_activation"],
        gated_ffn=config["gated_ffn"],
        pipeline_desc=" -> ".join(pipeline_parts),
        **tile_vars,
    )


def main():
    parser = argparse.ArgumentParser(description="xe-fuse pipeline code generator")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()),
                        help="Use a built-in model preset")
    parser.add_argument("--config", help="Path to custom model config JSON")
    parser.add_argument("--output", "-o", help="Output .cpp path")
    parser.add_argument("--autotune", action="store_true",
                        help="Auto-select tile shapes per GEMM stage based on sweep data")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Target sequence length for tile selection (default: 2048)")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available model presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return

    if not args.output:
        parser.error("--output is required")

    if args.preset:
        config = MODEL_PRESETS[args.preset]
        preset_name = args.preset
    elif args.config:
        with open(args.config) as f:
            config = json.load(f)
        preset_name = Path(args.config).stem
    else:
        parser.error("Provide --preset or --config")
        return

    code = generate_pipeline_cpp(config, preset_name,
                                 seq_len=args.seq_len, autotune=args.autotune)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(code)

    n_ffn = 2 * config["I"] if config["gated_ffn"] else config["I"]
    print(f"Generated: {args.output}")
    print(f"  Model:  {config['name']}")
    print(f"  Dims:   H={config['H']}, H_kv={config['H_kv']}, I={config['I']}, N_ffn={n_ffn}")
    print(f"  Q/K:    {'K4 (RMSNorm+RoPE)' if config['use_rope'] else 'K1 (RMSNorm)'}")
    print(f"  FFN:    {config['ffn_activation']}")
    if args.autotune:
        from tile_selector import select_tile
        k = "k4" if config["use_rope"] else "k1"
        tile = select_tile(args.seq_len, config["H"], config["H"], k)
        print(f"  Autotune: M={args.seq_len} -> tile={tile}")


if __name__ == "__main__":
    main()
