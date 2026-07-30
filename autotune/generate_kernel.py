#!/usr/bin/env python3
"""
xe-fuse kernel generator — produces C++ benchmark files from kernel specs.

This is the foundation for LLM-driven kernel optimization (Phase 3).
An LLM agent describes a kernel as a Python dict, this script generates
the C++ source, and the runner compiles + benchmarks it on GPU.

Usage:
    uv run generate_kernel.py --spec spec.json --output kernel.cpp
    uv run generate_kernel.py --preset k1 --output kernel.cpp

Kernel spec format (JSON):
{
    "name": "K4v2_merged",
    "evt_description": "D = RoPE(acc * R[m], cos_sin) via merged visitor",
    "tile_shape": "_256, _256, _32",
    "evt_typedefs": "using EVT = b::RoPEScaled<TileShape, float, float>;",
    "aux_data": [
        {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42},
        {"name": "cos_sin", "type": "float", "shape": "M * N * L", "init_seed": 2024}
    ],
    "evt_args": [
        "// child 0: ColBroadcast<R>",
        "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
        "scale_args.ptr_col = block_scale.get();",
        "scale_args.null_default = float(1);",
        "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
        "",
        "// child 1: AuxLoad<cos_sin>",
        "typename b::AuxLoad<float>::Arguments cs_args;",
        "cs_args.ptr_aux = block_cos_sin.get();",
        "cs_args.null_default = float(0);",
        "cs_args.dAux = cutlass::make_cute_packed_stride(",
        "    cute::Stride<int64_t, cute::Int<1>, int64_t>{}, make_shape(M, N, L));",
        "",
        "// root: XeRoPEScaledCompute",
        "typename xe_fuse::XeRoPEScaledCompute::Arguments rope_args{};",
        "",
        "typename EVT::Arguments evt_args{scale_args, cs_args, rope_args};"
    ]
}
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PRESETS = {
    "k1": {
        "name": "K1_RmsNorm",
        "evt_description": "D = acc * R[m]",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::ScaleRows<b::Acc, TileShape, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "typename b::Acc::Arguments accum_args{};",
            "",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "",
            "typename b::MulOp<>::Arguments mul_args{};",
            "",
            "typename EVT::Arguments evt_args{accum_args, scale_args, mul_args};"
        ]
    },
    "k3": {
        "name": "K3_RoPE",
        "evt_description": "D = RoPE(acc, cos_sin)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::RoPE<float>;",
        "aux_data": [
            {"name": "cos_sin", "type": "float", "shape": "M * N * L", "init_seed": 2024}
        ],
        "evt_args": [
            "typename b::AuxLoad<float>::Arguments cs_args;",
            "cs_args.ptr_aux = block_cos_sin.get();",
            "cs_args.null_default = float(0);",
            "cs_args.dAux = cutlass::make_cute_packed_stride(",
            "    cute::Stride<int64_t, cute::Int<1>, int64_t>{}, make_shape(M, N, L));",
            "",
            "typename xe_fuse::XeRoPECompute::Arguments rope_args{};",
            "",
            "typename EVT::Arguments evt_args{cs_args, rope_args};"
        ]
    },
    "k4": {
        "name": "K4_RmsNormRoPE",
        "evt_description": "D = RoPE(acc * R[m], cos_sin) via composed tree",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::RoPEComposed<b::ScaleRows<b::Acc, TileShape, float>, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42},
            {"name": "cos_sin", "type": "float", "shape": "M * N * L", "init_seed": 2024}
        ],
        "evt_args": [
            "// child 0: ScaleRows (inner tree)",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "typename b::MulOp<float, float>::Arguments mul_args{};",
            "typename b::ScaleRows<b::Acc, TileShape, float>::Arguments rms_args{accum_args, scale_args, mul_args};",
            "",
            "// child 1: AuxLoad<cos_sin>",
            "typename b::AuxLoad<float>::Arguments cs_args;",
            "cs_args.ptr_aux = block_cos_sin.get();",
            "cs_args.null_default = float(0);",
            "cs_args.dAux = cutlass::make_cute_packed_stride(",
            "    cute::Stride<int64_t, cute::Int<1>, int64_t>{}, make_shape(M, N, L));",
            "",
            "typename xe_fuse::XeRoPEComputeTwoChild::Arguments rope_args{};",
            "typename EVT::Arguments evt_args{rms_args, cs_args, rope_args};"
        ]
    },
    "k4v2": {
        "name": "K4v2_RoPEScaled",
        "evt_description": "D = RoPE(acc * R[m], cos_sin) via merged visitor (flat tree)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::RoPEScaled<TileShape, float, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42},
            {"name": "cos_sin", "type": "float", "shape": "M * N * L", "init_seed": 2024}
        ],
        "evt_args": [
            "// child 0: ColBroadcast<R>",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "",
            "// child 1: AuxLoad<cos_sin>",
            "typename b::AuxLoad<float>::Arguments cs_args;",
            "cs_args.ptr_aux = block_cos_sin.get();",
            "cs_args.null_default = float(0);",
            "cs_args.dAux = cutlass::make_cute_packed_stride(",
            "    cute::Stride<int64_t, cute::Int<1>, int64_t>{}, make_shape(M, N, L));",
            "",
            "// root: XeRoPEScaledCompute (merged)",
            "typename xe_fuse::XeRoPEScaledCompute::Arguments rope_args{};",
            "",
            "typename EVT::Arguments evt_args{scale_args, cs_args, rope_args};"
        ]
    },
    "k0a": {
        "name": "K0a_ResidualGamma",
        "evt_description": "D = gamma[n] * (acc + residual)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::ScaleCols<b::AddResidual<bf16>, TileShape, float>;",
        "aux_data": [
            {"name": "residual", "type": "bf16", "shape": "M * N * L", "init_seed": 2021},
            {"name": "gamma", "type": "float", "shape": "N * L", "init_seed": 99}
        ],
        "evt_args": [
            "// RowBroadcast<gamma> (child 0 of outer Mul)",
            "typename b::RowBroadcast<0, TileShape, float>::Arguments gamma_args;",
            "gamma_args.ptr_row = block_gamma.get();",
            "gamma_args.null_default = float(1);",
            "gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};",
            "",
            "// Inner Add tree (child 1 of outer Mul): Acc + AuxLoad<residual>",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::AuxLoad<bf16>::Arguments res_args;",
            "res_args.ptr_aux = block_residual.get();",
            "res_args.null_default = bf16(0);",
            "res_args.dAux = cutlass::make_cute_packed_stride(",
            "    cute::Stride<int64_t, cute::Int<1>, int64_t>{}, make_shape(M, N, L));",
            "typename b::AddOp<>::Arguments add_args{};",
            "typename b::AddResidual<bf16>::Arguments inner_args{accum_args, res_args, add_args};",
            "",
            "// Outer Mul",
            "typename b::MulOp<>::Arguments mul_args{};",
            "typename EVT::Arguments evt_args{gamma_args, inner_args, mul_args};"
        ]
    },
    "k2": {
        "name": "K2_RmsNormSwiGLU",
        "evt_description": "D = SwiGLU(acc * R[m])",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "// Inner: ScaleRows",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "typename b::MulOp<float, float>::Arguments mul_args{};",
            "typename b::ScaleRows<b::Acc, TileShape, float>::Arguments rms_args{accum_args, scale_args, mul_args};",
            "",
            "// Outer: SwiGLU",
            "typename xe_fuse::XePairwiseCompute<xe_fuse::SwiGLUFn>::Arguments swiglu_args{};",
            "typename EVT::Arguments evt_args{rms_args, swiglu_args};"
        ]
    },
    "k2_geglu": {
        "name": "K2_RmsNormGeGLU",
        "evt_description": "D = GeGLU(acc * R[m])",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::GeGLU<b::ScaleRows<b::Acc, TileShape, float>>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "// Inner: ScaleRows",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "typename b::MulOp<float, float>::Arguments mul_args{};",
            "typename b::ScaleRows<b::Acc, TileShape, float>::Arguments rms_args{accum_args, scale_args, mul_args};",
            "",
            "// Outer: GeGLU",
            "typename xe_fuse::XePairwiseCompute<xe_fuse::GeGLUFn>::Arguments geglu_args{};",
            "typename EVT::Arguments evt_args{rms_args, geglu_args};"
        ]
    },
    "k1v2": {
        "name": "K1v2_ScaleRowsMerged",
        "evt_description": "D = acc * R[m] via merged visitor (flat tree)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::ScaleRowsMerged<TileShape, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "// child 0: ColBroadcast<R>",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "",
            "// root: XeScaleRowsCompute (merged)",
            "typename xe_fuse::XeScaleRowsCompute::Arguments visitor_args{};",
            "",
            "typename EVT::Arguments evt_args{scale_args, visitor_args};"
        ]
    },
    "k2v2": {
        "name": "K2v2_SwiGLUScaled",
        "evt_description": "D = SwiGLU(acc * R[m]) via merged visitor (flat tree)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::SwiGLUScaled<TileShape, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "// child 0: ColBroadcast<R>",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "",
            "// root: XeScaleRowsSwiGLUCompute (merged)",
            "typename xe_fuse::XeScaleRowsSwiGLUCompute::Arguments visitor_args{};",
            "",
            "typename EVT::Arguments evt_args{scale_args, visitor_args};"
        ]
    },
    "k2v2_geglu": {
        "name": "K2v2_GeGLUScaled",
        "evt_description": "D = GeGLU(acc * R[m]) via merged visitor (flat tree)",
        "tile_shape": "_256, _256, _32",
        "evt_typedefs": "using EVT = b::GeGLUScaled<TileShape, float>;",
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "evt_args": [
            "// child 0: ColBroadcast<R>",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;",
            "scale_args.ptr_col = block_scale.get();",
            "scale_args.null_default = float(1);",
            "scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "",
            "// root: XeScaleRowsGeGLUCompute (merged)",
            "typename xe_fuse::XeScaleRowsGeGLUCompute::Arguments visitor_args{};",
            "",
            "typename EVT::Arguments evt_args{scale_args, visitor_args};"
        ]
    },
    "w8a8_dequant": {
        "name": "W8A8_Dequant",
        "evt_description": "D_bf16 = int32_acc * scale_token[m] * scale_channel[n]",
        "tile_shape": "_256, _256, _32",
        "element_a": "int8_t",
        "element_b": "int8_t",
        "element_d": "bf16",
        "element_acc": "int32_t",
        "element_compute": "float",
        "alignment_ab": 32,
        "alignment_cd": 8,
        "evt_typedefs": "using EVT = b::DequantW8A8<TileShape, float, float>;",
        "aux_data": [
            {"name": "scale_token", "type": "float", "shape": "M * L", "init_seed": 42},
            {"name": "scale_channel", "type": "float", "shape": "N * L", "init_seed": 99}
        ],
        "evt_args": [
            "// Inner Mul: Acc * scale_token[m] (ColBroadcast, Idx=0)",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments token_args;",
            "token_args.ptr_col = block_scale_token.get();",
            "token_args.null_default = float(1);",
            "token_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "typename b::MulOp<>::Arguments inner_mul_args{};",
            "",
            "// Outer Mul: (acc * scale_token) * scale_channel[n] (RowBroadcast, Idx=1)",
            "typename b::RowBroadcast<1, TileShape, float>::Arguments channel_args;",
            "channel_args.ptr_row = block_scale_channel.get();",
            "channel_args.null_default = float(1);",
            "channel_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};",
            "typename b::MulOp<>::Arguments outer_mul_args{};",
            "",
            "// Compose: EVT = Mul(Mul(Acc, ColBcast), RowBcast)",
            "using InnerMul = b::Mul<b::Acc, b::ColBroadcast<0, TileShape, float>>;",
            "typename InnerMul::Arguments inner_args{accum_args, token_args, inner_mul_args};",
            "typename EVT::Arguments evt_args{inner_args, channel_args, outer_mul_args};"
        ]
    },
    "w8a8_dequant_biased": {
        "name": "W8A8_DequantBiased",
        "evt_description": "D_bf16 = int32_acc * scale_token[m] * scale_channel[n] + bias[n]",
        "tile_shape": "_256, _256, _32",
        "element_a": "int8_t",
        "element_b": "int8_t",
        "element_d": "bf16",
        "element_acc": "int32_t",
        "element_compute": "float",
        "alignment_ab": 32,
        "alignment_cd": 8,
        "evt_typedefs": "using EVT = b::DequantW8A8Biased<TileShape, float, float, float>;",
        "aux_data": [
            {"name": "scale_token", "type": "float", "shape": "M * L", "init_seed": 42},
            {"name": "scale_channel", "type": "float", "shape": "N * L", "init_seed": 99},
            {"name": "bias", "type": "float", "shape": "N * L", "init_seed": 77}
        ],
        "evt_args": [
            "// Inner Mul: Acc * scale_token[m] (ColBroadcast, Idx=0)",
            "typename b::Acc::Arguments accum_args{};",
            "typename b::ColBroadcast<0, TileShape, float>::Arguments token_args;",
            "token_args.ptr_col = block_scale_token.get();",
            "token_args.null_default = float(1);",
            "token_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};",
            "typename b::MulOp<>::Arguments inner_mul_args{};",
            "",
            "// Middle Mul: (acc * scale_token) * scale_channel[n] (RowBroadcast, Idx=1)",
            "typename b::RowBroadcast<1, TileShape, float>::Arguments channel_args;",
            "channel_args.ptr_row = block_scale_channel.get();",
            "channel_args.null_default = float(1);",
            "channel_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};",
            "typename b::MulOp<>::Arguments mid_mul_args{};",
            "",
            "// Outer Add: dequant + bias[n] (RowBroadcast, Idx=2)",
            "typename b::RowBroadcast<2, TileShape, float>::Arguments bias_args;",
            "bias_args.ptr_row = block_bias.get();",
            "bias_args.null_default = float(0);",
            "bias_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};",
            "typename b::AddOp<>::Arguments add_args{};",
            "",
            "// Compose: EVT = Add(Mul(Mul(Acc, ColBcast), RowBcast), RowBcast_bias)",
            "using InnerMul = b::Mul<b::Acc, b::ColBroadcast<0, TileShape, float>>;",
            "using DequantMul = b::DequantW8A8<TileShape, float, float>;",
            "typename InnerMul::Arguments inner_args{accum_args, token_args, inner_mul_args};",
            "typename DequantMul::Arguments dequant_args{inner_args, channel_args, mid_mul_args};",
            "typename EVT::Arguments evt_args{dequant_args, bias_args, add_args};"
        ]
    }
}


def generate_aux_allocations(aux_data: list[dict]) -> str:
    lines = []
    type_map = {"float": "float", "bf16": "bf16", "int8": "int8_t", "int32": "int32_t"}
    for aux in aux_data:
        ctype = type_map.get(aux["type"], aux["type"])
        name = aux["name"]
        shape = aux["shape"]
        seed = aux.get("init_seed", 2020)
        lines.append(f"  cutlass::DeviceAllocation<{ctype}> block_{name}(static_cast<size_t>({shape}));")
        lines.append(f"  initialize_block(block_{name}, {seed});")
    return "\n".join(lines)


def generate_cpp(spec: dict, defaults: dict | None = None) -> str:
    defaults = defaults or {}
    template_path = Path(__file__).parent / "kernel_template.cpp.j2"

    if template_path.exists():
        try:
            from jinja2 import Template
            with open(template_path) as f:
                tmpl = Template(f.read())

            elem_a = spec.get("element_a", "bf16")
            elem_b = spec.get("element_b", "bf16")
            elem_d = spec.get("element_d", "bf16")
            elem_acc = spec.get("element_acc", "float")
            elem_compute = spec.get("element_compute", "float")
            align_ab = spec.get("alignment_ab", 8)
            align_cd = spec.get("alignment_cd", 8)
            make_gemm_extra = ""
            if align_ab != 8 or align_cd != 8:
                make_gemm_extra = (f",\n    cutlass::layout::RowMajor, cutlass::layout::RowMajor, "
                                   f"{align_ab}, {align_cd}")

            return tmpl.render(
                kernel_name=spec["name"],
                timestamp=datetime.now().isoformat(),
                evt_description=spec["evt_description"],
                tile_shape=spec.get("tile_shape", "_256, _256, _32"),
                evt_typedefs=spec["evt_typedefs"],
                aux_allocations=generate_aux_allocations(spec.get("aux_data", [])),
                evt_args_construction="  " + "\n  ".join(spec.get("evt_args", [])),
                default_m=defaults.get("m", 4096),
                default_n=defaults.get("n", 4096),
                default_k=defaults.get("k", 4096),
                default_iterations=defaults.get("iterations", 200),
                default_verify=defaults.get("verify", 0),
                has_verify=False,
                element_a=elem_a,
                element_b=elem_b,
                element_d=elem_d,
                element_acc=elem_acc,
                element_compute=elem_compute,
                make_gemm_extra=make_gemm_extra,
            )
        except ImportError:
            pass

    # Fallback: inline template (no jinja2 dependency)
    return generate_cpp_inline(spec, defaults)


def generate_cpp_inline(spec: dict, defaults: dict | None = None) -> str:
    defaults = defaults or {}
    m = defaults.get("m", 4096)
    n = defaults.get("n", 4096)
    k = defaults.get("k", 4096)
    iters = defaults.get("iterations", 200)
    verify = defaults.get("verify", 0)

    aux_alloc = generate_aux_allocations(spec.get("aux_data", []))
    evt_args_code = "  " + "\n  ".join(spec.get("evt_args", []))

    # Element types and alignments (with backward-compatible defaults)
    elem_a = spec.get("element_a", "bf16")
    elem_b = spec.get("element_b", "bf16")
    elem_d = spec.get("element_d", "bf16")
    elem_acc = spec.get("element_acc", "float")
    elem_compute = spec.get("element_compute", "float")
    align_ab = spec.get("alignment_ab", 8)
    align_cd = spec.get("alignment_cd", 8)

    # MakeGemm template args beyond the 7 positional defaults
    make_gemm_extra = ""
    if align_ab != 8 or align_cd != 8:
        make_gemm_extra = (f",\n    cutlass::layout::RowMajor, cutlass::layout::RowMajor, "
                           f"{align_ab}, {align_cd}")

    return f"""\
// Auto-generated xe-fuse kernel benchmark
// Kernel: {spec["name"]}
// Generated: {datetime.now().isoformat()}
// EVT tree: {spec["evt_description"]}

#include "xe-fuse/builder/epilogue_builder.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl_common.hpp"
#include "helper.h"

#include <iostream>

using namespace cute;
namespace b = xe_fuse::builder;
using bf16 = cutlass::bfloat16_t;

using TileShape = Shape<{spec.get("tile_shape", "_256, _256, _32")}>;

// ---- EVT tree ----
{spec["evt_typedefs"]}

using KernelConfig = b::MakeGemm<EVT, {elem_a}, {elem_b}, {elem_d}, {elem_acc}, {elem_compute}, TileShape{make_gemm_extra}>;
using GemmOp = typename KernelConfig::Gemm;

struct Options {{
  int m = {m}, n = {n}, k = {k}, l = 1;
  int iterations = {iters};
  int verify = {verify};

  void parse(int argc, char const** args) {{
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, {m});
    cmd.get_cmd_line_argument("n", n, {n});
    cmd.get_cmd_line_argument("k", k, {k});
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, {iters});
    cmd.get_cmd_line_argument("verify", verify, {verify});
  }}
}};

int main(int argc, const char** argv) {{
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{{}}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{{}}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(KernelConfig::StrideC{{}}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(KernelConfig::StrideD{{}}, make_shape(M, N, L));

  cutlass::DeviceAllocation<{elem_a}> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<{elem_b}> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<{elem_d}> block_D(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

{aux_alloc}

{evt_args_code}

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  }};
  typename GemmOp::GemmKernel::Arguments arguments{{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {{M, N, K, L}},
    {{block_A.get(), stride_A, block_B.get(), stride_B}},
    epilogue_args, hw_info
  }};

  GemmOp gemm_op;
  size_t ws_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {{
    std::cerr << "can_implement failed" << std::endl;
    return 1;
  }}
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  std::cout << "Disposition: launched" << std::endl;

  if (opts.iterations > 0) {{
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("{spec["name"]}: [%4.3f]TFlop/s  (%6.4f)ms\\n", tflops / time_s, time_s * 1000);
  }}

  return 0;
}}
"""


def generate_standalone_cpp(spec: dict, defaults: dict | None = None) -> str:
    """Generate a standalone (non-GEMM) kernel benchmark."""
    defaults = defaults or {}
    m = defaults.get("m", 4096)
    n = defaults.get("n", 4096)
    iters = defaults.get("iterations", 200)

    aux_alloc = generate_aux_allocations(spec.get("aux_data", []))
    op_code = "\n  ".join(spec.get("op_code", []))

    return f"""\
// Auto-generated xe-fuse standalone kernel benchmark
// Op: {spec["name"]}
// Generated: {datetime.now().isoformat()}
// Description: {spec["evt_description"]}

#include "xe-fuse/standalone/ops.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/command_line.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <iostream>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

struct Options {{
  int m = {m}, n = {n}, l = 1;
  int iterations = {iters};

  void parse(int argc, char const** args) {{
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, {m});
    cmd.get_cmd_line_argument("n", n, {n});
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, {iters});
  }}
}};

int main(int argc, const char** argv) {{
  Options opts;
  opts.parse(argc, argv);

  int M = opts.m, N = opts.n, L = opts.l;
  auto q = compat::get_default_queue();

  cutlass::DeviceAllocation<bf16> block_D(static_cast<size_t>(M) * N * L);
  initialize_block(block_D, 2023);

{aux_alloc}

  // Warmup
  for (int i = 0; i < 5; ++i) {{
  {op_code}
    q.wait();
  }}

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < opts.iterations; ++i) {{
  {op_code}
    q.wait();
  }}

  float time_s = timer.seconds() / opts.iterations;
  size_t bytes_rw = static_cast<size_t>(M) * N * L * sizeof(bf16) * 2;
  double bw_gb = bytes_rw / (time_s * 1e9);

  std::cout << "Problem Size: " << M << 'x' << N << 'x' << L << std::endl;
  printf("{spec["name"]}: %6.4f ms  %.1f GB/s  (%.1f MB r+w)\\n",
         time_s * 1000, bw_gb, bytes_rw / (1024.0 * 1024.0));

  return 0;
}}
"""


STANDALONE_PRESETS = {
    "sa_scale_rows": {
        "name": "SA_ScaleRows",
        "evt_description": "D[m,n] *= scale[m] (standalone RMSNorm scaling)",
        "standalone": True,
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42}
        ],
        "op_code": [
            "xe_fuse::standalone::scale_rows(q, block_D.get(), block_scale.get(), M, N, L);"
        ]
    },
    "sa_residual_gamma": {
        "name": "SA_ResidualGamma",
        "evt_description": "D[m,n] = gamma[n] * (D + residual) (standalone)",
        "standalone": True,
        "aux_data": [
            {"name": "residual", "type": "bf16", "shape": "M * N * L", "init_seed": 2021},
            {"name": "gamma", "type": "float", "shape": "N * L", "init_seed": 99}
        ],
        "op_code": [
            "xe_fuse::standalone::residual_gamma(q, block_D.get(), block_residual.get(),",
            "    block_gamma.get(), M, N, L);"
        ]
    },
    "sa_swiglu": {
        "name": "SA_SwiGLU",
        "evt_description": "D = SwiGLU(D) pairwise (standalone)",
        "standalone": True,
        "aux_data": [],
        "op_code": [
            "xe_fuse::standalone::swiglu(q, block_D.get(), M, N, L);"
        ]
    },
    "sa_rope_scaled": {
        "name": "SA_RoPEScaled",
        "evt_description": "D = RoPE(D * scale[m], cos_sin) (standalone)",
        "standalone": True,
        "aux_data": [
            {"name": "scale", "type": "float", "shape": "M * L", "init_seed": 42},
            {"name": "cos_sin", "type": "float", "shape": "M * N * L", "init_seed": 2024},
            {"name": "tmp", "type": "bf16", "shape": "M * N * L", "init_seed": 0}
        ],
        "op_code": [
            "q.memcpy(block_tmp.get(), block_D.get(), static_cast<size_t>(M) * N * L * sizeof(bf16));",
            "q.wait();",
            "xe_fuse::standalone::rope_scaled(q, block_D.get(), block_tmp.get(),",
            "    block_scale.get(), block_cos_sin.get(), M, N, L);"
        ]
    }
}


def main():
    parser = argparse.ArgumentParser(description="xe-fuse kernel generator")
    all_presets = {**PRESETS, **STANDALONE_PRESETS}

    parser.add_argument("--spec", help="Path to kernel spec JSON file")
    parser.add_argument("--preset", choices=list(all_presets.keys()),
                        help="Use a built-in kernel preset")
    parser.add_argument("--output", "-o", help="Output .cpp path")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tile", type=str, default=None,
                        help="Override tile shape, e.g. '128x256x32' or 'auto'")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        print("GEMM epilogue presets:")
        for name, spec in PRESETS.items():
            print(f"  {name:16s}  {spec['evt_description']}")
        print("\nStandalone presets:")
        for name, spec in STANDALONE_PRESETS.items():
            print(f"  {name:16s}  {spec['evt_description']}")
        return

    if not args.output and not args.list_presets:
        parser.error("--output is required unless using --list-presets")
        return

    if args.preset:
        spec = all_presets[args.preset]
    elif args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
    else:
        parser.error("Provide --preset or --spec")
        return

    # Tile shape override: explicit, auto, or default from preset
    if args.tile:
        if args.tile == "auto":
            from tile_selector import select_tile
            kernel_tag = args.preset.split("_")[0] if args.preset else "bare"
            spec["tile_shape"] = select_tile(args.m, args.n, args.k, kernel_tag)
        else:
            parts = args.tile.replace("x", ", _").lstrip("_")
            spec["tile_shape"] = f"_{parts}"

    defaults = {"m": args.m, "n": args.n, "k": args.k, "iterations": args.iterations}
    if spec.get("standalone"):
        code = generate_standalone_cpp(spec, defaults)
    else:
        code = generate_cpp(spec, defaults)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(code)

    print(f"Generated: {args.output}")
    print(f"  Kernel: {spec['name']}")
    print(f"  Tile:   {spec.get('tile_shape', '_256, _256, _32')}")
    print(f"  EVT:    {spec['evt_description']}")


if __name__ == "__main__":
    main()
