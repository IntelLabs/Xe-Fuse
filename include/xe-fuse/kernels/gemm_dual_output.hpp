#pragma once

// K0: gemm_dual_output — D = gamma * (acc + residual), aux = acc + residual
//
// Split-tree pattern: the input tree (acc + residual) is evaluated once, then:
//   1. Stored to an auxiliary buffer (raw sum, for compute_rstd)
//   2. Multiplied by gamma → stored as primary output D
//
// EVT tree:
//   Sm90SplitTreeVisitor<
//     XeEVT<Add, AccFetch, AuxLoad<residual>>,           // InputTree: acc + residual
//     XeEVT<Mul, AccFetch, RowBroadcast<gamma>>,         // OutputTree: input * gamma → D
//     XeEVT<AuxStore<bf16>, AccFetch>                     // AuxOutTree: store raw sum
//   >

#include "xe-fuse/builder/epilogue_builder.hpp"

namespace xe_fuse {

template <
  typename ElementA_        = cutlass::bfloat16_t,
  typename ElementB_        = cutlass::bfloat16_t,
  typename ElementD_        = cutlass::bfloat16_t,
  typename ElementResidual_ = cutlass::bfloat16_t,
  typename ElementGamma_    = float,
  typename ElementAcc_      = float,
  typename ElementCompute_  = float,
  typename TileShape_       = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmDualOutput {
  using ElementA        = ElementA_;
  using ElementB        = ElementB_;
  using ElementD        = ElementD_;
  using ElementResidual = ElementResidual_;
  using ElementGamma    = ElementGamma_;
  using ElementAcc      = ElementAcc_;
  using ElementCompute  = ElementCompute_;
  using TileShape       = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC   = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD   = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideAux = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // InputTree: acc + residual
  using InputTree = builder::AddResidual<ElementResidual, ElementCompute>;

  // OutputTree: input * gamma[n] (receives input tree result as accumulator)
  using OutputTree = builder::EVT<
      builder::MulOp<ElementCompute, ElementCompute>,
      builder::Acc,
      builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>
  >;

  // AuxOutTree: store raw sum to auxiliary buffer
  using AuxOutTree = builder::EVT<
      builder::AuxStore<ElementD, StrideAux>,
      builder::Acc
  >;

  // Split-tree: evaluate InputTree, store to aux, then apply OutputTree
  using EVT = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
      InputTree, OutputTree, AuxOutTree>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, 8,
      ElementD, StrideD, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT
    >::CollectiveOp;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAcc,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Helper to build EVT arguments
  static typename EVT::Arguments make_evt_args(
      ElementResidual const* residual_ptr, StrideD stride_residual,
      ElementGamma const* gamma_ptr, int N,
      ElementD* aux_ptr, StrideAux stride_aux) {

    // InputTree: Add<Acc, AuxLoad<residual>>
    typename builder::Acc::Arguments accum_args{};
    typename builder::AuxLoad<ElementResidual>::Arguments res_args;
    res_args.ptr_aux = residual_ptr;
    res_args.null_default = ElementResidual(0);
    res_args.dAux = stride_residual;
    typename builder::AddOp<>::Arguments add_args{};
    typename InputTree::Arguments input_args{accum_args, res_args, add_args};

    // AuxOutTree: EVT<AuxStore, Acc>
    typename builder::Acc::Arguments aux_acc_args{};
    typename builder::AuxStore<ElementD, StrideAux>::Arguments aux_store_args;
    aux_store_args.ptr_aux = aux_ptr;
    aux_store_args.null_default = ElementD(0);
    aux_store_args.dAux = stride_aux;
    typename AuxOutTree::Arguments aux_out_args{aux_acc_args, aux_store_args};

    // OutputTree: Mul<Acc, RowBroadcast<gamma>>
    typename builder::Acc::Arguments out_acc_args{};
    typename builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>::Arguments gamma_args;
    gamma_args.ptr_row = gamma_ptr;
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};
    typename builder::MulOp<>::Arguments mul_args{};
    typename OutputTree::Arguments output_args{out_acc_args, gamma_args, mul_args};

    // SplitTreeVisitor args order: InputTree, AuxOutTrees..., OutputTree
    return {input_args, aux_out_args, output_args};
  }
};

}  // namespace xe_fuse
