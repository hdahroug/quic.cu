// Machete Hopper WGMMA mixed-precision GEMM and prepacking wrapper.
#include "quantization/machete/machete_mm_launcher.cuh"
#include "quantization/machete/machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

#include <torch/extension.h>

namespace machete {

using namespace vllm;

struct Sch_small {
  using TileShapeNM = cute::Shape<cute::_128, cute::_16>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using TileScheduler = cutlass::gemm::StreamKScheduler;
};

struct Sch_med {
  using TileShapeNM = cute::Shape<cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using TileScheduler = cutlass::gemm::StreamKScheduler;
};

struct Sch_big {
  using TileShapeNM = cute::Shape<cute::_128, cute::_256>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using TileScheduler = cutlass::gemm::StreamKScheduler;
};

template <typename Sch>
using Kernel_u4 = MacheteKernelTemplate<
    cutlass::half_t, cutlass::uint4b_t, cutlass::half_t, float,
    cutlass::half_t, cutlass::half_t, void, void,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative, Sch>;

torch::Tensor impl_u4_small(MMArgs args) {
  return run_impl<Kernel_u4<Sch_small>>(args);
}
torch::Tensor impl_u4_med(MMArgs args) {
  return run_impl<Kernel_u4<Sch_med>>(args);
}
torch::Tensor impl_u4_big(MMArgs args) {
  return run_impl<Kernel_u4<Sch_big>>(args);
}

torch::Tensor mm_dispatch(MMArgs args) {
  TORCH_CHECK(args.A.is_cuda() && args.B.is_cuda(), "Machete tensors must be CUDA");
  TORCH_CHECK(args.A.dtype() == at::kHalf, "Machete expects f16 activations");
  TORCH_CHECK(args.b_type == vllm::kU4, "Machete expects uint4 weights");
  TORCH_CHECK(args.maybe_group_scales && args.maybe_group_zeros,
              "Machete group scales and zero-points are required");
  TORCH_CHECK(args.maybe_group_scales->dtype() == at::kHalf &&
                  args.maybe_group_zeros->dtype() == at::kHalf,
              "Machete group scales and zero-points must be f16");
  int64_t M = args.A.size(0);
  if (M <= 16) {
    return impl_u4_small(args);
  }
  if (M <= 128) {
    return impl_u4_med(args);
  }
  return impl_u4_big(args);
}

torch::Tensor prepack_B_dispatch(PrepackBArgs args) {
  using Layout = PrepackedLayoutBTemplate<
      cutlass::half_t, cutlass::uint4b_t, cutlass::half_t, float,
      cutlass::layout::ColumnMajor,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  return prepack_impl<Layout>(args.B);
}

}

torch::Tensor py_machete_mm(torch::Tensor A, torch::Tensor B_q,
                            torch::Tensor group_scales,
                            torch::Tensor group_zeros, int64_t group_size) {
  return machete::mm_dispatch({.A = A,
                               .B = B_q,
                               .b_type = vllm::kU4,
                               .maybe_out_type = std::nullopt,
                               .maybe_group_scales = group_scales,
                               .maybe_group_zeros = group_zeros,
                               .maybe_group_size = group_size,
                               .maybe_channel_scales = std::nullopt,
                               .maybe_token_scales = std::nullopt,
                               .maybe_schedule = std::nullopt});
}

torch::Tensor py_machete_prepack_B(torch::Tensor B) {
  return machete::prepack_B_dispatch(
      {.B = B,
       .a_type = at::kHalf,
       .b_type = vllm::kU4,
       .maybe_group_scales_type = at::kHalf});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("machete_mm", &py_machete_mm);
  m.def("machete_prepack_B", &py_machete_prepack_B);
}
