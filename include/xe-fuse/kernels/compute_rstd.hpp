#pragma once

#include <sycl/sycl.hpp>

namespace xe_fuse {

// Standalone rstd reduction kernel.
// Computes R[m] = 1 / sqrt( mean_n(X[m,n]^2) + eps ) for each row.
// Uses sub-group reduction across N.

template <typename ElementInput, typename ElementOutput = float>
void launch_compute_rstd(
    sycl::queue& q,
    ElementInput const* input_ptr,
    ElementOutput* rstd_ptr,
    int M, int N, int L,
    float eps = 1e-6f)
{
  constexpr int SG_SIZE = 16;
  int work_groups = M * L;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(work_groups * SG_SIZE, SG_SIZE),
      [=](sycl::nd_item<1> item) {
        int row = item.get_group(0);
        int lane = item.get_local_id(0);

        float sum_sq = 0.0f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float val = static_cast<float>(input_ptr[row * N + col]);
          sum_sq += val * val;
        }

        auto sg = item.get_sub_group();
        for (int offset = SG_SIZE / 2; offset > 0; offset /= 2) {
          sum_sq += sycl::shift_group_left(sg, sum_sq, offset);
        }

        if (lane == 0) {
          float mean_sq = sum_sq / static_cast<float>(N);
          rstd_ptr[row] = static_cast<ElementOutput>(
              1.0f / sycl::sqrt(mean_sq + eps));
        }
      }
    );
  });
}

}  // namespace xe_fuse
