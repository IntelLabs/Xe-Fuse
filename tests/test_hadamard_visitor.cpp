// xe-fuse test: XeHadamardCompute<16> correctness
//
// Directly instantiates the XeHadamardCompute visitor inside a SYCL kernel
// and compares each group of 16 outputs to a CPU Walsh-Hadamard Transform
// reference.
//
// The test kernel uses nd_range<1> with local size 16 (one sub-group per group),
// which matches how shfl_xor_sync operates inside the visitor (uses get_nd_item<1>).
//
// Correctness check: relative error < 1e-5 for all elements.
// In practice the WHT butterfly on float32 is near-exact for values < 2^20.

#include "xe-fuse/visitors/xe_hadamard_compute.hpp"

#include "cutlass/array.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <cmath>
#include <random>
#include <vector>

// ── CPU WHT reference ─────────────────────────────────────────────────────────

// Iterative in-place WHT on x[0..n-1], normalized by 1/sqrt(n).
// Matches the butterfly convention in XeHadamardCompute:
//   stage k (stride = 2^k):
//     index with bit-k == 0 → even lane: a + b
//     index with bit-k == 1 → odd lane:  a - b
static void wht_reference(float* x, int n) {
  for (int step = 1; step < n; step <<= 1) {
    for (int i = 0; i < n; i += step * 2) {
      for (int j = 0; j < step; ++j) {
        float a = x[i + j];
        float b = x[i + step + j];
        x[i + j]        = a + b;
        x[i + step + j] = a - b;
      }
    }
  }
  float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
  for (int i = 0; i < n; ++i) x[i] *= inv_sqrt_n;
}

// ── Device kernel ─────────────────────────────────────────────────────────────

// Each work-group has 16 work items (= 1 sub-group on XeGPU B70).
// Each group processes 16 consecutive float elements.
static sycl::event launch_hadamard_kernel(float const* d_input,
                                          float* d_output,
                                          int num_groups)
{
  return compat::get_default_queue().submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(num_groups) * 16, 16),
      [=](sycl::nd_item<1> item) {
        int group_id = static_cast<int>(item.get_group(0));
        int lane     = static_cast<int>(item.get_local_id(0));
        int idx      = group_id * 16 + lane;

        // Single-element fragment
        constexpr int kFragSize = 1;
        cutlass::Array<float, kFragSize> frg_input;
        frg_input[0] = d_input[idx];

        cutlass::Array<float, kFragSize> frg_acc;
        frg_acc[0] = 0.0f;

        // Apply XeHadamardCompute<16>: 4-stage butterfly across the sub-group
        xe_fuse::XeHadamardCompute<16>::ConsumerStoreCallbacks cb{};
        auto result = cb.visit(frg_acc, /*epi_v=*/0, /*epi_m=*/0, /*epi_n=*/0, frg_input);

        d_output[idx] = result[0];
      }
    );
  });
}

// ── Options ──────────────────────────────────────────────────────────────────

struct Options {
  int num_groups = 4096;  // number of 16-element WHT groups (total_elems = groups * 16)
  int iterations = 100;
  int verify     = 1;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    cmd.get_cmd_line_argument("groups",     num_groups, 4096);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify",     verify,     1);
  }
};

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  int total_elems = opts.num_groups * 16;

  // Initialize random host input in [-1, 1]
  std::vector<float> h_input(total_elems);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_input) v = dist(rng);

  // CPU WHT reference: apply separately to each group of 16
  std::vector<float> h_ref = h_input;
  for (int g = 0; g < opts.num_groups; ++g)
    wht_reference(h_ref.data() + g * 16, 16);

  // Device buffers
  cutlass::device_memory::allocation<float> d_input(total_elems);
  cutlass::device_memory::allocation<float> d_output(total_elems);

  compat::get_default_queue().memcpy(d_input.get(), h_input.data(),
                                     total_elems * sizeof(float));
  compat::wait();

  // Run (warm-up + result)
  launch_hadamard_kernel(d_input.get(), d_output.get(), opts.num_groups);
  compat::wait();

  // Correctness check
  bool passed = true;
  if (opts.verify) {
    std::vector<float> h_output(total_elems);
    compat::get_default_queue().memcpy(h_output.data(), d_output.get(),
                                       total_elems * sizeof(float));
    compat::wait();

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int   num_fail    = 0;

    for (int i = 0; i < total_elems; ++i) {
      float abs_err = std::abs(h_output[i] - h_ref[i]);
      float ref_mag = std::abs(h_ref[i]) + 1e-6f;
      float rel_err = abs_err / ref_mag;
      max_abs_err   = std::max(max_abs_err, abs_err);
      max_rel_err   = std::max(max_rel_err, rel_err);
      if (rel_err > 1e-5f) ++num_fail;
    }

    passed = (num_fail == 0);
    printf("Correctness: %s\n", passed ? "PASSED" : "FAILED");
    printf("  num_groups=%d  total_elems=%d\n", opts.num_groups, total_elems);
    printf("  max_abs_err=%.2e  max_rel_err=%.2e  failures=%d\n",
           max_abs_err, max_rel_err, num_fail);
  }

  // Benchmark
  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i)
      launch_hadamard_kernel(d_input.get(), d_output.get(), opts.num_groups);
    compat::wait();

    float time_ms = timer.milliseconds() / static_cast<float>(opts.iterations);
    float gb      = static_cast<float>(total_elems) * sizeof(float) * 2.0f / 1e9f;
    float bw_gbs  = gb / (time_ms * 1e-3f);

    printf("Throughput: %.3f ms/iter  %.1f GB/s  (%d iters)\n",
           time_ms, bw_gbs, opts.iterations);
  }

  return passed ? 0 : 1;
}
