// xe-fuse test: QK norm + RoPE
// launch_qk_norm_rope: per-head RMSNorm followed by RoPE in a single kernel pass.

#include "xe-fuse/kernels/qk_norm_rope.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>
#include <vector>
#include <cstdio>

using ElementQ = cutlass::bfloat16_t;
using ElementF = float;

struct Options {
  int seq_len   = 4096;
  int num_heads = 32;
  int head_dim  = 128;
  int batch     = 1;
  int iterations = 100;
  int verify = 1;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("seq_len",    seq_len,   4096);
    cmd.get_cmd_line_argument("num_heads",  num_heads, 32);
    cmd.get_cmd_line_argument("head_dim",   head_dim,  128);
    cmd.get_cmd_line_argument("batch",      batch,     1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify",     verify,     1);
  }
};

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  int M         = opts.batch * opts.seq_len;
  int num_heads = opts.num_heads;
  int head_dim  = opts.head_dim;

  size_t input_size  = static_cast<size_t>(M) * num_heads * head_dim;
  size_t cos_sin_size = static_cast<size_t>(M) * head_dim;
  size_t gamma_size  = static_cast<size_t>(head_dim);

  cutlass::DeviceAllocation<ElementQ> block_in(input_size);
  cutlass::DeviceAllocation<ElementQ> block_out(input_size);
  cutlass::DeviceAllocation<ElementQ> block_ref(input_size);
  cutlass::DeviceAllocation<ElementF> block_gamma(gamma_size);
  cutlass::DeviceAllocation<ElementF> block_cos_sin(cos_sin_size);

  initialize_block(block_in, 2025);
  sycl::queue q = compat::get_default_queue();

  // gamma: uniform (0.5, 1.5)
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::vector<float> h(gamma_size);
    for (auto& v : h) v = dist(rng);
    compat::get_default_queue().memcpy(block_gamma.get(), h.data(), h.size() * sizeof(float));
  }

  // cos_sin: realistic RoPE frequencies
  {
    std::vector<float> h(cos_sin_size);
    for (int tok = 0; tok < M; ++tok) {
      for (int k = 0; k < head_dim / 2; ++k) {
        float freq  = 1.0f / std::pow(10000.f, 2.f * k / static_cast<float>(head_dim));
        float angle = static_cast<float>(tok) * freq;
        int base = tok * head_dim;
        h[base + 2 * k]     = std::cos(angle);
        h[base + 2 * k + 1] = std::sin(angle);
      }
    }
    compat::get_default_queue().memcpy(block_cos_sin.get(), h.data(), h.size() * sizeof(float));
  }
  compat::wait();

  xe_fuse::launch_qk_norm_rope<ElementQ>(
      q, block_in.get(), block_out.get(),
      block_gamma.get(), block_cos_sin.get(),
      M, num_heads, head_dim);
  compat::wait();

  if (opts.verify) {
    // CPU reference
    std::vector<ElementQ> h_in(input_size);
    std::vector<float>    h_gamma(gamma_size);
    std::vector<float>    h_cos_sin(cos_sin_size);
    compat::get_default_queue().memcpy(h_in.data(),      block_in.get(),      input_size  * sizeof(ElementQ)).wait();
    compat::get_default_queue().memcpy(h_gamma.data(),   block_gamma.get(),   gamma_size  * sizeof(float)).wait();
    compat::get_default_queue().memcpy(h_cos_sin.data(), block_cos_sin.get(), cos_sin_size * sizeof(float)).wait();

    std::vector<ElementQ> h_ref(input_size);
    constexpr float eps = 1e-6f;

    for (int tok = 0; tok < M; ++tok) {
      for (int h = 0; h < num_heads; ++h) {
        int row_base = tok * num_heads * head_dim + h * head_dim;
        int cs_base  = tok * head_dim;

        // RMSNorm
        float sum_sq = 0.f;
        for (int d = 0; d < head_dim; ++d) {
          float v = static_cast<float>(h_in[row_base + d]);
          sum_sq += v * v;
        }
        float rstd = 1.f / std::sqrt(sum_sq / head_dim + eps);

        // normalize + gamma
        std::vector<float> normed(head_dim);
        for (int d = 0; d < head_dim; ++d)
          normed[d] = static_cast<float>(h_in[row_base + d]) * rstd * h_gamma[d];

        // RoPE
        for (int d = 0; d < head_dim; d += 2) {
          float x0 = normed[d], x1 = normed[d + 1];
          float cos_v = h_cos_sin[cs_base + d];
          float sin_v = h_cos_sin[cs_base + d + 1];
          h_ref[row_base + d]     = static_cast<ElementQ>( x0 * cos_v + x1 * sin_v);
          h_ref[row_base + d + 1] = static_cast<ElementQ>(-x0 * sin_v + x1 * cos_v);
        }
      }
    }
    compat::get_default_queue().memcpy(block_ref.get(), h_ref.data(), input_size * sizeof(ElementQ)).wait();

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref.get(), block_out.get(), block_out.size(),
        ElementQ(0.05f), ElementQ(0.05f));

    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return 1;
  } else {
    std::cout << "Disposition is skipped." << std::endl;
  }

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i)
      xe_fuse::launch_qk_norm_rope<ElementQ>(
          q, block_in.get(), block_out.get(),
          block_gamma.get(), block_cos_sin.get(),
          M, num_heads, head_dim);
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double bytes = 2.0 * input_size * sizeof(ElementQ);  // 1 read + 1 write
    printf("Problem: M=%d num_heads=%d head_dim=%d\n", M, num_heads, head_dim);
    printf("qk_norm_rope: [%4.3f]GB/s  (%6.4f)ms\n",
           bytes * 1e-9 / time_s, time_s * 1000);
  }

  return 0;
}
