#pragma once

// Single-template-parameter reduce functors for XeColReduction.
//
// XeColReduction requires `template <class> class Fn` template template parameters.
// These wrappers satisfy that interface.

#include "cutlass/functional.h"
#include "cutlass/array.h"
#include <sycl/sycl.hpp>
#include <type_traits>

namespace xe_fuse {

template <class T> struct reduce_max : cutlass::maximum<T, false> {};
template <class T> struct reduce_sum : cutlass::plus<T>     {};

// Numerically stable log-sum-exp: logsumexp(a, b) = max(a,b) + log(exp(a-max) + exp(b-max))
// Identity element: -infinity. Use XeColReduction with reduction_identity = -inf
// to compute per-tile lse = log(sum_n exp(D[m,n])) without overflow.
//
// Single primary template with if constexpr handles float and Array<float,N>.
// Avoids explicit full specializations which icpx/IGC may not instantiate in device code.
template <class T>
struct reduce_logsumexp {
  CUTLASS_DEVICE T operator()(T const& a, T const& b) const {
    if constexpr (std::is_same_v<T, float>) {
      float m = sycl::fmax(a, b);
      if (m == -INFINITY) return m;
      return m + sycl::log(sycl::exp(a - m) + sycl::exp(b - m));
    } else {
      // Array<float, N> path: element-wise logsumexp
      T result;
      reduce_logsumexp<float> fn;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < T::kElements; ++i) result[i] = fn(a[i], b[i]);
      return result;
    }
  }
};

// Atomic log-sum-exp for GmemReduceFn in XeColReduction<IsAtomic=true>.
// Uses a CAS loop to atomically combine per-CTA partial logsumexp into the final
// lse output buffer. Circumvents the FinalReduction=false copy_aligned bug where
// only 8 M-rows per CTA are written due to N-stride=0 in the partition.
// Pre-initialize the output buffer to -infinity (reduction identity).
template <class T>
struct atomic_reduce_logsumexp {
  CUTLASS_DEVICE T operator()(T* ptr, T const& val) const {
    static_assert(std::is_same_v<T, float>,
                  "atomic_reduce_logsumexp only supports float");
    sycl::atomic_ref<float,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ar(*ptr);
    float old = ar.load();
    float new_val;
    do {
      if (val == -INFINITY) return old;
      float m = sycl::fmax(old, val);
      new_val = (m == -INFINITY) ? val
                                 : m + sycl::log(sycl::exp(old - m) + sycl::exp(val - m));
    } while (!ar.compare_exchange_weak(old, new_val));
    return old;
  }
};

}  // namespace xe_fuse

// Mark atomic_reduce_logsumexp as an atomic functor so XeColReduction sets IsAtomic=true
// and takes the filter_zeros path that correctly writes all M rows.
namespace cutlass {
template <class T>
struct is_atomic<xe_fuse::atomic_reduce_logsumexp<T>> : platform::true_type {};
}  // namespace cutlass
