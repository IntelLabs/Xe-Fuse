# xe-fuse tests
# Uses xe_fuse_add_executable() from cmake/XeFuseHelpers.cmake

xe_fuse_add_executable(test_k0a          test_k0a.cpp)
xe_fuse_add_executable(test_k0a_colred   test_k0a_colred.cpp)
xe_fuse_add_executable(test_k1           test_k1.cpp)
xe_fuse_add_executable(test_k2           test_k2.cpp)
xe_fuse_add_executable(test_k3           test_k3.cpp)
xe_fuse_add_executable(test_k4           test_k4.cpp)
xe_fuse_add_executable(test_k5           test_k5.cpp)
xe_fuse_add_executable(test_k5_fused     test_k5_fused.cpp)
xe_fuse_add_executable(test_k5_fused_atomic test_k5_fused_atomic.cpp)
xe_fuse_add_executable(test_k8           test_k8.cpp)
xe_fuse_add_executable(test_k9           test_k9.cpp)
xe_fuse_add_executable(test_k9_fused     test_k9_fused.cpp)
xe_fuse_add_executable(test_k9_fused_atomic test_k9_fused_atomic.cpp)

xe_fuse_add_executable(test_qk_norm_rope    test_qk_norm_rope.cpp)
xe_fuse_add_executable(test_k0g             test_k0g.cpp)
xe_fuse_add_executable(test_fp8_k2          test_fp8_k2.cpp)

xe_fuse_add_executable(bench_qk_norm_rope    bench_qk_norm_rope.cpp)
xe_fuse_add_executable(bench_norm_quantize   bench_norm_quantize.cpp)
xe_fuse_add_executable(bench_swiglu_requant  bench_swiglu_requant.cpp)
