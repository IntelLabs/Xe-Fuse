# xe_fuse_add_executable(NAME source1 [source2...] [DEFS def1 def2...])
#
# Convenience wrapper used by tests/ and examples/ that:
#   1. Creates an executable linked to _xe_fuse_test_harness
#      (which pulls in xe_fuse::xe_fuse, CUTLASS, DPCPP::DPCPP,
#       tools/util/include, MKL, and sycl_common.hpp)
#   2. Calls add_sycl_to_target() so icpx emits correct -fsycl / -fsycl-targets flags
#   3. Applies any per-target compile definitions supplied via DEFS
#
function(xe_fuse_add_executable NAME)
  cmake_parse_arguments(_XFE "" "" "DEFS" ${ARGN})
  set(_sources ${_XFE_UNPARSED_ARGUMENTS})

  add_executable(${NAME} ${_sources})
  target_link_libraries(${NAME} PRIVATE _xe_fuse_test_harness)
  add_sycl_to_target(TARGET ${NAME} SOURCES ${_sources})

  if(_XFE_DEFS)
    target_compile_definitions(${NAME} PRIVATE ${_XFE_DEFS})
  endif()
endfunction()
