# nn-rvv: RVV-targeted neural-network kernels.
#
# Builds a static library `nnrvv` with public header `nn_rvv/layers.h`.
#
# Usable in two modes:
#   (1) As a subdirectory of a larger CMake project (the parent supplies the
#       toolchain, compile flags, and linker setup). The parent only needs to
#       `add_subdirectory(...)` and then `target_link_libraries(<tgt> nnrvv)`.
#   (2) Standalone, configured directly via cmake. In that mode, the consumer
#       still owns the toolchain choice; nn-rvv just produces libnnrvv.a.
#
# This file deliberately does not set CMAKE_C_COMPILER, march/mabi, or linker
# flags — that's the caller's responsibility, since the right values depend on
# the target platform (Saturn/Spike/etc.).

cmake_minimum_required(VERSION 3.10)

# Only declare a project() when we're the top-level build; otherwise we inherit
# the parent's project settings (and toolchain).
if(NOT DEFINED PROJECT_NAME OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(nn_rvv LANGUAGES ASM C)
endif()

# Collect all C and ASM sources under src/.
file(GLOB_RECURSE NN_RVV_C_SRC   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.c")
file(GLOB_RECURSE NN_RVV_ASM_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.S")

add_library(nnrvv STATIC ${NN_RVV_C_SRC} ${NN_RVV_ASM_SRC})

# Public include: consumers `#include "nn_rvv/layers.h"`
target_include_directories(nnrvv PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Private include: internal headers like ops/matmul/matmul.h
target_include_directories(nnrvv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

option(NN_RVV_MAX_PERF
  "Compile nn-rvv kernels with aggressive optimization flags" ON)

if(NN_RVV_MAX_PERF)
  target_compile_options(nnrvv PRIVATE -O3 -funroll-loops -fno-math-errno -fno-trapping-math)
endif()
