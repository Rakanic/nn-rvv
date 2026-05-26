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

#################################
# Options
#################################

# Compile-time hart count for the threading runtime.
#   1            → no threading, parallel_for is an inline call.
#   2 or higher  → work-stealing runtime is compiled in. Requires the parent
#                  project to provide CLINT MSIP + mhartid headers (clint.h,
#                  riscv.h via the parent's `clint` + `rocketcore` targets).
set(NN_RVV_N_HARTS "1" CACHE STRING
    "Number of harts nn-rvv kernels distribute work across (1 disables threading)")

option(NN_RVV_MAX_PERF
  "Compile nn-rvv kernels with aggressive optimization flags" ON)

#################################
# Sources
#################################

file(GLOB_RECURSE NN_RVV_C_SRC   CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.c")
file(GLOB_RECURSE NN_RVV_ASM_SRC CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.S")

# Threading runtime sources. Compiled in unconditionally — when N_HARTS==1
# the file bodies are #if'd out, so the cost is just preprocessing.
set(NN_RVV_THREADING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/threading/hthread.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/threading/parallel_for.c")

add_library(nnrvv STATIC ${NN_RVV_C_SRC} ${NN_RVV_ASM_SRC} ${NN_RVV_THREADING_SRC})

#################################
# Include paths
#################################

# Public include: consumers `#include "nn_rvv/layers.h"` and "nn_rvv/threading.h"
target_include_directories(nnrvv PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Private include: internal headers like ops/matmul/matmul.h, threading/hthread.h
target_include_directories(nnrvv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                          "${CMAKE_CURRENT_SOURCE_DIR}/src/threading")

#################################
# Compile flags
#################################

# Propagate the hart count to every consumer (kernels and downstream code both
# need to see the same value of NN_RVV_N_HARTS).
target_compile_definitions(nnrvv PUBLIC NN_RVV_N_HARTS=${NN_RVV_N_HARTS})

if(NN_RVV_MAX_PERF)
  target_compile_options(nnrvv PRIVATE -O3 -funroll-loops -fno-math-errno -fno-trapping-math)
endif()

#################################
# Threading runtime dependencies
#################################

if(NN_RVV_N_HARTS GREATER 1)
  # The work-stealing runtime needs:
  #   - CLINT MSIP writes (the `((CLINT_Type *)CLINT_BASE)` macro lives in
  #     chip_config.h, which the parent project's `chip-config` target
  #     publicly exposes — and which itself transitively pulls in `clint` and
  #     `rocketcore` for the type definitions and the READ_CSR macro).
  # Linking against just `clint` + `rocketcore` is not enough: `clint`
  # declares the CLINT_Type struct but not the `CLINT` instance pointer.
  if(TARGET chip-config)
    target_link_libraries(nnrvv PUBLIC chip-config)
  elseif(TARGET clint AND TARGET rocketcore)
    # Allow consumers that supply equivalents through `clint` + `rocketcore`
    # to opt in here too, but warn since they must also ensure `CLINT` is
    # actually defined somewhere on the include path.
    target_link_libraries(nnrvv PUBLIC clint rocketcore)
    message(WARNING
      "nn-rvv: linking against `clint` + `rocketcore` without `chip-config`. "
      "Make sure the `CLINT` MMIO macro (typically defined in chip_config.h) "
      "is visible to nn-rvv's threading sources.")
  else()
    message(FATAL_ERROR
      "NN_RVV_N_HARTS=${NN_RVV_N_HARTS} requires the parent project to provide "
      "CLINT MSIP + mhartid access. Expected a `chip-config` target (preferred) "
      "or at minimum `clint` + `rocketcore` targets. Either set NN_RVV_N_HARTS=1, "
      "or add those targets before adding the nn-rvv subdirectory.")
  endif()
  message(STATUS "nn-rvv: threading runtime ON (NN_RVV_N_HARTS=${NN_RVV_N_HARTS})")
else()
  message(STATUS "nn-rvv: threading runtime OFF (single-hart build)")
endif()
