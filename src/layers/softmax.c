#include "nn_rvv/layers.h"
#include "ops/reduce/reduce.h"
#include <string.h>
#include "riscv_vector.h"
#include <math.h>
#include "ops/ara/exp.h"

/* In-place 1D softmax over n floats. Three vectorized passes (max,
 * exp(x-max)+sum, scale). Faster than softmax_vec(channels=n, innerSize=1)
 * which strip-mines on innerSize and degenerates to scalar at vl=1. */
void softmax_f32(float *x, size_t n) {
    if (n == 0) return;

    float max_val = max_f32(x, n);

    float sum = 0.0f;
    {
        size_t remaining = n;
        float *p = x;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(p, vl);
            vx = __riscv_vfsub_vf_f32m4(vx, max_val, vl);
            vx = __exp_f32m4(vx, vl);
            __riscv_vse32_v_f32m4(p, vx, vl);
            vfloat32m1_t vr = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vr = __riscv_vfredusum_vs_f32m4_f32m1(vx, vr, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(vr);
            p += vl; remaining -= vl;
        }
    }

    float inv = 1.0f / sum;
    {
        size_t remaining = n;
        float *p = x;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(p, vl);
            vx = __riscv_vfmul_vf_f32m4(vx, inv, vl);
            __riscv_vse32_v_f32m4(p, vx, vl);
            p += vl; remaining -= vl;
        }
    }
}

void softmax_vec(
    const float *i, 
    float *o, 
    size_t channels,
    size_t innerSize) {


  size_t avl = innerSize;
  size_t vl;

  // Stripmining pointers
  float *_i = (float *)i;
  float *_o = (float *)o;
  // Channel pointers
  float *__i = (float *)i;
  float *__o = (float *)o;

  // Vector registers
  vfloat32m1_t max_chunk_v;
  vfloat32m1_t buf_chunk_v;
  vfloat32m1_t num_chunk_v;
  vfloat32m1_t den_chunk_v;
  vfloat32m1_t res_chunk_v;

  // Stripmine on innerSize
  for (vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl) {

    vl = __riscv_vsetvl_e32m1(avl);

    /*
      Calculate the maximum along the channel dimension
    */

    // Initialize the max vector
    max_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
    // Bump the pointer
    __i += innerSize;
    for (size_t ch = 1; ch < channels; ++ch) {
      // Load a chunk of the input vector
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      // Bump the channel pointer
      __i += innerSize;
      // Calculate the elm-wise maximum between the two chunks
      max_chunk_v = __riscv_vfmax_vv_f32m1(max_chunk_v, buf_chunk_v, vl);
    }
    // Restore the channel pointer
    __i = _i;

    /*
      Fetch, subtract, exponentiate along the channel dimension
    */

    // Initialize accumulator
    den_chunk_v = __riscv_vfmv_v_f_f32m1(0, vl);
    for (size_t ch = 0; ch < channels; ++ch) {
      // Fetch one chunk from channel ch
      buf_chunk_v = __riscv_vle32_v_f32m1(__i, vl);
      // Subtract the maximum
      buf_chunk_v = __riscv_vfsub_vv_f32m1(buf_chunk_v, max_chunk_v, vl);
      // Exponentiate
      buf_chunk_v = __exp_f32m1(buf_chunk_v, vl);
      // Store the numerator to memory
      __riscv_vse32_v_f32m1(__o, buf_chunk_v, vl);
      // Accumulate
      den_chunk_v = __riscv_vfadd_vv_f32m1(den_chunk_v, buf_chunk_v, vl);
      // Bump channel pointers
      __i += innerSize;
      __o += innerSize;
    }
    // Restore the pointers
    __i = _i;
    __o = _o;

    /*
      Divide by the computed sum
    */

    for (size_t ch = 0; ch < channels; ++ch) {
      // Load numerator from memory
      num_chunk_v = __riscv_vle32_v_f32m1(__o, vl);
      // Divide
      res_chunk_v = __riscv_vfdiv_vv_f32m1(num_chunk_v, den_chunk_v, vl);
      // Store the result to memory
      __riscv_vse32_v_f32m1(__o, res_chunk_v, vl);
      // Bump channel pointers
      __o += innerSize;
    }
    // Bump stripmining pointers
    _i += vl;
    _o += vl;
    // Reset channel pointers
    __i = _i;
    __o = _o;
  }
}