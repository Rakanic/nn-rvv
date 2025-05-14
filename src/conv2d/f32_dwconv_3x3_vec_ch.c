#include "conv2d.h"

#include <riscv_vector.h> 
#include <stdint.h>



void xnn_f32_dwconv_minmax_ukernel_9p8vc__rvv(
    size_t channels,
    size_t output_width,
    float* input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment
) 
{
  size_t input_columns = output_width + 2;
  float* i0 = input;  
  do {
    float* i1 = i0 + channels;
    float* i2 = i1 + channels;
    float* i3 = i0 + input_columns * channels;
    float* i4 = i3 + channels;
    float* i5 = i4 + channels;
    float* i6 = i3 + input_columns * channels;
    float* i7 = i6 + channels;
    float* i8 = i7 + channels;
    input = ( float*) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    const float* w_new;
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    size_t vl;
    size_t dist;
    do {
      vl = __riscv_vsetvl_e32m8(c);
      dist = MAX(vl, channels);
      w_new = w + vl;
      // load bias to vAcc
      vfloat32m8_t vAcc = __riscv_vundefined_f32m8();
      vAcc = __riscv_vle32_v_f32m8_tu(vAcc, w, vl);
      w += dist;

      vfloat32m8_t va = __riscv_vundefined_f32m8();
      vfloat32m8_t vb = __riscv_vundefined_f32m8();
      va = __riscv_vle32_v_f32m8_tu(va, i0, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);  
      w  += dist;
      i0 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      va = __riscv_vle32_v_f32m8_tu(va, i1, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i1 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      
      va = __riscv_vle32_v_f32m8_tu(va, i2, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i2 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      
      va = __riscv_vle32_v_f32m8_tu(va, i3, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl); 
      w  += dist;
      i3 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      va = __riscv_vle32_v_f32m8_tu(va, i4, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i4 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      va = __riscv_vle32_v_f32m8_tu(va, i5, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i5 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      va = __riscv_vle32_v_f32m8_tu(va, i6, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i6 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);
      
      va = __riscv_vle32_v_f32m8_tu(va, i7, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl); 
      w  += dist;
      i7 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      va = __riscv_vle32_v_f32m8_tu(va, i8, vl);
      vb = __riscv_vle32_v_f32m8_tu(vb, w, vl);
      w  += dist;
      i8 += vl;
      vAcc = __riscv_vfmacc_vv_f32m8_tu(vAcc, va, vb, vl);

      __riscv_vse32_v_f32m8(output, vAcc, vl);
      output += vl;
      c -= vl;
      w = w_new;
    } while(c != 0);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void dwconv_3x3_f32_VCH(
  size_t rows, size_t cols, 
  size_t channels,
  float* input,
  const float* weights,
  float* output,
  intptr_t input_stride,
  size_t output_increment
) {
  for (size_t output_row = 0; output_row < rows; output_row++) {
      xnn_f32_dwconv_minmax_ukernel_9p8vc__rvv(
          channels, 
          cols,
          input + output_row * (cols+2) * channels, 
          weights, 
          output + output_row * (cols) * channels, 
          input_stride, output_increment
      );
  }
}