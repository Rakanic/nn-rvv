#include "conv2d.h"
#include "lib_kernels.h"

#include <riscv_vector.h> 
#include <stdint.h>


void dwconv_3x3_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const int8_t *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 2) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = weights + channels + ch * 9;
        
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        // Compute the convolution for this channel.
        vec_conv_3x3_int8_relu(rows, cols, a_stride, b_stride, k_ch, a_ch, b_ch, (int16_t) weights[ch], (float) requant_params.zero_point, requant_params.scale[ch]);
    }

}
//     size_t vl;

//     // 1) load int16 vector
//     vl = __riscv_vsetvl_e16m2(5);
//     vint16m2_t v_in  = __riscv_vmv_v_x_i16m2(5, vl);
//     vint16m2_t v1  = __riscv_vwmacc_vx_i16m2(v_in, 2, __riscv_vmv_v_x_i8m1(4, vl), vl);

//     vint32m4_t v_i32 = __riscv_vwcvt_x_x_v_i32m4(v1, vl);

//     // 3) convert int32→float32
//     vfloat32m4_t v_fp = __riscv_vfcvt_f_x_v_f32m4(v_i32, vl);

//     float tmp[vl];
//     __riscv_vse32_v_f32m4(tmp, v_fp, vl);

//     // Print each lane
//     for (size_t i = 0; i < vl; i++) {
//         printf("elem %d: %d\n", i, (int) tmp[i]);
//     }
// }

