#include "nn_rvv/layers.h"
#include "ops/ara/exp.h"

#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

/* SiLU(x) = x / (1 + exp(-x)),  in place. */
void silu_f32(float *x, size_t n) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vx  = __riscv_vle32_v_f32m4(x, vl);
        vfloat32m4_t vne = __riscv_vfneg_v_f32m4(vx, vl);
        vfloat32m4_t ve  = __exp_f32m4(vne, vl);
        vfloat32m4_t vd  = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        vfloat32m4_t vy  = __riscv_vfdiv_vv_f32m4(vx, vd, vl);
        __riscv_vse32_v_f32m4(x, vy, vl);
        x += vl; n -= vl;
    }
}

/* GELU (tanh approximation):
 *   0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))   where c = sqrt(2/pi)
 *   tanh(y) = 1 - 2 / (1 + exp(2y))
 */
void gelu_f32(float *x, size_t n) {
    const float C = 0.7978845608028654f;        /* sqrt(2/pi)               */
    const float A = 0.044715f;
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vx   = __riscv_vle32_v_f32m4(x, vl);
        vfloat32m4_t vsq  = __riscv_vfmul_vv_f32m4(vx, vx, vl);          /* x^2 */
        vfloat32m4_t vcb  = __riscv_vfmul_vv_f32m4(vsq, vx, vl);         /* x^3 */
        /* inner = C * (x + A * x^3) */
        vfloat32m4_t vin  = __riscv_vfmul_vf_f32m4(vcb, A, vl);
        vin               = __riscv_vfadd_vv_f32m4(vin, vx, vl);
        vin               = __riscv_vfmul_vf_f32m4(vin, C, vl);
        /* tanh = 1 - 2 / (1 + exp(2*inner)) */
        vfloat32m4_t v2y  = __riscv_vfmul_vf_f32m4(vin, 2.0f, vl);
        vfloat32m4_t ve   = __exp_f32m4(v2y, vl);
        vfloat32m4_t vd   = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        vfloat32m4_t vinv = __riscv_vfrdiv_vf_f32m4(vd, 2.0f, vl);       /* 2 / vd */
        vfloat32m4_t vth  = __riscv_vfrsub_vf_f32m4(vinv, 1.0f, vl);     /* 1 - inv */
        /* y = 0.5 * x * (1 + tanh) */
        vfloat32m4_t vsum = __riscv_vfadd_vf_f32m4(vth, 1.0f, vl);
        vfloat32m4_t vy   = __riscv_vfmul_vv_f32m4(vx, vsum, vl);
        vy                = __riscv_vfmul_vf_f32m4(vy, 0.5f, vl);
        __riscv_vse32_v_f32m4(x, vy, vl);
        x += vl; n -= vl;
    }
}

/* SwiGLU multiply for one row.
 *   gate_up is INTERLEAVED: gate_up[2i] = gate, gate_up[2i+1] = up
 *   out[i] = SiLU(gate_up[2i]) * gate_up[2i+1]
 *
 * Strided loads at 2*sizeof(float) stride pull the two streams apart. */
void swiglu_multiply_f32(float *out, const float *gate_up, size_t inter) {
    const size_t stride = 2 * sizeof(float);
    while (inter > 0) {
        size_t vl = __riscv_vsetvl_e32m4(inter);
        vfloat32m4_t vg  = __riscv_vlse32_v_f32m4(gate_up,     stride, vl);
        vfloat32m4_t vu  = __riscv_vlse32_v_f32m4(gate_up + 1, stride, vl);
        /* SiLU(g) */
        vfloat32m4_t vne = __riscv_vfneg_v_f32m4(vg, vl);
        vfloat32m4_t ve  = __exp_f32m4(vne, vl);
        vfloat32m4_t vd  = __riscv_vfadd_vf_f32m4(ve, 1.0f, vl);
        vfloat32m4_t vsl = __riscv_vfdiv_vv_f32m4(vg, vd, vl);
        /* * u */
        vfloat32m4_t vy  = __riscv_vfmul_vv_f32m4(vsl, vu, vl);
        __riscv_vse32_v_f32m4(out, vy, vl);
        out += vl; gate_up += 2 * vl; inter -= vl;
    }
}
