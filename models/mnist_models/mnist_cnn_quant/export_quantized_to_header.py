import numpy as np

def generate_header(model, filename="model_params_self.h"):
    """
    Generates a C header file exporting quantization parameters and weights 
    of `model` (NetQuantized) as uint8 arrays, padding zeros for biases.
    """
    layer_map = {
        "dw0": "conv0_dw",
        "pw0": "conv0_pw",
        "dw1": "conv1_dw",
        "pw1": "conv1_pw",
        "fc0": "fc0",
        "fc1": "fc1",
    }

    with open(filename, "w") as f:
        f.write("// Auto-generated symmetric-quant model_params.h\n")
        f.write("#ifndef MODELPARAMS_H\n#define MODELPARAMS_H\n\n")
        f.write("#include <stdint.h>\n#include \"lib_layers.h\"\n\n")
        f.write("#define BATCHES 1\n\n")

        # Input / output quantization params
        f.write(f"const quantization_params_t qp_input  = {{ {(1/model.input_scale):.8e}f, 0 }};\n")
        f.write(f"const quantization_params_t qp_logits = {{ {1/model.fc1.output_scale:.8e}f, 0 }};\n\n")

        # Requantization scales
        for layer_name, cname in layer_map.items():
            layer = getattr(model, layer_name)
            if hasattr(layer, 'out_channels'):
                out_ch = layer.out_channels
            elif hasattr(layer, 'out_features'):
                out_ch = layer.out_features
            else:
                raise ValueError(f"Layer {layer_name} has no out_channels or out_features")

            layer = getattr(model, layer_name)
            scales = np.array(layer.output_scale, dtype=np.float32).flatten()
            scales = np.array([scales[0] for _ in range(out_ch)])
            f.write(f"static float rq_{cname}_scale[{scales.size}] = {{\n    ")
            f.write(", ".join(f"{s:.8e}f" for s in scales))
            f.write("\n};\n")
            f.write(f"const requantization_params_t rq_{cname} = {{ rq_{cname}_scale, 0 }};\n\n")

        # Weight arrays with zero padding for biases
        for layer_name in layer_map:
            layer = getattr(model, layer_name)
            if layer_name in ["fc0", "fc1"]:
                w_int8 = layer.weight.detach().cpu().numpy().astype(np.int8).T.flatten()
                w_uint8 = ((w_int8.astype(np.int16) + 256) % 256).astype(np.uint8).tolist()
            else:
                w_int8 = layer.weight.detach().cpu().numpy().astype(np.int8).flatten()
                w_uint8 = ((w_int8.astype(np.int16) + 256) % 256).astype(np.uint8).tolist()

            # determine number of output channels/features
            if hasattr(layer, 'out_channels'):
                out_ch = layer.out_channels
            elif hasattr(layer, 'out_features'):
                out_ch = layer.out_features
            else:
                raise ValueError(f"Layer {layer_name} has no out_channels or out_features")

            zeros = [0] * (4 * out_ch)
            arr = zeros + w_uint8

            f.write(f"static const uint8_t {layer_name}_wb_q[] = {{\n    ")
            for i, val in enumerate(arr):
                if i and i % 16 == 0:
                    f.write("\n    ")
                f.write(f"{val}, ")
            f.write("\n};\n\n")

        f.write("#endif // MODELPARAMS_H\n")