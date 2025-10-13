import json
import os
import Models.models as models
import torch

f = open(os.path.join("..", "configs.json"))
conf = json.load(f)

def main():
    torch.set_default_dtype(torch.float32)

    rand_inp = torch.randn(1,1,28,28).float()

    model_32bit_adv = models.get_Wk17a_prec_32_adv().float()
    model_64bit_adv = models.get_Wk17a_prec_64_adv().float()

    model_p1 = models.get_Wk17a_order_pattern_1_adv().float()
    model_p2 = models.get_Wk17a_order_pattern_2_adv().float()
    model_p3 = models.get_Wk17a_order_pattern_3_adv().float()

    model_order = models.get_Wk17a_order_bias_adv().float()

    opset_version = 9

    torch.onnx.export(
        model_order,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_order_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

    torch.onnx.export(
        model_p2,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_order_pattern_2_f64_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

    torch.onnx.export(
        model_32bit_adv,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_32bit_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

    torch.onnx.export(
        model_64bit_adv,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_64bit_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

    torch.onnx.export(
        model_p1,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_order_pattern_1_f64_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

    torch.onnx.export(
        model_p3,
        rand_inp,
        os.path.join(*conf["Model_path"], "wk17a_order_pattern_3_f64_adversary.onnx"),
        opset_version=opset_version,
        input_names=["X"], output_names=["Y"],
        do_constant_folding=False,
        export_params=True,
    )

if __name__ == "__main__":
    main()

f.close()