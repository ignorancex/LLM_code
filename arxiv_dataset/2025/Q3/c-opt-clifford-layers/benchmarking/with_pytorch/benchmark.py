from cliffordlayers.cliffordalgebra import CliffordAlgebra
from networks import CliffordBasicBlock2d, CliffordBasicBlock3d
from cliffordlayers.nn.modules.cliffordconv_opt2 import CliffordConv2d as CliffordConv2dOpt2
from cliffordlayers.nn.modules.cliffordconv_opt2 import CliffordConv3d as CliffordConv3dOpt2
from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d, CliffordConv3d
from gcan_opt5 import MultiVectorActOpt5
import torch
import time

def get_act_functional(act_module, n_blades):
    def python_reference_forward(input):
        nonlocal act_module
        old_shape = input.shape
        input = input.view(input.shape[0], -1, n_blades)
        v = act_module.algebra.embed(input, act_module.input_blades)
        if act_module.agg == "linear":
            v = v * torch.sigmoid(act_module.conv(v[..., act_module.kernel_blades]))
        elif act_module.agg == "sum":
            v = v * torch.sigmoid(v[..., act_module.kernel_blades].sum(dim=-1, keepdim=True))
        elif act_module.agg == "mean":
            v = v * torch.sigmoid(v[..., act_module.kernel_blades].mean(dim=-1, keepdim=True))
        else:
            raise ValueError(f"Aggregation {act_module.agg} not implemented.")
        return act_module.algebra.get(v, act_module.input_blades).view(old_shape)
    return python_reference_forward

def get_act_functional_c_backed(act_module_c_opt, n_blades):
    def c_backed_forward(input_5d_tensor):
        nonlocal act_module_c_opt
        old_shape = input_5d_tensor.shape
        input_3d = input_5d_tensor.view(input_5d_tensor.shape[0], -1, n_blades)
        output_3d = act_module_c_opt(input_3d)
        return output_3d.view(old_shape)
    return c_backed_forward

def measure_speed(model, x, num_runs=5):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        start_time = time.time()
        for _ in range(num_runs):
            model(x)
        end_time = time.time()
        return (end_time - start_time) * 1000 / num_runs  # ms

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    for dim in [2, 3]:
        for agg in ["linear", "sum", "mean"]:
            res_for_test = None
            for backend_act in ["py", "c"]:
                for backend_conv in ["py", "c"]:
                    print(f"Testing for {dim}D, agg={agg}, backend_act={backend_act}, backend_conv={backend_conv}")
                    channels = 17
                    batch_size = 8
                    im_size = 100
                    filter_size = 17
                    if dim == 3:
                        filter_size = 7
                        im_size = 30
                        channels = 7
                    torch.manual_seed(42)
                    acts = [
                        MultiVectorActOpt5(
                        channels=channels * im_size ** dim,
                        algebra=CliffordAlgebra([1, -1]) if dim == 2 else CliffordAlgebra([1, -1, 1]),
                        input_blades=(0, 1, 2, 3) if dim == 2 else (0, 1, 2, 3, 4, 5, 6, 7),
                        agg=agg,
                        backend=backend_act),
                        MultiVectorActOpt5(
                        channels=channels * (im_size - filter_size + 1) ** dim,
                        algebra=CliffordAlgebra([1, -1]) if dim == 2 else CliffordAlgebra([1, -1, 1]),
                        input_blades=(0, 1, 2, 3) if dim == 2 else (0, 1, 2, 3, 4, 5, 6, 7),
                        agg=agg,
                        backend=backend_act)
                    ]
                    if dim == 2:
                        model = CliffordBasicBlock2d(
                            g=[1, -1],
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=filter_size,
                            activation=[get_act_functional(acts[0], 4), get_act_functional(acts[1], 4)] if backend_act == "py" else [get_act_functional_c_backed(acts[0], 4), get_act_functional_c_backed(acts[1], 4)],
                            conv2d_class=CliffordConv2dOpt2 if backend_conv == "c" else CliffordConv2d
                        )
                    else:
                        model = CliffordBasicBlock3d(
                            g=[1, -1, 1],
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=filter_size,
                            activation=[get_act_functional(acts[0], 8), get_act_functional(acts[1], 8)] if backend_act == "py" else [get_act_functional_c_backed(acts[0], 8), get_act_functional_c_backed(acts[1], 8)],
                            conv3d_class=CliffordConv3dOpt2 if backend_conv == "c" else CliffordConv3d
                        )
                    if dim == 2:
                        x = torch.randn(batch_size, channels, im_size, im_size, 2**dim)
                    else:
                        x = torch.randn(batch_size, channels, im_size, im_size, im_size, 2**dim)
                    y = model(x)
                    time_in_ms = measure_speed(model, x)
                    print(f"Time for {dim}D, agg={agg}, backend_act={backend_act}, backend_conv={backend_conv}: {time_in_ms:.2f} ms")
                    if res_for_test is None:
                        res_for_test = y
                    else:
                        torch.testing.assert_close(res_for_test, y, rtol=1e-4, atol=1e-4)
