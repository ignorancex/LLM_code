import torch, unittest, ctypes, os
from cliffordlayers.nn.modules.gcan import MultiVectorAct
from cliffordlayers.cliffordalgebra import CliffordAlgebra
import numpy as np

def repack_blades(x: torch.Tensor, kernel_blades):
    return x[..., kernel_blades].contiguous().cpu().numpy().astype(np.float32)

class TestMultiVectorActOptimizedVersions(unittest.TestCase):

    def setUp(self):
        self.algebra        = CliffordAlgebra([1, 1])
        self.batch_size     = 4
        self.channels       = 8
        self.input_blades   = (0, 1, 2, 3)
        self.kernel_blades  = (0, 1, 2, 3)
        self.input          = torch.rand(self.batch_size,
                                         self.channels,
                                         len(self.input_blades),
                                         dtype=torch.float32)

        self.debug = False

        self.lib_specs = [
            ("./clib/multivector_activation/multivector_act.so",      "multivector_act_forward_base", False),
            ("./clib/multivector_activation/multivector_act_opt1.so", "multivector_act_forward_opt1", False),
            ("./clib/multivector_activation/multivector_act_opt2.so", "multivector_act_forward_opt2", True),
            ("./clib/multivector_activation/multivector_act_opt3.so", "multivector_act_forward_opt3", True),
            ("./clib/multivector_activation/multivector_act_opt4.so", "multivector_act_forward_opt4", True),
            ("./clib/multivector_activation/multivector_act_opt5.so", "multivector_act_forward_opt5", True),
        ]
        self.agg_modes = ["linear", "sum", "mean"]

    def python_reference_forward(self, inp, act_module):
        v = act_module.algebra.embed(inp, act_module.input_blades)
        if act_module.agg == "linear":
            v = v * torch.sigmoid(act_module.conv(v[..., act_module.kernel_blades]))
        elif act_module.agg == "sum":
            v = v * torch.sigmoid(v[..., act_module.kernel_blades].sum(dim=-1, keepdim=True))
        elif act_module.agg == "mean":
            v = v * torch.sigmoid(v[..., act_module.kernel_blades].mean(dim=-1, keepdim=True))
        else:
            raise ValueError(f"Aggregation {act_module.agg} not implemented.")
        return act_module.algebra.get(v, act_module.input_blades)

    def _run_and_compare(self, act_module, lib_path, fn_name, packed):
        lib = ctypes.CDLL(lib_path)
        fn = getattr(lib, fn_name)

        B, C, NB = self.batch_size, self.channels, len(self.input_blades)
        K         = len(self.kernel_blades)
        agg_id    = {"linear": 0, "sum": 1, "mean": 2}[act_module.agg]

        v_full_np  = self.input.contiguous().cpu().numpy().astype(np.float32)
        v_full_ptr = v_full_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if packed:
            v_pack_np  = repack_blades(self.input, self.kernel_blades)
            v_pack_ptr = v_pack_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        out_np  = np.empty_like(v_full_np)
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if agg_id == 0:
            w_torch = act_module.conv.weight.detach().reshape(C, K)
            b_torch = act_module.conv.bias.detach()
            w_np = w_torch.contiguous().cpu().numpy().astype(np.float32)
            b_np = b_torch.contiguous().cpu().numpy().astype(np.float32)
            w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            w_ptr = b_ptr = ctypes.POINTER(ctypes.c_float)()

        if packed:
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # v_full
                ctypes.POINTER(ctypes.c_float),  # v_pack
                ctypes.POINTER(ctypes.c_float),  # w
                ctypes.POINTER(ctypes.c_float),  # bias
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            fn(v_full_ptr, v_pack_ptr, w_ptr, b_ptr,
               B, C, NB, K, agg_id, out_ptr)
        else:
            kidx_np  = np.array(self.kernel_blades, dtype=np.int32)
            kidx_ptr = kidx_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # v_full
                ctypes.POINTER(ctypes.c_float),  # w
                ctypes.POINTER(ctypes.c_float),  # bias
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),    # kernel_indices
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            fn(v_full_ptr, w_ptr, b_ptr,
               B, C, NB, K, kidx_ptr, agg_id, out_ptr)

        out_c  = torch.tensor(out_np)
        out_py = self.python_reference_forward(self.input, act_module)

        if self.debug:
            print(f"\n[DEBUG] lib: {lib_path}, agg: {act_module.agg}")
            print("C Output:\n", out_c)
            print("Py Output:\n", out_py)

        self.assertTrue(torch.allclose(out_c, out_py, atol=1e-5),
                        f"Mismatch for {os.path.basename(lib_path)} / {act_module.agg}")

    def test_all_versions(self):
        for lib_path, fn_name, packed in self.lib_specs:
            if not os.path.exists(lib_path):
                continue
            for agg in self.agg_modes:
                with self.subTest(lib=os.path.basename(lib_path), agg=agg):
                    act = MultiVectorAct(
                        channels=self.channels,
                        algebra=self.algebra,
                        input_blades=self.input_blades,
                        kernel_blades=self.kernel_blades,
                        agg=agg
                    )
                    self._run_and_compare(act, lib_path, fn_name, packed)

if __name__ == "__main__":
    unittest.main()
