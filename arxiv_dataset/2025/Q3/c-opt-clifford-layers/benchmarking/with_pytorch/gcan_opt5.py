import torch, ctypes, warnings
from torch import nn
from cliffordlayers.utils.load_from_clib import load_from_clib

class MultiVectorActOpt5(nn.Module):
    """
    Two–way implementation of the multi-vector activation:
      • backend="c"   →  calls `multivector_act_forward_opt5` (packed blades)
      • backend="py"  →  pure-Python reference (identical math)
    If the shared library cannot be loaded, we silently fall back to Python.
    """

    def __init__(
        self,
        channels,
        algebra,
        input_blades,
        kernel_blades=None,
        agg: str = "linear",
        backend: str = "c",
    ):
        super().__init__()
        self.algebra = algebra
        self.input_blades = tuple(input_blades)
        self.kernel_blades = tuple(kernel_blades) if kernel_blades is not None else self.input_blades
        self.channels = channels
        self.agg = agg
        self.backend = backend.lower()

        if self.agg == "linear":
            self.conv = nn.Conv1d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=len(self.kernel_blades),
                groups=self.channels,
            )

        self._lib = None
        if self.backend == "c":
            try:
                self._lib = load_from_clib("multivector_activation/multivector_act_opt5.so")
                self._fn = self._lib.multivector_act_forward_opt5
                self._fn.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # v_full
                    ctypes.POINTER(ctypes.c_float),  # v_pack
                    ctypes.POINTER(ctypes.c_float),  # w
                    ctypes.POINTER(ctypes.c_float),  # bias
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int,                     # agg_mode
                    ctypes.POINTER(ctypes.c_float),   # out
                ]
                self._fn.restype = None
            except OSError as err:
                warnings.warn(f"C backend unavailable ({err}). Falling back to pure Python.")
                self.backend = "py"

    def forward(self, input_3d_tensor):
        v_embedded = self.algebra.embed(input_3d_tensor, self.input_blades)
        B, C, NB = v_embedded.shape

        if self.backend == "py":
            if self.agg == "linear":
                out = v_embedded * torch.sigmoid(self.conv(v_embedded[..., self.kernel_blades]))
            elif self.agg == "sum":
                out = v_embedded * torch.sigmoid(v_embedded[..., self.kernel_blades].sum(dim=-1, keepdim=True))
            elif self.agg == "mean":
                out = v_embedded * torch.sigmoid(v_embedded[..., self.kernel_blades].mean(dim=-1, keepdim=True))
            else:
                raise ValueError(f"Aggregation {self.agg} not implemented.")
            return self.algebra.get(out, self.input_blades)

        agg_mode = {"linear": 0, "sum": 1, "mean": 2}[self.agg]
        output = torch.empty_like(v_embedded)

        if self.agg == "linear":
            conv_weights = self.conv.weight.view(C, -1).contiguous()
            conv_bias = self.conv.bias.contiguous()
            conv_w_ptr = ctypes.cast(conv_weights.data_ptr(), ctypes.POINTER(ctypes.c_float))
            conv_b_ptr = ctypes.cast(conv_bias.data_ptr(), ctypes.POINTER(ctypes.c_float))
        else:
            conv_w_ptr = conv_b_ptr = ctypes.POINTER(ctypes.c_float)()

        v_pack = v_embedded[..., self.kernel_blades].contiguous()

        v_full_ptr = ctypes.cast(v_embedded.data_ptr(), ctypes.POINTER(ctypes.c_float))
        v_pack_ptr = ctypes.cast(v_pack.data_ptr(), ctypes.POINTER(ctypes.c_float))
        out_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))

        self._fn(
            v_full_ptr,
            v_pack_ptr,
            conv_w_ptr,
            conv_b_ptr,
            B,
            C,
            NB,
            len(self.kernel_blades),
            agg_mode,
            out_ptr,
        )

        return self.algebra.get(output, self.input_blades)
