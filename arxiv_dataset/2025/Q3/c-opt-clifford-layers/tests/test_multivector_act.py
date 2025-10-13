import torch
import unittest
from cliffordlayers.nn.modules.gcan import MultiVectorAct
from cliffordlayers.cliffordalgebra import CliffordAlgebra


def python_reference_forward(input, act_module):
    """Manually replicates the Python-only behavior of the original forward method."""
    v = act_module.algebra.embed(input, act_module.input_blades)
    if act_module.agg == "linear":
        v = v * torch.sigmoid(act_module.conv(v[..., act_module.kernel_blades]))
    elif act_module.agg == "sum":
        v = v * torch.sigmoid(v[..., act_module.kernel_blades].sum(dim=-1, keepdim=True))
    elif act_module.agg == "mean":
        v = v * torch.sigmoid(v[..., act_module.kernel_blades].mean(dim=-1, keepdim=True))
    else:
        raise ValueError(f"Aggregation {act_module.agg} not implemented.")
    return act_module.algebra.get(v, act_module.input_blades)


class TestMultiVectorAct(unittest.TestCase):
    def setUp(self):
        self.algebra = CliffordAlgebra([1, 1])
        self.batch_size = 4
        self.channels = 3

        # Fixed test input
        self.input = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]],
                [[1.3, 1.4, 1.5, 1.6],
                 [1.7, 1.8, 1.9, 2.0],
                 [2.1, 2.2, 2.3, 2.4]],
                [[2.5, 2.6, 2.7, 2.8],
                 [2.9, 3.0, 3.1, 3.2],
                 [3.3, 3.4, 3.5, 3.6]],
                [[3.7, 3.8, 3.9, 4.0],
                 [4.1, 4.2, 4.3, 4.4],
                 [4.5, 4.6, 4.7, 4.8]]
            ], dtype=torch.float32
        )

        self.kernel_blades = (0, 1, 2, 3)

    def _compare_outputs(self, act_module):
        out_c = act_module(self.input)
        out_py = python_reference_forward(self.input, act_module)

        self.assertTrue(torch.allclose(out_c, out_py, atol=1e-5), "Mismatch between C and Python outputs")
        self.assertEqual(out_c.shape, out_py.shape)

    def test_forward_linear(self):
        act = MultiVectorAct(
            channels=self.channels,
            algebra=self.algebra,
            input_blades=self.kernel_blades,
            kernel_blades=self.kernel_blades,
            agg="linear"
        )
        self._compare_outputs(act)

    def test_forward_sum(self):
        act = MultiVectorAct(
            channels=self.channels,
            algebra=self.algebra,
            input_blades=self.kernel_blades,
            kernel_blades=self.kernel_blades,
            agg="sum"
        )
        self._compare_outputs(act)

    def test_forward_mean(self):
        act = MultiVectorAct(
            channels=self.channels,
            algebra=self.algebra,
            input_blades=self.kernel_blades,
            kernel_blades=self.kernel_blades,
            agg="mean"
        )
        self._compare_outputs(act)


if __name__ == "__main__":
    unittest.main()
