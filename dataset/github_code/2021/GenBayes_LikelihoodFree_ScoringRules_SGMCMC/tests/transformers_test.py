import unittest

import numpy as np

from src.transformers import DummyTransformer, BoundedVarTransformer
import torch

class DummyTransformerTests(unittest.TestCase):
    def test(self):
        transformer = DummyTransformer()
        x = [np.array([3.2]), np.array([2.4])]
        self.assertEqual(x, transformer.transform(x))
        self.assertEqual(x, transformer.inverse_transform(x))
        self.assertEqual(0, transformer.jac_log_det_inverse_transform(x))
        self.assertEqual(0, transformer.jac_log_det(x))


class BoundedVarTransformerTests(unittest.TestCase):
    def setUp(self):
        self.transformer_lower_bounded = BoundedVarTransformer(lower_bound=np.array([0, 0]),
                                                               upper_bound=np.array([None, None]))
        self.transformer_two_sided = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10, 10]))
        self.transformer_mixed = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10, None]))
        self.transformer_dummy = BoundedVarTransformer(lower_bound=np.array([None, None]),
                                                       upper_bound=np.array([None, None]))
        self.list_transformers = [self.transformer_dummy, self.transformer_mixed,
                                  self.transformer_two_sided, self.transformer_lower_bounded]

    def test(self):
        x = [np.array([3.2]), np.array([2.4])]
        for transformer in self.list_transformers:
            self.assertEqual(len(x), len(transformer.inverse_transform(transformer.transform(x))))
            self.assertTrue(np.allclose(np.array(x), np.array(transformer.inverse_transform(transformer.transform(x)))))
            self.assertAlmostEqual(transformer.jac_log_det(x),
                                   transformer.jac_log_det_inverse_transform(transformer.transform(x)), delta=1e-7)

        # test dummy transformer actually does nothing:
        self.assertEqual(x, self.transformer_dummy.transform(x))
        self.assertEqual(x, self.transformer_dummy.inverse_transform(x))
        self.assertEqual(0, self.transformer_dummy.jac_log_det_inverse_transform(x))
        self.assertEqual(0, self.transformer_dummy.jac_log_det(x))

    def test_torch(self):
        x = [np.array([8.2]), np.array([5.5])]
        x_torch = torch.tensor(x).reshape(2)
        self.assertEqual(self.transformer_two_sided.inverse_transform(x), [np.array(x) for x in self.transformer_two_sided.inverse_transform(x_torch, use_torch=True).numpy()] )

        L = np.array([0])
        U = np.array([1])
        self.transformer_simple = BoundedVarTransformer(lower_bound=L, upper_bound=U)
        x_torch = torch.tensor([0.21])
        # -1.3972991 is calculated analytically using the d/d\theta expit(\theta), see https://rpubs.com/kaz_yos/stan_jacobian
        self.assertAlmostEqual(self.transformer_simple.jac_log_det_inverse_transform(x_torch, use_torch=True).detach().numpy(), np.array(-1.3972991), delta=1e-7)

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            transformer = BoundedVarTransformer(lower_bound=[0, 0], upper_bound=[10, 10])
        with self.assertRaises(RuntimeError):
            transformer = BoundedVarTransformer(lower_bound=np.array([0, 0]), upper_bound=np.array([10]))
        with self.assertRaises(NotImplementedError):
            transformer = BoundedVarTransformer(lower_bound=np.array([None, 0]), upper_bound=np.array([10, 10]))
        with self.assertRaises(RuntimeError):
            self.transformer_lower_bounded.transform(x=[np.array([3.2]), np.array([-2.4])])
        with self.assertRaises(RuntimeError):
            self.transformer_two_sided.transform(x=[np.array([13.2]), np.array([2.4])])


if __name__ == '__main__':
    unittest.main()
