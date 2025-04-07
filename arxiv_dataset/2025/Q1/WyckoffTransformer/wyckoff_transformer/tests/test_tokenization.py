import unittest
import torch
from ..tokenization import argsort_multiple, SpaceGroupEncoder


class TestArgsortMultiple(unittest.TestCase):
    @torch.no_grad()
    def test_argsort_multiple_basic(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]])
        expected_output = torch.tensor([[2, 0, 1], [2, 1, 0]])
        output = argsort_multiple(tensor1, tensor2, dim=1)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_single_tensor(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        expected_output = torch.tensor([[1, 2, 0], [2, 1, 0]])
        output = argsort_multiple(tensor1, dim=1)
        self.assertTrue(torch.equal(output, expected_output))

    @torch.no_grad()
    def test_argsort_multiple_invalid_input(self):
        tensor1 = torch.tensor([[3, 1, 2], [6, 5, 4]])
        tensor2 = torch.tensor([[9, 8, 7], [6, 5, 4]])
        tensor3 = torch.tensor([[3, 2, 1], [4, 5, 6]])
        with self.assertRaises(NotImplementedError):
            argsort_multiple(tensor1, tensor2, tensor3, dim=1)

class TestSpaceGroupEncoder(unittest.TestCase):
    def test_space_group_encoder(self):
        all_group_encoder = SpaceGroupEncoder.from_sg_set(range(1, 231))
        all_groups = set()
        for group_number in range(1, 231):
            group = tuple(all_group_encoder[group_number])
            if group in all_groups:
                raise ValueError(f"Duplicate group: {group}")
            all_groups.add(group)        


if __name__ == '__main__':
    unittest.main()