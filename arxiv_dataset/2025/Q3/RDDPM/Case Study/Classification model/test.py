import unittest
import torch
from model import ModifiedVGG19
from augment import get_augmentation

class TestVGG19Model(unittest.TestCase):
    def test_model_output(self):
        model = ModifiedVGG19()
        x = torch.randn(1, 1, 300, 300)  # A random 300x300 grayscale image
        output = model(x)
        self.assertEqual(output.shape, (1, 1))  # Should output a single number
    
    def test_augmentation(self):
        transform = get_augmentation()
        self.assertIsNotNone(transform)
    
if __name__ == "__main__":
    unittest.main()