import random

import numpy as np
import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import seed_everything


def test_reproducibility():
    """Test that seed_everything ensures reproducible results."""
    seed = 42

    seed_everything(seed)
    result1 = perform_operations()

    seed_everything(seed)
    result2 = perform_operations()

    assert torch.equal(
        result1["torch_tensor"], result2["torch_tensor"]
    ), "Torch tensor results differ"
    assert np.array_equal(
        result1["numpy_array"], result2["numpy_array"]
    ), "NumPy array results differ"
    assert result1["random_number"] == result2["random_number"], "Random numbers differ"
    assert torch.equal(
        result1["torch_output"], result2["torch_output"]
    ), "Torch tensor results differ"


def perform_operations():
    """Simulate some operations involving randomness."""

    class dummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                3,
                16,
                3,
            )
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.dropout1 = nn.Dropout(0.25)
            self.linear1 = nn.Linear(25088, 128)
            self.rnn = nn.GRU(128, 64, 2, batch_first=True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.linear1(x)
            x, _ = self.rnn(x)
            return x

    random_number = random.randint(0, 100)

    numpy_array = np.random.rand(3, 3)

    torch_tensor = torch.rand(3, 3)

    torch_model = dummyModel()
    dummy_img = torch.rand(1, 3, 32, 32)

    torch_output = torch_model(dummy_img)

    return {
        "random_number": random_number,
        "numpy_array": numpy_array,
        "torch_tensor": torch_tensor,
        "torch_output": torch_output,
    }
