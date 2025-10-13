import torch
import torch.nn as nn


class CrossStitchUnit(nn.Module):
    def __init__(self, num_tasks):
        super(CrossStitchUnit, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = nn.Parameter(torch.eye(num_tasks))  # Initialize cross-stitch weights as identity matrix

    def forward(self, inputs):
        """
        inputs: List of tensors, one for each task, where each tensor has shape [batch_size, features].
        """
        # Stack inputs into a single tensor of shape [num_tasks, batch_size, features]
        inputs = torch.stack(inputs, dim=0)  # Shape: [num_tasks, batch_size, features]

        # Reshape for matrix multiplication
        batch_size, features = inputs.shape[1], inputs.shape[2]
        inputs = inputs.permute(1, 2, 0)  # Shape: [batch_size, features, num_tasks]

        # Apply cross-stitch weights
        outputs = torch.matmul(inputs, self.alpha)  # Shape: [batch_size, features, num_tasks]

        # Reshape back to original format: list of tensors (one for each task)
        outputs = outputs.permute(2, 0, 1)  # Shape: [num_tasks, batch_size, features]
        return [outputs[i] for i in range(self.num_tasks)]

