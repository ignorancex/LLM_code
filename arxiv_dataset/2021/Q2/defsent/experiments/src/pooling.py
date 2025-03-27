import torch
import torch.nn as nn
from torch import Tensor


# using @torch.jit.script is slower than this simple implementaion.
class NonParametricPooling(nn.Module):
    def __init__(self, pooling_name: str) -> None:
        super().__init__()
        self.pooling_name = pooling_name

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling_name == "CLS":
            return x[:, 0]

        # masked tokens are marked as `0`
        sent_len = attention_mask.sum(dim=1, keepdim=True)
        if self.pooling_name == "SEP":
            batch_size = x.size(0)
            batch_indices = torch.LongTensor(range(batch_size))
            sep_indices = (sent_len.long() - 1).squeeze()
            return x[batch_indices, sep_indices]

        mask_value = 0 if self.pooling_name in ["Mean", "Sum"] else -1e6
        x[attention_mask.long() == 0, :] = mask_value

        if self.pooling_name == "Mean":
            return x.sum(dim=1) / sent_len

        elif self.pooling_name == "Max":
            return x.max(dim=1).values

        elif self.pooling_name == "Sum":
            return x.sum(dim=1)

        else:
            raise ValueError(f"No such a pooling name! {self.pooling_name}")
