import ctypes
import torch
from math import prod

# def as_tensor(pointer, shape, torch_type):
#     arr = (pointer._type_ * prod(shape)).from_address(
#         ctypes.addressof(pointer.contents))
    
#     return torch.frombuffer(arr, dtype=torch_type).view(*shape)