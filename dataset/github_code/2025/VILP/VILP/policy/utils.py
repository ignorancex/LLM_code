import torch
def interpolate_tensor(input_tensor, n):

    batch_size, steps, dim = input_tensor.shape
    new_steps = steps + (steps - 1) * n
    output_tensor = torch.zeros((batch_size, new_steps, dim), device=input_tensor.device, dtype=input_tensor.dtype)
    output_tensor[:, ::n+1, :] = input_tensor
    
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        output_tensor[:, 1+i::n+1, :] = alpha * input_tensor[:, 1:, :] + (1 - alpha) * input_tensor[:, :-1, :]
    
    return output_tensor