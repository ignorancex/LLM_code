import torch


def mse_loss(x, n_obs) -> torch.Tensor:
    return torch.sum(x**2, dim=(-2, -1)) / n_obs


def l1_loss(x, n_obs) -> torch.Tensor:
    return torch.sum(torch.abs(x), dim=(-2, -1)) / n_obs


def l2_loss(x, n_obs) -> torch.Tensor:
    return torch.sqrt(torch.sum(x**2, dim=(-2, -1)) / n_obs)


def batched_loss(x, n_obs) -> torch.Tensor:
    assert len(x.shape) == 4, "Input tensor must have shape [batch_size, channels, resolution, resolution]"
    batch_size = x.shape[0]

    # Calculate L2 loss per channel
    channel_losses = torch.sqrt(torch.sum(x**2, dim=(0, 2, 3)) / (batch_size * n_obs))

    # Repeat the loss for each sample in the batch and reshape to [batch_size, channels]
    batch_losses = channel_losses.repeat(batch_size, 1)
    return batch_losses


def huber_loss(x, n_obs, delta=1.0) -> torch.Tensor:
    abs_x = torch.abs(x)
    return torch.sum(torch.where(abs_x < delta, 0.5 * abs_x**2, delta * (abs_x - 0.5 * delta)), dim=(-2, -1)) / n_obs


def get_loss_func(loss_type):
    if loss_type == "mse":
        return mse_loss
    elif loss_type == "l1":
        return l1_loss
    elif loss_type == "l2":
        return l2_loss
    elif loss_type == "batched":
        return batched_loss
    elif loss_type.startswith("huber"):
        delta = float(loss_type.split("-")[1]) if "-" in loss_type else 1.0
        return lambda x, n_obs: huber_loss(x, n_obs, delta=delta)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
