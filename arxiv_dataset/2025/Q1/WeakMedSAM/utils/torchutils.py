import math
import torch


def get_cosine_schedule_with_warmup(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return (current_step / max(1, num_warmup_steps)) * 0.9 + 0.1
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0,
            0.5 * (1.0 + math.cos(math.pi * progress)) * 0.9 + 0.1,
        )

    return lr_lambda


def warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_cosine_schedule_with_warmup(
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        ),
    )


def max_norm(p):
    e = 1e-6
    if p.dim() == 3:
        C, H, W = p.size()
        max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        min_v = torch.min(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        p = (p - min_v + e) / (max_v - min_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        p = (p - min_v + e) / (max_v - min_v + e)
    return p
