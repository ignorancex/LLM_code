import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from typing import no_type_check


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonOpt(Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        params_list: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params_list}:
            b = torch.empty(1, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params_list if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[0]],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @no_type_check
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            params_world: list[Tensor] = []

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"]
                        * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params)):
                p = params[base_i]
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if g.ndim == 4:
                    g = g.view(len(g), -1)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                if base_i > 0:
                    update_prev()
                update_buffer.copy_(g)
                params_world = params[base_i : base_i + 1]
            update_prev()


class Muon(Optimizer):
    def __init__(self, model, muon_lr, adamw_lr):
        muon_params = [p for p in model.encoderINR.parameters() if p.ndim >= 2] + [
            p
            for layer in model.decoderINRs.net[:-1]
            for p in layer.parameters()
            if p.ndim >= 2
        ]
        adamw_params = (
            [p for p in model.encoderINR.parameters() if p.ndim < 2]
            + [
                p
                for layer in model.decoderINRs.net[:-1]
                for p in layer.parameters()
                if p.ndim < 2
            ]
            + [p for p in model.decoderINRs.net[-1].parameters()]
        )
        self.muon = MuonOpt(muon_params, muon_lr)
        self.adamw = AdamW(adamw_params, adamw_lr)
        super().__init__(self.muon.param_groups + self.adamw.param_groups, {})

    @no_type_check
    @torch.no_grad()
    def step(self, closure=None) -> None:
        self.muon.step()
        self.adamw.step()
