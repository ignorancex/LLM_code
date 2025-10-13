import torch
from torch import nn
from torch.nn import functional as F
from .activations import get_activation, Sine
from ..configs import (
    MLPConfig,
    ActivationType,
    LoraType,
    SkipConnectionType,
    QuantizationConfig,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2 ** (k - 1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out * torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        grad_input = grad_output.clone()
        return grad_input, None


def ffnerv_weight_quantize(x: torch.Tensor, wbit: int) -> torch.Tensor:
    if wbit == 32:
        weight_q = x
    else:
        weight = torch.tanh(x)
        weight_q = qfn.apply(weight, wbit)
    return weight_q  # pyright: ignore[reportReturnType]


class BatchLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_decoders: int,
        activation: ActivationType,
        skip_connection: SkipConnectionType,
        sep_patch_pix: bool,
        is_last: bool,
        qcfg: QuantizationConfig,
        bias=True,
        dtype=None,
    ) -> None:
        super().__init__()
        self.qcfg = qcfg
        self.is_last = is_last
        self.sep_patch_pix = sep_patch_pix
        self.activation_type = activation
        self.num_decoders = num_decoders
        self.in_features = in_features
        self.out_features = out_features
        self.ps = int(
            (self.out_features // 3) ** 0.5
        )  # out_features = 27 for 3x3 patch
        if sep_patch_pix and is_last:
            in_features = in_features // (self.ps * self.ps)
        self.skip_connection: SkipConnectionType = skip_connection
        self.weight = nn.Parameter(
            torch.empty(
                (num_decoders, in_features, out_features), device=device, dtype=dtype
            )
        )
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros((num_decoders, 1, out_features), device=device, dtype=dtype)
            )
        self.activation, init, _ = get_activation(activation)
        init(self)
        self.using_lora = False

    def get_weight(self) -> torch.Tensor:
        if self.qcfg.ffnerv_qat and self.training and not self.is_last:
            weight = ffnerv_weight_quantize(self.weight, self.qcfg.quant_bit)
        elif self.qcfg.qat and self.training and not self.is_last:
            axis = self.qcfg.quant_axis[-1]
            axis = axis if axis >= 0 else self.weight.ndim + axis
            dims = tuple(i for i in range(self.weight.ndim) if i != axis)
            bits = self.qcfg.quant_bit
            quant_max = 2**bits - 1
            ch_min = self.weight.amin(dim=dims).detach()
            ch_max = self.weight.amax(dim=dims).detach()

            scales = (ch_max - ch_min) / quant_max
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)

            zero_points = (-ch_min / scales).round().clamp(0, quant_max)

            # Apply fake quantization
            weight = torch.fake_quantize_per_channel_affine(
                self.weight, scales, zero_points, axis, 0, quant_max
            )
        else:
            weight = self.weight
        return weight

    def forward(
        self, input: torch.Tensor, decoder_idx, abcd: torch.Tensor | None
    ) -> torch.Tensor:
        """
        X: n_weights x batch_size x in
        Returns: n_weights x batch_size x out
        """
        if self.using_lora and self.lora_type == "sinlora":
            return self.sinlora_forward(input, decoder_idx)
        elif self.using_lora and self.lora_type == "lora":
            return self.lora_forward(input, decoder_idx)
        elif self.using_lora and self.lora_type == "group":
            return self.group_lora_forward(input, decoder_idx)
        elif self.using_lora and self.lora_type == "no_lora":
            return self.no_lora_forward(input, decoder_idx)
        elif not self.using_lora and self.sep_patch_pix and self.is_last:
            return self.sep_patch_pix_forward(input, decoder_idx)
        idx = slice(None) if decoder_idx is None else decoder_idx
        output = input @ self.get_weight()[idx]
        if self.use_bias:
            output = output + self.bias[idx]
        if isinstance(self.activation, Sine):
            output = self.activation(output, abcd, decoder_idx)
        else:
            output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}, num_decoders={self.num_decoders}, bias={self.use_bias}"
        if self.using_lora and self.lora_type == "sinlora":
            s = s + f"\nlora_rank={self.lora_rank}, omega={self.omega}, g={self.g}"
        elif self.using_lora and self.lora_type == "lora":
            s = s + "\nlora_rank={self.lora_rank}"
        return s

    def lora(self, lora_type: LoraType, lora_rank: int, omega: float | None = None):
        self.lora_type = lora_type
        if lora_type == "sinlora" and omega is not None:
            self.lora_rank = lora_rank
            self.omega = omega
            self.g = self.in_features**0.5
            self.weight = nn.Parameter(self.weight[self.num_decoders // 2].clone())
            self.U = nn.Parameter(
                torch.zeros((self.num_decoders, self.in_features, lora_rank)).cuda()
            )
            nn.init.kaiming_uniform_(self.U, a=5**0.5)
            self.V = nn.Parameter(
                torch.zeros((self.num_decoders, lora_rank, self.out_features)).cuda()
            )
        elif lora_type == "lora":
            self.lora_rank = lora_rank
            self.weight = nn.Parameter(self.weight[self.num_decoders // 2].clone())
            self.U = nn.Parameter(
                torch.zeros((self.num_decoders, self.in_features, lora_rank)).cuda()
            )
            nn.init.kaiming_uniform_(self.U, a=5**0.5)
            self.V = nn.Parameter(
                torch.zeros((self.num_decoders, lora_rank, self.out_features)).cuda()
            )
        elif lora_type == "group":
            self.lora_rank = lora_rank
            self.weight = nn.Parameter(self.weight[self.num_decoders // 2].clone())
            UV = torch.empty((self.num_decoders, 2, lora_rank, self.out_features))
            UV[:, 0] = 1
            UV[:, 1] = 0
            self.UV = nn.Parameter(UV.cuda())
        elif lora_type == "no_lora":
            self.weight = nn.Parameter(self.weight[self.num_decoders // 2].T.clone())
            self.bias = nn.Parameter(self.bias[self.num_decoders // 2].clone())
        self.using_lora = True

    def sep_patch_pix_forward(self, input: torch.Tensor, decoder_idx):
        import einops

        idx = slice(None) if decoder_idx is None else decoder_idx
        if input.ndim == 2:
            input = input.unsqueeze(0)
        input = einops.rearrange(input, "g n (d p) -> g n p 1 d", p=self.ps**2)
        weight = self.get_weight()[idx]
        weight = einops.rearrange(weight, "g d (c p) -> g () p d c", c=3)
        output = input @ weight
        output = einops.rearrange(output, "g n p () c -> g n (c p)")
        if self.use_bias:
            output = output + self.bias[idx]
        if isinstance(self.activation, Sine):
            output = self.activation(output, None, decoder_idx)
        else:
            output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output

    def sinlora_forward(self, input: torch.Tensor, decoder_idx):
        idx = slice(None) if decoder_idx is None else decoder_idx
        adapter = torch.sin(self.omega * self.U @ self.V) / self.g
        output = input @ (self.get_weight() + adapter[idx])
        if self.use_bias:
            output = output + self.bias[idx]
        output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output

    def lora_forward(self, input: torch.Tensor, decoder_idx):
        idx = slice(None) if decoder_idx is None else decoder_idx
        adapter = self.U @ self.V
        output = input @ (self.get_weight() + adapter[idx])
        if self.use_bias:
            output = output + self.bias[idx]
        output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output

    def group_lora_forward(self, input: torch.Tensor, decoder_idx):
        idx = slice(None) if decoder_idx is None else decoder_idx
        # UV is num_decoders x 2 x in_features x out_features
        interp_shape = self.in_features, self.out_features
        UV = F.interpolate(self.UV, interp_shape, mode="nearest-exact")
        U, V = UV[idx, 0], UV[idx, 1]
        output = input @ (self.get_weight() * U + V)
        if self.use_bias:
            output = output + self.bias[idx]
        output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output

    def no_lora_forward(self, input: torch.Tensor, decoder_idx):
        output = F.linear(input, self.get_weight(), self.bias)
        output = self.activation(output)
        if output.shape[-1] == input.shape[-1] and self.skip_connection == "+":
            output = output + input
        elif output.shape[-1] == input.shape[-1] and self.skip_connection == "*":
            output = output * input
        return output


class BatchMLP(nn.Module):
    def __init__(
        self,
        dim_out: int,
        num_decoders: int,
        cfg: MLPConfig,
        skip_connection: SkipConnectionType,
        sep_patch_pix: bool,
        qcfg: QuantizationConfig,
    ):
        super().__init__()
        self.num_decoders = num_decoders
        self.num_layers = cfg.num_layers
        self.dim_hidden = cfg.dim_hidden
        layers = []
        first_dim_in = cfg.pos_encode_cfg.dim_out
        if self.num_layers == -1 and dim_out != cfg.pos_encode_cfg.dim_in:
            raise ValueError("Different in and out dims for identity BatchMLP")
        if cfg.bottleneck_override is not None:
            dims = cfg.bottleneck_override
        elif cfg.bottleneck:
            # dims should go first_dim_in -> dim_out
            dims = torch.linspace(
                first_dim_in, dim_out, self.num_layers + 2, dtype=torch.int
            ).tolist()
        else:
            dims = torch.full((self.num_layers + 2,), self.dim_hidden).tolist()
            if self.num_layers != -1:
                dims[0] = first_dim_in
                dims[-1] = dim_out

        for ind, (in_feats, out_feats) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == len(dims) - 2
            act = cfg.final_activation if ind == self.num_layers else cfg.activation
            layers.append(
                BatchLinear(
                    in_features=in_feats,
                    out_features=out_feats,
                    num_decoders=num_decoders,
                    bias=cfg.use_bias,
                    activation=act,
                    skip_connection=skip_connection,
                    sep_patch_pix=sep_patch_pix,
                    is_last=is_last,
                    qcfg=qcfg,
                )
            )
        self.net = nn.ModuleList(layers).to(device)

    def forward(self, input: torch.Tensor, decoder_idx, abcd: torch.Tensor | None):
        for layer in self.net:
            input = layer(input, decoder_idx=decoder_idx, abcd=abcd)
        return input

    def forward_batched(self, input: torch.Tensor, batch_size: int, chunk_size: int):
        # input is n x d
        input_chunked = torch.chunk(input, chunk_size)
        outs = []
        for input_chunk in input_chunked:
            chunk_outs = []
            for i in range(0, self.num_decoders, batch_size):
                # chunk_out is batch_size x n_chunked x dim_out
                chunk_out = self.forward(input_chunk, slice(i, i + batch_size), None)
                chunk_outs.append(chunk_out)
            # concatenates
            outs.append(torch.concatenate(chunk_outs, dim=0))
        return torch.concatenate(outs, dim=1)
