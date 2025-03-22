import torch
import torch.types

from flash_dropout.functional.utils import blockwise_dropout_mask
from flash_dropout.triton.dsd_matmul import blockwise_dsd_matmul
from flash_dropout.triton.sdd_matmul import blockwise_sdd_matmul


class BlockwiseDropoutMatmulCtx(torch.autograd.function.FunctionCtx):
    saved_tensors: tuple[torch.Tensor, ...]
    block_size: int
    p: float


class BlockwiseDropoutMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: BlockwiseDropoutMatmulCtx,
        input: torch.Tensor,  # (MK sparse)
        weight: torch.Tensor,  # (NK dense)
        block_size: int,
        p: float,
    ):
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"

        mask = blockwise_dropout_mask(input, block_size, p=p)

        ctx.save_for_backward(input, weight, mask)
        ctx.block_size = block_size
        ctx.p = p

        # y (MN dense) = x (MK sparse) x w^T (KN dense)
        result = blockwise_dsd_matmul(
            input, mask, weight.T, block_size, scale=1 / (1 - p)
        )
        return result

    @staticmethod
    def backward(
        ctx: BlockwiseDropoutMatmulCtx,
        grad_output: torch.Tensor,
    ):
        input, weight, mask = ctx.saved_tensors
        block_size = ctx.block_size
        p = ctx.p

        # dx (MK sparse) = dy (MN dense) x w (NK dense)
        table = torch.nonzero(~mask, as_tuple=False)
        grad_input = blockwise_sdd_matmul(
            grad_output, weight, table, block_size, scale=1 / (1 - p)
        )

        # dw (NK dense) = (x^T (KM sparse) x dy (MN dense)).T
        grad_weight = blockwise_dsd_matmul(
            input.T, mask.T, grad_output, block_size, scale=1 / (1 - p)
        ).T

        return grad_input, grad_weight, None, None


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: int, p: float
) -> torch.Tensor:
    return BlockwiseDropoutMatmul.apply(input, weight, block_size, p)  # type: ignore
