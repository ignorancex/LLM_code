import torch
import torch.nn.functional as F


def im2col_conv2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    padding: tuple[int, int] = (0, 0),
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    N, C, H, W = input.shape
    K, C, R, S = kernel.shape
    P = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    Q = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1

    input_im2col = F.unfold(
        input, kernel_size=(R, S), padding=padding, stride=stride, dilation=dilation
    )  # (N CRS PQ)
    input_im2col = input_im2col.permute(0, 2, 1)  # (N PQ CRS)

    kernel = kernel.view(K, -1)  # (K CRS)

    output = F.linear(input_im2col, kernel)  # (N PQ K)
    output = output.permute(0, 2, 1).view(N, K, P, Q)  # (N K PQ)

    return output


def im2col(
    input: torch.Tensor,
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
    stride: tuple[int, int],
) -> torch.Tensor:
    dilation = (1, 1)
    N, C, H, W = input.shape
    N_stride, C_stride, H_stride, W_stride = input.stride()
    P = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    Q = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return input.as_strided(
        size=(N, C, P, Q, *kernel_size),
        stride=(
            N_stride,
            C_stride,
            H_stride * stride[0],
            W_stride * stride[1],
            H_stride * dilation[0],
            W_stride * dilation[1],
        ),
        storage_offset=(-padding[0] * H_stride - padding[1] * W_stride),
    )


if __name__ == "__main__":
    N, C, H, W = 1, 3, 32, 32
    K, R, S = 3, 3, 3

    input_ = torch.randn(N, C, H, W)
    # kernel = torch.randn(K, C, R, S)

    # output = im2col_conv2d(
    #     input_, kernel, padding=(0, 0), stride=(1, 1), dilation=(1, 1)
    # )
    # print(output.shape)  # torch.Size([1, 3, 32, 32]

    torch.set_printoptions(profile="full")

    print(im2col(input_, (R, S), padding=(1, 1), stride=(1, 1)))
