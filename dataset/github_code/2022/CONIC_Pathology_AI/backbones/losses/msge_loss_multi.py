import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss

####
class MSGEMultiLoss(_Loss):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """
    def __init__(self):
        super(MSGEMultiLoss, self).__init__()

    def forward(self, true: torch.Tensor, pred: torch.Tensor, np_map_gt: torch.Tensor, device: int) -> torch.Tensor:
        # true, pred, np_map_gt
        def get_sobel_kernel(size):
            """Get sobel kernel with a given size."""
            assert size % 2 == 1, "Must be odd, get size=%d" % size

            h_range = torch.arange(
                -size // 2 + 1,
                size // 2 + 1,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            v_range = torch.arange(
                -size // 2 + 1,
                size // 2 + 1,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            h, v = torch.meshgrid(h_range, v_range)
            kernel_h = h / (h * h + v * v + 1.0e-15)
            kernel_v = v / (h * h + v * v + 1.0e-15)
            return kernel_h, kernel_v

        ####
        def get_gradient_hv(hv):
            """For calculating gradient."""
            kernel_h, kernel_v = get_sobel_kernel(5)
            kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
            kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

            h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
            v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW
            ch_45 = hv[..., 2].unsqueeze(1)
            ch_135 = hv[..., 3].unsqueeze(1)

            # can only apply in NCHW mode

            kernel_45 = (kernel_h + kernel_v) / 2
            kernel_135 = (kernel_h - kernel_v) / 2
            
            h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
            v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
            ch_dh_45 = F.conv2d(ch_45, kernel_45, padding=2)
            ch_dh_135 = F.conv2d(ch_135, kernel_135, padding=2)

            dhv = torch.cat([h_dh_ch, v_dv_ch, ch_dh_45, ch_dh_135], dim=1)
            dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
            return dhv

        true = true.permute(0, 2, 3, 1).contiguous()
        pred = pred.permute(0, 2, 3, 1).contiguous()

        if len(np_map_gt.shape) == 4:
            np_map_gt = np_map_gt.permute(0, 2, 3, 1).contiguous()
            focus = np_map_gt[..., 1] 
        elif len(np_map_gt.shape) == 3:
            focus = np_map_gt

        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus, focus, focus], axis=-1)

        true_grad = get_gradient_hv(true)
        pred_grad = get_gradient_hv(pred)

        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss
