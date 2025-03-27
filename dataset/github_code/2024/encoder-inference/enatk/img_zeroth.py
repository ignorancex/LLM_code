from typing import Union, Tuple

import numpy as np
import torch

from .utils import get_scheduler


class ImgAtkZeroth:
    def __init__(
        self,
        steps: int,
        scheduler: str,
        scheduler_pms: dict,
        img_size: Union[int, Tuple[int]],
        gd_est_eps: float,
        gd_est_num: int,
        device="cuda:0",
        logger=None,
    ):

        self.steps = steps
        self.scheduler = get_scheduler(scheduler, **scheduler_pms)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size

        self.gd_est_eps = gd_est_eps
        self.gd_est_num = gd_est_num

        self.device = device
        self.logger = logger

    """ Return an image that its representation is very similar to `target`.

    Args:
        encoder: Pretrained encoder.
        trans: Transform an image of type `np.ndarray` to `torch.Tensor`.
        target (torch.Tensor): Targeted representation.
    """
    @torch.inference_mode()
    def attack(
        self,
        encoder,
        target: torch.Tensor,
        gen_num: int = 1,
        query_bs: int = None,
        # classifier=None,
    ) -> np.ndarray:

        size = self.img_size
        device = self.device
        logger = self.logger

        if query_bs is None:
            query_bs = self.gd_est_num * gen_num

        scheduler = self.scheduler
        scheduler.reset()

        xs = torch.rand((gen_num, 3, *size), dtype=torch.float32, device=device)

        target = target.to(device) # shape [D,]

        for step in range(self.steps):
            """ two-point zeroth-order gradient estimation BEGIN """

            noise = torch.rand([self.gd_est_num, 3, *size], dtype=torch.float32, device=device) - 0.5
            norm = (noise.view(len(noise), -1) ** 2).sum(dim=-1).sqrt().view(len(noise), *((1,)*(len(noise.shape)-1)))
            noise = noise / norm

            noi_eps = noise * self.gd_est_eps
            inp_cache = np.empty([len(noise)*gen_num, *size, 3], dtype=np.uint8)
            loss = torch.zeros(len(noise)*gen_num, dtype=torch.float32, device=device)

            for i in range(gen_num):
                st = i * len(noise)
                ed = st + len(noise)
                inp_cache[st : ed] = self._internal_trans_inv(xs[i].unsqueeze(0) + noi_eps) # shape: [B, C, W, H]
            for i in range(0, len(loss), query_bs):
                ed = min(len(loss), i+query_bs)
                loss[i:ed] += ((encoder(inp_cache[i:ed]).to(device) - target) ** 2).view(ed-i, -1).mean(dim=-1)

            for i in range(gen_num):
                st = i * len(noise)
                ed = st + len(noise)
                inp_cache[st : ed] = self._internal_trans_inv(xs[i].unsqueeze(0) - noi_eps) # shape: [B, C, W, H]
            for i in range(0, len(loss), query_bs):
                ed = min(len(loss), i+query_bs)
                loss[i:ed] -= ((encoder(inp_cache[i:ed]).to(device) - target) ** 2).view(ed-i, -1).mean(dim=-1)

            loss = loss.reshape(gen_num, len(noise), *((1,)*(len(noise.shape)-1)))
            factor = (3 * size[0] * size[1]) / (2 * self.gd_est_eps)

            lr = scheduler.get_lr()
            for i in range(gen_num):
                gd = (loss[i] * noise).mean(dim=0, keepdims=False) * factor
                xs[i].add_(gd, alpha= -lr / gd.abs().max())
            xs.clamp_(0, 1)
            scheduler.step()

            """ two-point zeroth-order gradient estimation END """

            if logger is not None:
                inp_xs = self._internal_trans_inv(xs)
                ori_loss = ((encoder(inp_xs).to(device) - target) ** 2).view(len(xs), -1).mean(dim=-1)
                logger.info("step [{}/{}]: lr = {:.3e}".format(step+1, self.steps, lr))
                logger.info("loss: avg = {:.3e}, min = {:.3e}, max = {:.3e}"
                            .format(ori_loss.mean().item(), ori_loss.min().item(), ori_loss.max().item()))

                # if classifier is not None:
                #     orig_y = classifier(target).softmax(dim=-1)
                #     pred_y = classifier(encoder(inp_x)).softmax(dim=-1)
                #     logger.info("[orig]: label = {}, simplex = {}".format(orig_y.argmax(dim=-1)[0].item(), orig_y))
                #     logger.info("[pred]: label = {}, simplex = {}".format(pred_y.argmax(dim=-1)[0].item(), pred_y))

                logger.info("")

        xs = self._internal_trans_inv(xs)
        return xs

    @staticmethod
    def _internal_trans_inv(x: torch.Tensor) -> np.ndarray:
        x = (x * 255).round().clamp(0, 255).type(torch.uint8)
        x = x.permute([0, 2, 3, 1]).cpu().numpy()
        return x
