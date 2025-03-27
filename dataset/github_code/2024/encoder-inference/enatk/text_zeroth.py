from typing import List, Union
import numpy as np
import torch

from .utils import get_scheduler


""" Designed based on
    `Fast Adversarial Attacks on Language Models In One GPU Minute <https://arxiv.org/abs/2402.15570>`_
"""
class TextAtkZerothV2:
    def __init__(
        self,
        length: int,
        chars: str,
        beam_k1: int,
        beam_k2: int,
        device="cuda:0",
        logger=None,
    ):

        self.length = length
        self.chars = chars

        self.beam_k1 = beam_k1
        self.beam_k2 = min(beam_k2, len(chars))

        self.device = device
        self.logger = logger

    """ Return text? that its representation is very similar to `target`.

    Args:
        encoder: Pretrained encoder.
        target (torch.Tensor): Targeted representation.
    """
    @torch.inference_mode()
    def attack(
        self,
        encoder,
        target: torch.Tensor,
        gen_num: int = 1,
        query_bs: int = None,
    ) -> dict:

        length = self.length
        device = self.device
        logger = self.logger
        chs = self.chars

        beam_k1 = self.beam_k1
        beam_k2 = self.beam_k2

        if query_bs is None:
            query_bs = self.gd_est_num * gen_num

        if logger is not None:
            logger.info("length: {}".format(length))
            logger.info("chs: '{}'".format(chs))

        candidate_batches = [[""] for i in range(gen_num)]

        for step in range(self.length):
            inp_cache = []
            candi_num = []
            for batch in candidate_batches:
                candi_num.append(len(batch))
                for candi in batch:
                    indices = np.random.permutation(np.arange(len(chs)))
                    for i in range(beam_k2):
                        inp_cache.append(candi + chs[indices[i]])

            loss = torch.zeros(len(inp_cache), dtype=torch.float32, device=device)
            for i in range(0, len(loss), query_bs):
                ed = min(len(loss), i+query_bs)
                loss[i:ed] = ((encoder(inp_cache[i:ed]).to(device) - target) ** 2).view(ed-i, -1).mean(dim=-1)

            candidate_batches = []
            start = 0
            for i in range(gen_num):
                end = start + candi_num[i] * beam_k2
                _, indices = torch.topk(loss[start : end], min(beam_k1, end-start), largest=False, sorted=True)

                batch = []
                for idx in indices:
                    batch.append(inp_cache[start + idx])

                candidate_batches.append(batch)
                start = end

            if logger is not None:
                inp_xs = [batch[0] for batch in candidate_batches]
                ori_loss = ((encoder(inp_xs).to(device) - target) ** 2).mean(dim=-1)
                logger.info("step [{}/{}]:".format(step+1, self.length))
                logger.info("all top-1 losses: avg = {:.3e}, min = {:.3e}, max = {:.3e}"
                            .format(ori_loss.mean().item(), ori_loss.min().item(), ori_loss.max().item()))

        texts = [batch[0] for batch in candidate_batches]

        return {"vocabs": chs,
                "texts": texts,}


""" Designed based on
    `Gradient-based Adversarial Attacks against Text Transformers <https://arxiv.org/abs/2104.13733>`_
    Below method has been DEPRECATED!
"""
class TextAtkZeroth:
    def __init__(
        self,
        steps: int,
        scheduler: str,
        scheduler_pms: dict,
        length: int,
        chars: str,
        value_est_num: int,
        gd_est_eps: float,
        gd_est_num: int,
        device="cuda:0",
        logger=None,
    ):
        raise RuntimeError("This method has been DEPRECATED!")

        self.steps = steps
        self.scheduler = get_scheduler(scheduler, **scheduler_pms)

        self.length = length
        self.chars = chars

        self.value_est_num = value_est_num
        self.gd_est_eps = gd_est_eps
        self.gd_est_num = gd_est_num

        self.device = device
        self.logger = logger

    """ Return text? that its representation is very similar to `target`.

    Args:
        encoder: Pretrained encoder.
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
    ) -> dict:

        length = self.length
        device = self.device
        logger = self.logger
        chs = self.chars

        if query_bs is None:
            query_bs = self.gd_est_num * gen_num

        if logger is not None:
            logger.info("length: {}".format(length))
            logger.info("chs: '{}'".format(chs))

        chs_ary = np.chararray((len(chs),))
        chs_ary[:] = [*chs]

        logits = torch.empty((gen_num, length, len(chs)), device=device).normal_() # shape: [L, V]

        original_texts = self._internal_sampling(logits, 1, chs_ary)
        original_logits = logits.clone().cpu().numpy()

        scheduler = self.scheduler
        scheduler.reset()

        for step in range(self.steps):
            """ two-point zeroth-order gradient estimation BEGIN """
            # generate (self.gd_est_num) noise instead of (self.gd_est_num*len(logits))
            # to reduce memory usage
            noise = torch.rand([self.gd_est_num, *logits.shape[1:]], dtype=torch.float32, device=device) - 0.5
            norm = (noise.view(len(noise), -1) ** 2).sum(dim=-1).sqrt().view(len(noise), *((1,)*(len(noise.shape)-1)))
            noise = noise / norm # shape: [gd_est_num, L, V]

            noi_eps = noise * self.gd_est_eps
            inp_cache1, inp_cache2 = [], []
            loss1 = torch.zeros(len(noise)*gen_num*self.value_est_num, dtype=torch.float32, device=device)
            loss2 = torch.zeros(len(noise)*gen_num*self.value_est_num, dtype=torch.float32, device=device)

            for i in range(gen_num):
                black1, black2 = self._internal_symmetric_sampling(
                    logits1       = logits[i] + noi_eps,
                    logits2       = logits[i] - noi_eps,
                    per_samp_num  = self.value_est_num,
                    chs_ary       = chs_ary,
                )
                inp_cache1 += black1
                inp_cache2 += black2

            for i in range(0, len(loss1), query_bs):
                ed = min(len(loss1), i+query_bs)
                loss1[i:ed] = ((encoder(inp_cache1[i:ed]).to(device) - target) ** 2).view(ed-i, -1).mean(dim=-1)
                loss2[i:ed] = ((encoder(inp_cache2[i:ed]).to(device) - target) ** 2).view(ed-i, -1).mean(dim=-1)

            loss_shape = (gen_num, len(noise), self.value_est_num,) + (1,) * (len(noise.shape) - 1)
            loss1 = loss1.reshape(loss_shape).mean(dim=2) # shape: [gen_num, len(noise), ...]
            loss2 = loss2.reshape(loss_shape).mean(dim=2) # shape: [gen_num, len(noise), ...]
            factor = (logits.shape[0] * logits.shape[1]) / (2 * self.gd_est_eps)

            lr = scheduler.get_lr()
            for i in range(gen_num):
                gd = ((loss1[i] - loss2[i]) * noise).mean(dim=0, keepdims=False) * factor
                logits[i].add_(gd, alpha= -lr / gd.abs().max())
            scheduler.step()

            """ two-point zeroth-order gradient estimation END """

            if logger is not None:
                inp_xs = self._internal_sampling(logits, self.value_est_num, chs_ary)
                ori_loss = ((encoder(inp_xs).to(device) - target) ** 2).view(len(logits), -1).mean(dim=-1)
                logger.info("step [{}/{}]: lr = {:.3e}".format(step+1, self.steps, lr))
                logger.info("loss: avg = {:.3e}, min = {:.3e}, max = {:.3e}"
                            .format(ori_loss.mean().item(), ori_loss.min().item(), ori_loss.max().item()))

        texts = self._internal_sampling(logits, 1, chs_ary)
        logits = logits.cpu().numpy()

        return {"vocabs": chs,
                "original_logits": original_logits,
                "original_texts": original_texts,
                "logits": logits,
                "texts": texts,}

    @staticmethod
    def _internal_symmetric_sampling(logits1: torch.Tensor, logits2: torch.Tensor, per_samp_num: int, chs_ary: np.chararray) -> List[str]:
        B, L, V = logits1.shape
        gumbels = torch.empty([B, per_samp_num, L, V], device=logits1.device).exponential_().log()

        # gumbel-max trick
        tokens1 = (gumbels + logits1.unsqueeze(1)).argmax(dim=-1)
        indices1 = tokens1.view(B * per_samp_num, L).type(torch.uint8).to("cpu").numpy() # shape: [B*per_samp_num, L]
        raw_samples1 = chs_ary[indices1]

        tokens2 = (gumbels + logits2.unsqueeze(1)).argmax(dim=-1)
        indices2 = tokens2.view(B * per_samp_num, L).type(torch.uint8).to("cpu").numpy() # shape: [B*per_samp_num, L]
        raw_samples2 = chs_ary[indices2]

        samples1, samples2 = [], []
        for samp1, samp2 in zip(raw_samples1, raw_samples2):
            samples1.append(samp1.tobytes().decode("ascii"))
            samples2.append(samp2.tobytes().decode("ascii"))

        return samples1, samples2 # shape: [B*per_samp_num, *]
 

    """ logits should be in shape (B, L, V), where B is the batch-size,
        L is the number of different tokens, and V is the world-size.
    """
    @staticmethod
    def _internal_sampling(logits: Union[torch.Tensor, np.ndarray], per_samp_num: int, chs_ary: np.chararray) -> List[str]:
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32)
        B, L, V = logits.shape

        # gumbel-max trick
        gumbels = torch.empty([B, per_samp_num, L, V], device=logits.device).exponential_().log() + logits.unsqueeze(1)
        tokens = gumbels.argmax(dim=-1) # shape: [B, per_samp_num, L]; dtype: torch.int64

        indices = tokens.view(B * per_samp_num, L).type(torch.uint8).to("cpu").numpy() # shape: [B*per_samp_num, L]
        raw_samples = chs_ary[indices]

        samples = [samp.tobytes().decode("ascii") for samp in raw_samples]
        return samples # len(samples) = B * per_samp_num
