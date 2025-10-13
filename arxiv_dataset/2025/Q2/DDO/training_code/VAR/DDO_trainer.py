from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

import torch_fidelity

import os
import PIL.Image as PImage
from tqdm import tqdm
import numpy as np

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VAR_DDOTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, ref_var_wo_ddp: VAR, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        alpha: float, beta: float, uncond_ratio: float, # DDO parameters
    ):
        super(VAR_DDOTrainer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.uncond_ratio = uncond_ratio

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.ref_var_wo_ddp: VAR = ref_var_wo_ddp  # after torch.compile
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        del self.ref_var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        self.ref_var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_type_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn


    @torch.no_grad()
    def evaluate(self, cfg, outdir, seed=1):
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        sample_folder_dir = f"{outdir}/tmp"
        os.makedirs(sample_folder_dir, exist_ok=True)

        total_classes = 1000
        rank_classes = np.array_split(np.arange(total_classes), dist.get_world_size())[dist.get_rank()]
        device = dist.get_device()

        # sample
        B = 25
        for img_cls in tqdm(rank_classes, disable=(dist.get_rank() != 0)):
            for i in range(50 // B):
                label_B = torch.tensor([img_cls] * B, device=device)
                with torch.inference_mode():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                        recon_B3HW = self.var_wo_ddp.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=0, top_p=0, more_smooth=False, g_seed=int(seed+img_cls*(50 // B)+i))
                    bchw = recon_B3HW.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
                bchw = bchw.astype(np.uint8)
                for j in range(B):
                    img = PImage.fromarray(bchw[j])
                    img.save(f"{sample_folder_dir}/{(img_cls * 50 + i * B + j):06d}.png")
        dist.barrier()

        if dist.get_rank() == 0:
            fid_statistics_file = "fid_stats/adm_in256_stats.npz"
            metrics_dict = torch_fidelity.calculate_metrics(
                    input1=sample_folder_dir,
                    input2=None,
                    fid_statistics_file=fid_statistics_file,
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=False,
                )
            fid = metrics_dict["frechet_inception_distance"]
            inception_score = metrics_dict["inception_score_mean"]
        else:
            fid, inception_score = 0, 0

        dist.barrier()

        self.var_wo_ddp.train(training)
        return fid, inception_score
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen],
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)# inp_B3HW <bz,3,256,256> --> gt_idx_Bl [[x]*bz, [x,x,x,x]*bz, ....  ] 
        gt_BL = torch.cat(gt_idx_Bl, dim=1) # <bz, 680>
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # <bz, 679, 32>

        # Generate fake samples from the reference model online
        label_B_fake = label_B
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=False, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                inp_B3HW_fake = self.ref_var_wo_ddp.autoregressive_infer_cfg(B=B, label_B=label_B_fake, cfg=1.0, top_k=0, top_p=0, more_smooth=False, g_seed=g_it * dist.get_world_size() + dist.get_rank()) * 2 - 1

        gt_idx_Bl_fake: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW_fake)# inp_B3HW <bz,3,256,256> --> gt_idx_Bl [[x]*bz, [x,x,x,x]*bz, ....  ] 
        gt_BL_fake = torch.cat(gt_idx_Bl_fake, dim=1) # <bz, 680>
        x_BLCv_wo_first_l_fake: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl_fake) # <bz, 679, 32>
        
        # randomly mask conditions to maintain unconditional distribution and thus enable CFG sampling
        uncond_mask = torch.rand(label_B.shape[0], device=label_B.device) < self.uncond_ratio
        label_B = torch.where(uncond_mask, 1000, label_B) # 1000 is the uncond condition
        label_B_fake = torch.where(uncond_mask, 1000, label_B_fake) # 1000 is the uncond condition

        # concat real and fake data
        bz = label_B.shape[0]
        all_label_B = torch.cat([label_B, label_B_fake])
        all_x_BLCv_wo_first_l = torch.cat([x_BLCv_wo_first_l, x_BLCv_wo_first_l_fake])

        with self.var_opt.amp_ctx:
            all_logits = self.var(all_label_B, all_x_BLCv_wo_first_l) # <2bz, 680, 4096>
            logits_BLV = all_logits[:bz]
            logits_BLV_fake = all_logits[bz:]
            with torch.no_grad():
                ref_all_logits = self.ref_var_wo_ddp(all_label_B, all_x_BLCv_wo_first_l) # <2bz, 680, 4096>
                ref_logits_BLV = ref_all_logits[:bz]
                ref_logits_BLV_fake = ref_all_logits[bz:]

        img_logps = torch.gather(logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        fake_img_logps = torch.gather(logits_BLV_fake.log_softmax(-1), dim=2, index=gt_BL_fake.unsqueeze(2)).squeeze(2).sum(-1)
        img_ref_logps = torch.gather(ref_logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        fake_img_ref_logps = torch.gather(ref_logits_BLV_fake.log_softmax(-1), dim=2, index=gt_BL_fake.unsqueeze(2)).squeeze(2).sum(-1)
        img_logp_gap = img_logps - img_ref_logps
        fake_img_logp_gap = fake_img_logps - fake_img_ref_logps

        real_weight = torch.ones_like(uncond_mask)
        real_weight[uncond_mask] = max(self.alpha, 1.0)
        fake_weight = torch.ones_like(uncond_mask) * self.alpha
        fake_weight[uncond_mask] = 0.0
        loss = - (real_weight * F.logsigmoid(self.beta * img_logp_gap)).mean() - (fake_weight * F.logsigmoid(-self.beta * fake_img_logp_gap)).mean()
        loss = loss / max(self.alpha, 1.0)

        with torch.no_grad():
            acc = (img_logp_gap > fake_img_logp_gap).float().mean().detach()
            margin = (img_logp_gap - fake_img_logp_gap).mean().detach()
            sftloss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()
            sftloss_ref = self.train_loss(logits_BLV_fake.view(-1, V), gt_BL_fake.view(-1)).view(B, -1).mean()
            ref_sftloss = self.train_loss(ref_logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()
            ref_sftloss_fake = self.train_loss(ref_logits_BLV_fake.view(-1, V), gt_BL_fake.view(-1)).view(B, -1).mean()

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item() if grad_norm is not None else grad_norm
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        if g_it < 50 or (g_it + 1) % 100 == 0:
            real_gap = img_logp_gap.float().mean().detach()
            fake_gap = fake_img_logp_gap.float().mean().detach()
            uncond_gap = img_logp_gap[uncond_mask].float().mean().detach()

            dist.allreduce(acc)
            dist.allreduce(real_gap)
            dist.allreduce(fake_gap)
            dist.allreduce(uncond_gap)
            dist.allreduce(margin)
            dist.allreduce(sftloss)
            dist.allreduce(sftloss_ref)
            dist.allreduce(ref_sftloss)
            dist.allreduce(ref_sftloss_fake)

            if dist.is_master():
                tb_lg.update(head='AR_iter_loss', 
                            acc=acc.float().mean().detach().item() / dist.get_world_size(),
                            real_gap=real_gap.float().mean().detach().item() / dist.get_world_size(),
                            fake_gap=fake_gap.float().mean().detach().item() / dist.get_world_size(),
                            uncond_gap=uncond_gap.float().mean().detach().item() / dist.get_world_size(),
                            margin=margin.float().mean().detach().item() / dist.get_world_size(),
                            sftloss=sftloss.float().mean().detach().item() / dist.get_world_size(),
                            sftloss_ref=sftloss_ref.float().mean().detach().item() / dist.get_world_size(),
                            ref_sftloss=ref_sftloss.float().mean().detach().item() / dist.get_world_size(),
                            ref_sftloss_fake=ref_sftloss_fake.float().mean().detach().item() / dist.get_world_size(),
                            step=g_it)
    
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp',):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state['var_wo_ddp']