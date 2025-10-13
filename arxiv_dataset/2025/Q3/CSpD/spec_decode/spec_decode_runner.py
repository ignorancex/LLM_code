from typing import Optional
import math
import torch
from tqdm import tqdm
from models.mar import MAR


class DistributionProcessor:
    def __init__(self, x1_mean, x1_std, x1_log_var, x2_mean, x2_std, x2_log_var, dim, temperature=42.):
        """x has shape BC. x1 for q(x), x2 for p(x)."""
        self.x1_mean = x1_mean
        self.x1_std = x1_std
        self.x1_log_var = x1_log_var
        self.x2_mean = x2_mean
        self.x2_std = x2_std
        self.x2_log_var = x2_log_var
        self.temperature = temperature
        self.dim = dim
        self.fixed_term = torch.exp((x2_log_var - x1_log_var).sum(dim=-1))

    def x1_pdf_unormalize(self, x):
        return torch.exp(-0.5 * (((x - self.x1_mean) / self.x1_std / self.temperature) ** 2).sum(-1))

    def x2_pdf_unormalize(self, x):
        return torch.exp(-0.5 * (((x - self.x2_mean) / self.x2_std / self.temperature) ** 2).sum(-1))

    def x1_pdf(self, x):
        return torch.exp(-0.5 * (((x - self.x1_mean) / self.x1_std) ** 2).sum(-1)) / (
                (2 * math.pi) ** (self.dim / 2) * self.x1_std.prod(-1))

    def x2_pdf(self, x):
        return torch.exp(-0.5 * (((x - self.x2_mean) / self.x2_std) ** 2).sum(-1)) / (
                (2 * math.pi) ** (self.dim / 2) * self.x2_std.prod(-1))

    def judge_accept(self, x):
        q_x = self.x1_pdf_unormalize(x)
        p_x = self.x2_pdf_unormalize(x)
        return p_x * self.fixed_term * self.x1_std.prod(-1) / (q_x*self.x2_std.prod(-1))

    def subtracted_distribution_unormalize(self, x):
        ret = self.x2_pdf(x) - self.x1_pdf(x)/self.fixed_term
        proposal = self.x2_pdf(x)
        return ret.clamp(min=0)/proposal

    def sample_distribution(self):
        mask = torch.zeros((self.x2_mean.shape[0],), device=self.x2_mean.device).bool()
        sample = torch.zeros_like(self.x2_mean)
        for i in range(100):  # try at most 100 times
            proposal = torch.randn_like(self.x2_mean) * self.x2_std + self.x2_mean
            alpha = self.subtracted_distribution_unormalize(proposal)
            eps = torch.rand_like(alpha)

            accept_mask = eps < alpha
            sample[accept_mask] = proposal[accept_mask]
            mask = torch.logical_or(accept_mask, mask)

            if torch.all(mask):
                break
        return sample

    def speculative_sample(self, x):
        prob = self.judge_accept(x)
        guess = torch.rand_like(prob)
        accept_mask = prob > guess
        if torch.all(accept_mask):
            return x

        return torch.where(accept_mask.unsqueeze(-1), x, self.sample_distribution())


class SpecDecodeRunner:
    def __init__(self, draft_model: MAR, target_model: MAR, inner_temperature: float = 42.,ratio:float=0.15):
        self.draft_model = draft_model
        self.target_model = target_model
        self.seq_len = draft_model.seq_len
        self.token_embed_dim = draft_model.token_embed_dim
        self.sample_orders = draft_model.sample_orders
        self.unpatchify = draft_model.unpatchify
        self.temperature = inner_temperature
        self.ratio = ratio

    def run(self,
            bsz: int,
            num_iter=64,
            num_draft=8,
            cfg_schedule="linear",
            labels=Optional[torch.LongTensor],
            temperature=1.0,
            cfg=1.0,
            progress=False):
        mask_draft = torch.ones(bsz, self.seq_len, device='cuda')
        mask_target = torch.ones(bsz, self.seq_len, device='cuda')

        draft = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device='cuda')
        draft_mean_map = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device='cuda')
        draft_var_map = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device='cuda')
        draft_log_var_map = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device='cuda')

        # accept_map = torch.ones(bsz, self.seq_len, device='cuda',dtype=torch.bool)

        noise = torch.randn(bsz, self.seq_len, self.draft_model.diffloss.in_channels, device='cuda')
        ns = self.draft_model.diffloss.gen_diffusion.num_timesteps
        inter_noise = torch.randn(bsz, self.seq_len, ns, self.draft_model.diffloss.in_channels, device='cuda')
        orders = self.sample_orders(bsz)

        # get class embedding
        draft_class_embedding = self.draft_model.class_embed_cfg(bsz, labels, cfg)
        target_class_embedding = self.target_model.class_embed_cfg(bsz, labels, cfg)

        # expand batch for cfg to noise
        noise, inter_noise = self.draft_model.expand_batch(noise, inter_noise, cfg=cfg)

        indices = list(range(num_iter))
        milestone = num_iter * self.ratio
        if progress:
            indices = tqdm(indices)

        for step in indices:
            if step <= milestone:
                draft, mask_draft, draft_aux = self.target_model.sample_tokens_step(
                    draft,
                    mask_draft,
                    noise,
                    inter_noise,
                    orders, bsz,
                    target_class_embedding,
                    step, num_iter,
                    cfg_schedule, temperature,
                    cfg)
                mask_target = mask_draft.clone()
            else:
                # one draft step
                draft, mask_draft, draft_aux = self.draft_model.sample_tokens_step(
                    draft,
                    mask_draft,
                    noise,
                    inter_noise,
                    orders, bsz,
                    draft_class_embedding,
                    step, num_iter,
                    cfg_schedule, temperature,
                    cfg)

                draft_distribution = self.draft_model.get_distribution()
                draft_distribution.mean, draft_distribution.variance, draft_distribution.log_var, draft_distribution.sample = self.draft_model.recover_batch(
                    draft_distribution.mean,
                    draft_distribution.variance,
                    draft_distribution.log_var,
                    draft_distribution.sample,
                    cfg=cfg)
                draft_mean_map[draft_aux['index_to_pred']] = draft_distribution.mean
                draft_var_map[draft_aux['index_to_pred']] = draft_distribution.variance
                draft_log_var_map[draft_aux['index_to_pred']] = draft_distribution.log_var

                if (step % num_draft == 0 and step != 0) or step == num_iter - 1:  # do not verify at step 0
                    # one target step on tokens from draft
                    target, mask_target, target_aux = self.target_model.sample_tokens_step_with_mask(
                        draft.clone(),
                        mask_target,
                        noise,
                        inter_noise,
                        orders,
                        bsz,
                        target_class_embedding,
                        step,
                        num_iter,
                        mask_draft,
                        draft_aux['mask_len'],
                        cfg_schedule,
                        temperature,
                        cfg)
                    index_to_pred = target_aux['index_to_pred']

                    target_distribution = self.target_model.get_distribution()
                    target_distribution.mean, target_distribution.variance, target_distribution.log_var, target_distribution.sample = self.target_model.recover_batch(
                        target_distribution.mean,
                        target_distribution.variance,
                        target_distribution.log_var,
                        target_distribution.sample,
                        cfg=cfg)

                    draft_samples = draft[index_to_pred]
                    draft_mean = draft_mean_map[index_to_pred]
                    draft_var = draft_var_map[index_to_pred]
                    draft_log_var = draft_log_var_map[index_to_pred]
                    target_mean = target_distribution.mean
                    target_var = target_distribution.variance
                    target_log_var = target_distribution.log_var

                    processor = DistributionProcessor(
                        x1_mean=draft_mean,
                        x1_std=draft_var,
                        x1_log_var=draft_log_var,
                        x2_mean=target_mean,
                        x2_std=target_var,
                        x2_log_var=target_log_var,
                        dim=self.token_embed_dim,
                        temperature=self.temperature
                    )
                    draft[index_to_pred] = processor.speculative_sample(draft_samples)
        # unpatchify
        tokens = self.unpatchify(draft)
        return tokens
