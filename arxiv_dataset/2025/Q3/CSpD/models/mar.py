from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss

__all__ = ["mar_base", "mar_large", "mar_huge"]


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device='cuda')
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len],
                            src=torch.ones(bsz, seq_len, device='cuda')).bool()
    return masking


class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps="100",
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size ** 2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.device, x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)
        # dropping
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def enc_dec(self, tokens, mask, class_embedding):
        # mae encoder
        x = self.forward_mae_encoder(tokens, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        return z

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # encoder and decoder
        z = self.enc_dec(x, mask, class_embedding)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0,
                      progress=False):
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len, device='cuda')
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device='cuda')
        noise = torch.randn(bsz, self.seq_len, self.diffloss.in_channels, device='cuda')
        orders = self.sample_orders(bsz)

        # get class embedding
        class_embedding = self.class_embed_cfg(bsz, labels, cfg)

        # add cfg to noise
        noise, = self.expand_batch(noise, cfg=cfg)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # generate latents
        for step in indices:
            tokens, mask, _ = self.sample_tokens_step(tokens, mask, noise, None, orders, bsz, class_embedding,
                                                      step, num_iter, cfg_schedule, temperature, cfg)
        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

    def sample_tokens_step(self,
                           tokens,
                           mask,
                           noise,
                           inter_noise,
                           orders,
                           bsz,
                           class_embedding,
                           step,
                           num_iter,
                           cfg_schedule,
                           temperature,
                           cfg):
        cur_tokens = tokens

        # class embedding and CFG
        tokens, mask = self.expand_batch(tokens, mask, cfg=cfg)

        z = self.enc_dec(tokens, mask, class_embedding)

        mask_next, mask_len = self.generate_next_mask(previous_mask=mask, step=step, num_iter=num_iter,
                                                      orders=orders, bsz=bsz, device=z.device)
        mask_to_pred = self.generate_mask_pred(step, num_iter, bsz,
                                               previous_mask=mask, mask_next=mask_next, cfg=cfg)
        index_to_pred = mask_to_pred.nonzero(as_tuple=True)

        # sample token latents and noise for this step
        z = z[index_to_pred]
        n_eps = noise[index_to_pred]
        if inter_noise is not None:
            n_i = inter_noise[index_to_pred].transpose(0, 1)
        else:
            n_i = None

        # cfg schedule follow Muse
        cfg_iter = self.cfg_schedule(cfg_schedule=cfg_schedule, cfg=cfg, mask_len=mask_len)

        sampled_token_latent = self.diffloss.sample_with_inter_noise(z, n_eps, n_i, temperature, cfg_iter)

        # Remove null class samples
        sampled_token_latent, mask_to_pred, mask_len = self.recover_batch(sampled_token_latent, mask_to_pred,
                                                                          mask_len,
                                                                          cfg=cfg)
        index_to_pred = mask_to_pred.nonzero(as_tuple=True)
        cur_tokens[index_to_pred] = sampled_token_latent

        aux = dict(
            index_to_pred=index_to_pred,
            mask_len=mask_len
        )

        return cur_tokens, mask_next, aux

    def sample_tokens_step_with_mask(self,
                                     tokens,
                                     mask,
                                     noise,
                                     inter_noise,
                                     orders,
                                     bsz,
                                     class_embedding,
                                     step,
                                     num_iter,
                                     mask_next,
                                     mask_len,
                                     cfg_schedule,
                                     temperature,
                                     cfg):
        cur_tokens = tokens

        # class embedding and CFG
        tokens, mask = self.expand_batch(tokens, mask, cfg=cfg)

        z = self.enc_dec(tokens, mask, class_embedding)

        mask_to_pred = self.generate_mask_pred(step, num_iter, bsz,
                                               previous_mask=mask, mask_next=mask_next, cfg=cfg)
        index_to_pred = mask_to_pred.nonzero(as_tuple=True)

        # sample token latents and noise for this step
        z = z[index_to_pred]
        n_eps = noise[index_to_pred]
        n_i = inter_noise[index_to_pred].transpose(0, 1)

        # cfg schedule follow Muse
        cfg_iter = self.cfg_schedule(cfg_schedule=cfg_schedule, cfg=cfg, mask_len=mask_len)

        sampled_token_latent = self.diffloss.sample_with_inter_noise(z, n_eps, n_i, temperature, cfg_iter)
        # Remove null class samples
        sampled_token_latent, mask_to_pred, mask_len = self.recover_batch(sampled_token_latent, mask_to_pred,
                                                                          mask_len,
                                                                          cfg=cfg)
        index_to_pred = mask_to_pred.nonzero(as_tuple=True)
        cur_tokens[index_to_pred] = sampled_token_latent

        aux = dict(
            index_to_pred=index_to_pred,
            mask_len=mask_len
        )

        return cur_tokens, mask_next, aux

    def class_embed_cfg(self, bsz, labels, cfg):
        # get clas embedding
        if labels is not None:
            class_embedding = self.class_emb(labels)
        else:
            class_embedding = self.fake_latent.repeat(bsz, 1)
        if cfg != 1.0:
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

        return class_embedding

    def generate_next_mask(self, previous_mask, step, num_iter, orders, bsz, device):
        """mask ratio for the next round, following MaskGIT and MAGE."""
        mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
        mask_len = torch.tensor([np.floor(self.seq_len * mask_ratio)], device='cuda')

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.tensor([1], device=device),
                                 torch.minimum(torch.sum(previous_mask, dim=-1, keepdims=True) - 1, mask_len)).long()

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)

        return mask_next, mask_len

    @staticmethod
    def generate_mask_pred(step, num_iter, bsz, previous_mask, mask_next, cfg):
        if step >= num_iter - 1:
            mask_to_pred = previous_mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(previous_mask[:bsz].bool(), mask_next.bool())

        if cfg != 1.0:
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        return mask_to_pred

    def cfg_schedule(self, cfg_schedule, cfg, mask_len):
        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
        elif cfg_schedule == "constant":
            cfg_iter = cfg
        else:
            raise NotImplementedError
        return cfg_iter

    @staticmethod
    def expand_batch(*args, cfg: float):
        """expand batch for class embedding and CFG"""
        return [torch.cat([obj, obj], dim=0) for obj in args] if cfg != 1.0 else args

    @staticmethod
    def recover_batch(*args, cfg: float):
        """recover batch for  class embedding and CFG"""
        return [obj.chunk(2, dim=0)[0] for obj in args] if cfg != 1 else args

    def get_distribution(self):
        return self.diffloss.gen_diffusion.manager


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        diffloss_d=6, diffloss_w=1024, **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        diffloss_d=8, diffloss_w=1280, **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        diffloss_d=12, diffloss_w=1536, **kwargs)
    return model
