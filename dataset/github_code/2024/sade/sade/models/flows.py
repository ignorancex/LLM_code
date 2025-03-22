import logging

import normflows as nf
import torch
import torch.nn as nn
from einops import rearrange, repeat

from sade.models.distributions import GMM, MVN
from sade.models.layers import PositionalEncoding3D, SpatialNorm3D
from sade.models.layerspp import FlowAttentionBlock, get_conv_layer


class PatchFlow(torch.nn.Module):
    """
    Contructs a conditional flow model that operates on patches of an image.
    Each patch is fed into the same flow model i.e. parameters are shared across patches.
    The flow models are conditioned on a positional encoding of the patch location.
    The resulting patch-densities can then be recombined into a full image density.
    """

    def __init__(
        self,
        config,
        input_size,
    ):
        super().__init__()

        param_keys = ["kernel_size", "stride", "padding"]
        for key in param_keys:
            assert key in config.flow.local_patch_config
            assert (
                key in config.flow.global_patch_config
                if config.flow.use_global_context
                else True
            )

        channels = input_size[0]
        self.device = config.device

        # Base distribution
        dist_name = config.flow.base_distribution
        if dist_name == "gaussian_mixture":
            self.base_distribution = nf.distributions.base.GaussianMixture(
                n_modes=10, dim=channels, trainable=True
            )
        elif dist_name == "multivariate_gaussian_mixture":
            self.base_distribution = GMM(n_components=10, num_features=channels)
        elif dist_name == "multivariate_normal":
            self.base_distribution = MVN(channels)
        elif dist_name == "standard":
            self.base_distribution = nf.distributions.base.DiagGaussian(
                channels, trainable=False
            )
        else:
            raise NotImplementedError(f"Distribution {dist_name} is not supported")

        # Patch parameters
        self.local_patch_config = config.flow.local_patch_config
        self.global_patch_config = config.flow.global_patch_config

        self.kernel_size = self.local_patch_config.kernel_size
        self.stride = self.local_patch_config.stride
        self.padding = self.local_patch_config.padding

        # Used to chunk the input into in fast_forward (vectorized)
        self.patch_batch_size = config.flow.patch_batch_size

        # Params for neural nets within a flow block
        self.num_blocks = config.flow.num_blocks
        self.context_embedding_size = config.flow.context_embedding_size
        self.use_global_context = config.flow.use_global_context
        self.global_embedding_size = config.flow.global_embedding_size
        self.input_norm = config.flow.input_norm

        with torch.no_grad():
            # Pooling for local "patch" flow
            # Each patch-norm is input to the shared conditional flow model
            self.local_pooler = SpatialNorm3D(
                channels, **self.local_patch_config
            ).requires_grad_(False)

            # Compute the spatial resolution of the patches
            _, self.channels, h, w, d = self.local_pooler(torch.empty(1, *input_size)).shape
            self.spatial_res = (h, w, d)
            self.num_patches = h * w * d
            self.position_encoder = PositionalEncoding3D(self.context_embedding_size)
            logging.info(
                f"Generating {self.kernel_size}x{self.kernel_size}x{self.kernel_size} patches from input size: {input_size}"
            )
            logging.info(f"Pooled spatial resolution: {self.spatial_res}")
            logging.info(f"Number of flows / patches: {self.num_patches}")

        context_dims = self.context_embedding_size

        if self.use_global_context:
            # Pooling for global context

            c = input_channels = config.data.num_channels
            self.conv_init = get_conv_layer(
                3,
                c,
                c,
                kernel_size=self.global_patch_config["kernel_size"],
                stride=self.global_patch_config["stride"],
            )
            self.conv_pooler = get_conv_layer(3, c, c, kernel_size=3, stride=2)
            self.global_pooler = nn.Sequential(
                self.conv_init,
                self.conv_pooler,
            )

            # Spatial resolution of the global context patches
            _, _, h, w, d = self.global_pooler(torch.empty(1, c, *input_size[1:])).shape
            logging.info(f"Global Context Shape: {(h, w, d)}")
            self.global_attention = FlowAttentionBlock(
                input_size=(c, h, w, d),
                embed_dim=self.global_embedding_size,
                outdim=self.context_embedding_size,
            )
            context_dims += self.context_embedding_size

        num_features = self.channels
        self.flow = self.build_flow_head(
            num_features,
            context_dims,
            # input_norm=self.input_norm,
            num_blocks=self.num_blocks,
        )

        self.init_weights()
        self.to(self.device)
        self.flow.to(self.device)

    def init_weights(self):
        # Initialize weights with Xavier
        linear_modules = list(
            filter(lambda m: isinstance(m, nn.Linear), self.flow.modules())
        )
        total = len(linear_modules)
        # pdb.set_trace()
        for idx, m in enumerate(linear_modules):
            # Last layer gets init w/ zeros
            if idx == total - 1:
                nn.init.zeros_(m.weight.data)
            else:
                nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        if self.use_global_context:
            for m in self.global_attention.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # print(m)
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

    def build_flow_head(self, input_dim, conditioning_dim, num_blocks=2):
        num_res_blocks = 2
        hidden_units = 256
        flows = []

        flows.append(
            nf.flows.MaskedAffineAutoregressive(
                input_dim,
                hidden_features=hidden_units,
                num_blocks=2,
                context_features=conditioning_dim,
            )
        )

        for i in range(num_blocks):
            flows += [nf.flows.LULinearPermute(input_dim)]
            flows += [
                nf.flows.CoupledRationalQuadraticSpline(
                    input_dim,
                    num_res_blocks,
                    hidden_units,
                    num_context_channels=conditioning_dim,
                    num_bins=8
                    # activation=nn.LeakyReLU(0.2),
                )
            ]

        # Construct flow model
        flows = flows[::-1]  # Normflow expects flows in reverse order
        nfm = nf.ConditionalNormalizingFlow(q0=self.base_distribution, flows=flows)

        return nfm

    def forward(self, x_scores, x_batch, fast=True):
        batch_size = x_scores.shape[0]
        x_norm = self.local_pooler(x_scores)
        self.position_encoder = self.position_encoder.cpu()
        context = self.position_encoder(x_norm)
        global_context = None
        
        if self.use_global_context:
            global_pooled_image = self.global_pooler(x_batch)
            # Every patch gets the same global context
            global_context = self.global_attention(global_pooled_image)

        if fast:
            zs, log_jac_dets = self.fast_forward(x_norm, context, global_context)

        else:
            # Patches x batch x channels
            local_patches = rearrange(x_norm, "b c h w d -> (h w d) b c")
            context = rearrange(context, "b c h w d -> (h w d) b c")

            zs = []
            log_jac_dets = []

            for patch_feature, context_vector in zip(local_patches, context):
                c = context_vector.to(self.device)

                if self.use_global_context:
                    c = torch.cat([c, global_context], dim=1)

                z, ldj = self.flow.inverse_and_log_det(
                    patch_feature,
                    context=c,
                )
                zs.append(z)
                log_jac_dets.append(ldj)
                c = c.cpu()

        # Use einops to concatenate all patches
        zs = torch.cat(zs, dim=0)
        log_jac_dets = torch.cat(log_jac_dets, dim=0)

        if batch_size == 1:
            zs = zs.unsqueeze(1)
            log_jac_dets = log_jac_dets.unsqueeze(1)

        zs = rearrange(zs, "n b c -> (n b) c")
        log_jac_dets = rearrange(log_jac_dets, "n b -> (n b)")

        # zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        # log_jac_dets = torch.cat(log_jac_dets, dim=0).reshape(self.num_patches, B)

        return zs, log_jac_dets

    def fast_forward(self, x, local_ctx, global_ctx=None):
        # (Patches * batch) x channels
        local_ctx = rearrange(local_ctx, "b c h w d -> (h w d) b c")
        patches = rearrange(x, "b c h w d -> (h w d) b c")

        nchunks = self.num_patches // self.patch_batch_size
        nchunks += 1 if self.num_patches % self.patch_batch_size else 0

        patches = patches.chunk(nchunks, dim=0)
        ctx_chunks = local_ctx.chunk(nchunks, dim=0)
        zs, jacs = [], []

        if global_ctx is not None:
            gc = repeat(global_ctx, "b c -> (n b) c", n=self.patch_batch_size)

        for p, ctx in zip(patches, ctx_chunks):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            ctx = ctx.to(self.device)
            ctx = rearrange(ctx, "n b c -> (n b) c")
            p = rearrange(p, "n b c -> (n b) c")

            if global_ctx is not None:
                if len(gc) != len(ctx):
                    # Only needed for the last chunk
                    c = torch.cat([ctx, gc[: ctx.shape[0]]], dim=1)
                else:
                    c = torch.cat([ctx, gc], dim=1)

            z, ldj = self.flow.inverse_and_log_det(p, context=c)

            zs.append(z)
            jacs.append(ldj)

            del ctx, p

        return zs, jacs

    def nll(self, zs, log_jac_dets):
        return -torch.mean(self.base_distribution.log_prob(zs) + log_jac_dets)

    @torch.no_grad()
    def log_density(self, x_scores, x_batch, fast=True):
        self.eval()
        b = x_scores.shape[0]
        h, w, d = self.spatial_res
        zs, jacs = self.forward(x_scores, x_batch, fast=fast)
        logpx = self.base_distribution.log_prob(zs) + jacs
        logpx = rearrange(logpx, "(h w d b) -> b h w d", b=b, h=h, w=w, d=d)
        return logpx

    @staticmethod
    def stochastic_step(scores, x_batch, flow_model, opt=None, train=False, n_patches=1):
        if train:
            flow_model.train()
            opt.zero_grad(set_to_none=True)
        else:
            flow_model.eval()

        patches, context = PatchFlow.get_random_patches(
            scores, x_batch, flow_model, n_patches
        )

        patch_feature = patches.to(flow_model.device)
        context_vector = context.to(flow_model.device)
        patch_feature = rearrange(patch_feature, "n b c -> (n b) c")
        context_vector = rearrange(context_vector, "n b c -> (n b) c")

        if flow_model.use_global_context:
            global_pooled_image = flow_model.global_pooler(x_batch)
            global_context = flow_model.global_attention(global_pooled_image)
            gctx = repeat(global_context, "b c -> (n b) c", n=n_patches)

            # Concatenate global context to local context
            context_vector = torch.cat([context_vector, gctx], dim=1)

        z, ldj = flow_model.flow.inverse_and_log_det(
            patch_feature,
            context=context_vector,
        )

        loss = flow_model.nll(z, ldj) * n_patches

        if train:
            loss.backward()
            opt.step()

        return loss.item() / n_patches

    @staticmethod
    def get_random_patches(scores, x_batch, flow_model, n_patches):
        h = flow_model.local_pooler(scores).cpu()
        flow_model.position_encoder = flow_model.position_encoder.cpu()
        local_patches = rearrange(h, "b c h w d -> (h w d) b c")
        context = rearrange(flow_model.position_encoder(h), "b c h w d -> (h w d) b c")

        # Get brain masks
        mask = x_batch > x_batch.min()

        # Generous mask that will work for all samples
        mask = mask.sum(0).sum(0) > 0
        mask = mask.flatten().cpu()

        # Get indices for patches inside the brain mask
        masked_idxs = torch.arange(flow_model.num_patches)[mask]
        # Get random patches
        shuffled_idx = torch.randperm(len(masked_idxs))
        rand_idx = masked_idxs[shuffled_idx]
        rand_idx = rand_idx[:n_patches]

        return local_patches[rand_idx], context[rand_idx]
