from typing import Any
import torch
from torch import nn
import numpy as np


#   NOTE:  copied from
#          https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/optim.py#L103
class L1Scheduler:
    def __init__(
        self,
        l1_warm_up_steps: float,
        total_steps: int,
        final_l1_coefficient: float,
    ):

        self.l1_warmup_steps = l1_warm_up_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_l1_coefficient = 0.0
        else:
            self.current_l1_coefficient = final_l1_coefficient

        self.final_l1_coefficient = final_l1_coefficient

        self.current_step = 0
        self.total_steps = total_steps
        # assert isinstance(self.final_l1_coefficient, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_l1_coefficient}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_l1_coefficient = self.final_l1_coefficient * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_l1_coefficient = self.final_l1_coefficient  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_l1_coefficient": self.current_l1_coefficient,
            "final_l1_coefficient": self.final_l1_coefficient,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class GCC(nn.Module):
    def __init__(self, input_dims, num_blocks, expansion=32, dtype=torch.float64, device="cuda", topk=None, auxk=None, dead_steps_threshold=None, jumprelu=False, squeeze=10):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = input_dims * num_blocks
        self.gcc_dims = input_dims * expansion
        self.dtype = dtype
        self.device = device
        self.squeeze = squeeze
        self.jumprelu = jumprelu
        self.topk = topk

        self.init_weights()

        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            x.data *= dead_mask
            return x

        self.auxk_mask_fn = auxk_mask_fn
        # self.stats_last_nonzero = None
        self.register_buffer("stats_last_nonzero", torch.zeros(self.gcc_dims, dtype=torch.long))

    def init_weights(self):
        self.b_enc = nn.Parameter(
            torch.zeros(self.gcc_dims, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.gcc_dims, self.output_dims, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.input_dims, self.gcc_dims, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.output_dims, dtype=self.dtype, device=self.device)
        )

        if self.jumprelu:
            self.threshold = nn.Parameter(
                torch.zeros(self.gcc_dims, dtype=self.dtype, device=self.device)
            )

    def encode(self, x):
        preact_feats = x @ self.W_enc + self.b_enc
        top_dead_acts = None
        num_dead = 0

        if self.jumprelu:
            x = nn.ReLU()(preact_feats)
            x = x * self.squeeze_sigmoid(x - self.threshold)
        elif self.topk is not None:
            topk_res = torch.topk(preact_feats, k=self.topk, dim=-1)
            values = nn.ReLU()(topk_res.values)
            x = torch.zeros_like(preact_feats)
            x.scatter_(-1, topk_res.indices, values)

            self.stats_last_nonzero *= (x == 0).all(dim=0).long()
            self.stats_last_nonzero += 1

            auxk_acts = self.auxk_mask_fn(preact_feats)
            #   TODO: ensure all batch entries have at least auxk nonzero (dead) acts.
            #   On second thought, do I even need to check this?  As there are so many latents,
            #   and only k can fire each batch, perhaps it's fairly likely that at least 512
            #   do not fire for the first epoch.
            # if (torch.sum(auxk_acts != 0, dim=-1) >= self.auxk).all(dim=0):
            num_dead = torch.mean(torch.sum(auxk_acts != 0, dim=-1).type(torch.float))
            deadk_res = torch.topk(auxk_acts, k=self.auxk, dim=-1)
            top_dead_acts = torch.zeros_like(auxk_acts)
            top_dead_acts.scatter_(-1, deadk_res.indices, deadk_res.values)
            top_dead_acts = nn.ReLU()(top_dead_acts)

        return nn.ReLU()(x), top_dead_acts, num_dead

    def squeeze_sigmoid(self, x):
        # return 1 / (1+torch.exp(-self.squeeze*x))
        return (1.01 / (1+torch.exp(-self.squeeze*x))) - 0.01

    def decode(self, x, mu, std):
        x = x @ self.W_dec + self.b_dec
        x = x * std + mu
        return x

    def forward(self, x):
        dead_acts_recon = None
        x, mu, std = LN(x)

        features, top_dead_acts, num_dead = self.encode(x)
        out = self.decode(features, mu, std)

        if top_dead_acts is not None:
            dead_acts_recon = self.decode(top_dead_acts, mu, std)

        return features, out, dead_acts_recon, num_dead


class ThresholdClipper(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'threshold'):
            module.threshold.data = torch.clamp(module.threshold.data, min=0, max=None)


#   Wrapper so that we have control over the forward pass.
class ModelWrapper(nn.Module):
    target_neuron = None
    all_acts = []

    def __init__(self, resnet, expansion, device, use_gcc=False):
        super().__init__()
        self.device = device

        self.block_input = nn.Sequential()
        self.block_input.append(resnet.conv1)
        self.block_input.append(resnet.bn1)
        self.block_input.append(resnet.relu)
        self.block_input.append(resnet.maxpool)
        self.block_input.append(resnet.layer1)
        self.block_input.append(resnet.layer2)

        self.block_input.append(resnet.layer3[:5])
        self.block_output = nn.Sequential()
        self.block_output.append(resnet.layer3[5].conv1)
        self.block_output.append(resnet.layer3[5].bn1)
        self.block_output.append(resnet.layer3[5].relu)
        self.block_output.append(resnet.layer3[5].conv2)
        self.block_output.append(resnet.layer3[5].bn2)
        self.block_output.append(resnet.layer3[5].relu)
        self.block_output.append(resnet.layer3[5].conv3)
        self.block_output.append(resnet.layer3[5].bn3)

        # self.block_input.append(resnet.layer3)
        # self.block_input.append(resnet.layer4[0])
        # self.block_input.append(resnet.layer4[1].conv1)
        # self.block_input.append(resnet.layer4[1].bn1)
        # self.block_input.append(resnet.layer4[1].relu)
        # self.block_input.append(resnet.layer4[1].conv2)
        # self.block_input.append(resnet.layer4[1].bn2)

        self.use_gcc = use_gcc
        if self.use_gcc:
            self.map = nn.Linear(1024, 1024*expansion, bias=False)

    def forward(self, x):
        x = x.to(self.device)
        x = self.block_input(x)

        presum_out = self.block_output(x)
        x = x + presum_out

        x = nn.ReLU(inplace=True)(x)
        # x, _, _ = LN(x)

        if self.use_gcc:
            center_coord = x.shape[-1] // 2
            x = x[:, :, center_coord, center_coord]

            # x = torch.mean(torch.flatten(x, start_dim=-2), dim=-1)

            x = self.map(x)
            x = x[:, :, None, None]

        return x


if __name__ == "__main__":
    # #   GROUP CROSSCODERS
    # gcc_states = torch.load('/media/andrelongon/DATA/gcc_ckpts/scale1_1.6/vanilla_32exp_gcc_weights_10ep.pth')['W_dec']
    # w_dec = torch.reshape(gcc_states, (gcc_states.shape[0], 2, 256))

    # #   How to best measure similarity?  I'd like for magnitudes to also be the same which means avoiding normalizing.
    # #   Maybe could take euclid dist for now?  Also could measure variance in mags across the blocks and penalize for
    # #   larger variance (1/var ?).
    # all_sims = []
    # for latent in w_dec:
    #     latent = torch.nn.functional.normalize(latent)

    #     # all_sims.append(torch.dot(latent[0], latent[-1]))
    #     sims = []
    #     for i in range(latent.shape[0]):
    #         for j in range(i+1, latent.shape[0]):
    #             sims.append(torch.dot(latent[i], latent[j]))

    #     all_sims.append(torch.tensor(sims).mean())

    # print(torch.topk(torch.tensor(all_sims), 32, largest=True))
    # print(torch.topk(torch.tensor(all_sims), 32, largest=False))


    #   TRANSCODERS
    tc_states = torch.load('/media/andrelongon/DATA/tc_ckpts/group_scale1_1.5_1024batch_8exp/vanilla_8exp_gtc_weights_25ep.pth')
    w_enc = torch.transpose(tc_states['W_enc'].cpu(), 0, 1)
    # w_dec = tc_states['W_dec'].cpu()
    w_dec_same = tc_states['W_dec'][:, :256].cpu()
    w_dec_zoom = tc_states['W_dec'][:, 256:].cpu()

    w_enc = torch.nn.functional.normalize(w_enc)
    # w_dec = torch.nn.functional.normalize(w_dec)
    w_dec_same = torch.nn.functional.normalize(w_dec_same)
    w_dec_zoom = torch.nn.functional.normalize(w_dec_zoom)

    all_mags = []
    all_chunk_sims = []
    all_sims = []
    for i in range(w_enc.shape[0]):
        #   TODO (for group transcoder):  weight dot by first chunk's weight mag.
        #   Or just store in separate list and print alongside top 32 cosines to pick and choose those that have relatively
        #   low ones.
        #   NOTE:  as bn2 admits negative acts into the res stream summation, should I only consider mag of positive weights?
        #   I could also try to subtract enc direction from the dec weights, then take residual mag?
        #   Since input and post-sum stream is relu'd, I should only consider the dims that directly positively interfere.

        all_mags.append(torch.norm(torch.clamp(w_dec_same[i], min=0, max=None)))
        all_chunk_sims.append(torch.abs(torch.dot(w_dec_same[i], w_dec_zoom[i])))
        all_sims.append(torch.abs(torch.dot(w_enc[i], w_dec_zoom[i])))

    all_mags = np.array(all_mags)
    # all_chunk_sims = np.array(all_chunk_sims)
    print(f'MAG MEAN: {all_mags.mean()}')
    # print(f'CHUNK SIM MEAN: {all_chunk_sims.mean()}')
    top_vals, top_idx = torch.topk(torch.tensor(all_sims), 32, largest=False)
    print(all_mags[top_idx.tolist()])
    # print(all_chunk_sims[top_idx.tolist()])
    print(top_vals, top_idx)
    # print(torch.topk(torch.tensor(all_sims), 32, largest=False))