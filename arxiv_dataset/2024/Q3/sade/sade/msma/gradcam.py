import functools
import gc
import pdb
import sys

import numpy as np
import torch
from captum.attr import GuidedBackprop, LayerGradCam
from captum.attr._utils.attribution import LayerAttribution
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from sade.configs.ve import biggan_config
from sade.datasets.loaders import get_dataloaders
from sade.models import registry
from sade.models.distributions import GMM
from sade.msma.flow_model import FlowModel
from sade.run.utils import restore_pretrained_weights


class ScoreFlow(torch.nn.Module):
    def __init__(self, net, flow):
        super().__init__()
        self.net = net
        self.flow_model = flow

    def forward(self, x, t):
        b = x.shape[0]
        out = self.net(x, t).view(1, b, -1)
        out = torch.linalg.norm(out, dim=-1, keepdim=False)
        z, ldj = self.flow_model.flow(out)
        out = self.flow_model.base_distribution(z) + ldj
        out = -1 * out.view(-1, 1)
        return out


class ScoreDensity(torch.nn.Module):
    def __init__(self, net, density_model):
        super().__init__()
        self.net = net
        self.density_model = density_model

        # Checkpoint a few forwards for memory efficient backprop
        for dblock in self.net.down_layers.values():
            dblock.forward = functools.partial(
                checkpoint, dblock.forward, use_reentrant=False
            )

        # for upblock in scoreflow.net.down_layers.values():
        #     upblock.forward = functools.partial(
        #     checkpoint, upblock.forward , use_reentrant=False
        # )

    def forward(self, x, t):
        b = x.shape[0]
        out = self.net(x, t).view(1, b, -1)
        out = torch.linalg.norm(out, dim=-1, keepdim=False)
        out = self.density_model(out)

        out = -1 * out.view(-1, 1)
        return out


def run(config, workdir):
    # Initialize main model.

    n_timesteps = config.msma.n_timesteps = 10
    config.model.act = "relu" # This is important for the guided backpropagation to work
    score_model = registry.create_model(config, distributed=False)
    timesteps = registry.get_msma_sigmas(config)

    state = dict(model=score_model, step=0)
    state = restore_pretrained_weights(
        config.training.pretrained_checkpoint, state, config.device
    )
    score_model = score_model.eval()

    if "flow" == config.msma.density_model:
        density_model = FlowModel(config.msma.n_timesteps, device="cuda")
    elif "GMM" == config.msma.density_model:
        density_model = GMM(10, config.msma.n_timesteps).cuda()

    # density_model = density_model.eval()
    density_model_state = torch.load(
        f"{workdir}/msma/v0_{config.msma.density_model.lower()}_checkpoint.pth"
    )
    density_model.load_state_dict(density_model_state["model_state_dict"], strict=True)
    scoreflow = ScoreDensity(score_model, density_model)

    gb = GuidedBackprop(scoreflow)
    stop_layer = scoreflow.net.up_layers[-1]
    # pdb.set_trace()
    # stop_layer = scoreflow.net.conv_final[-1]

    # stop_layer.forward = functools.partial(checkpoint, stop_layer.forward, use_reentrant=False)
    layer_gc = LayerGradCam(scoreflow, stop_layer)

    # Build data iterator
    experiment_name = f"{experiment.inlier}_{experiment.ood}-relu"

    enhance_lesions = False
    if "-enhanced" in experiment.ood:
        enhance_lesions = True
        experiment.ood = experiment.ood.split("-")[0]

    config.eval.batch_size = 1
    config.training.batch_size = 1
    _, datasets = get_dataloaders(
        config,
        evaluation=True,
        num_workers=1,
        infinite_sampler=False,
    )

    _, inlier_ds, ood_ds = datasets

    inlier_ds = torch.utils.data.Subset(inlier_ds, list(range(4)))
    # ood_ds = torch.utils.data.Subset(ood_ds, list(range(4)))

    x_inlier_attrs = []
    x_ood_attrs = []

    for res_arr, ds in zip([x_inlier_attrs, x_ood_attrs], [inlier_ds, ood_ds]):
        ds_iter = tqdm(iter(ds))
        for x_batch in ds_iter:
            x = x_batch["image"].cuda().unsqueeze(0)

            if enhance_lesions and "label" in x_batch:
                # print("Enhancing tumors...")
                labels = x_batch["label"].cuda()
                x = x * labels * 1.5 + x * (1 - labels)
                labels = labels.cpu()

            x = torch.repeat_interleave(x, n_timesteps, dim=0)
            guided_backprop_attr = gb.attribute(
                x, target=None, additional_forward_args=timesteps
            )
            guided_backprop_attr = guided_backprop_attr.detach().cpu()
            # print("Finished Backprop:", guided_backprop_attr.shape)
            x = x.detach()
            x_backprop = guided_backprop_attr.sum(0).sum(0)
            # torch.cuda.empty_cache()
            grad_cam_attr = layer_gc.attribute(
                (x),
                target=None,
                additional_forward_args=timesteps,
                attribute_to_layer_input=True,
                relu_attributions=True,
            )[0]
            x_grad_cam = grad_cam_attr.detach().cpu()
            x_grad_cam = LayerAttribution.interpolate(
                x_grad_cam,
                x.shape[2:],
            )
            x_grad_cam = x_grad_cam.sum(0)[0]
            x = x.cpu()
            x_backprop = x_backprop.cpu()
            x_grad_cam = x_grad_cam.cpu()
            # pdb.set_trace()
            x_guided_grad_cam = x_grad_cam * x_backprop
            res_arr.append(x_guided_grad_cam)
            # print("Finished GradCam:", grad_cam_attr.shape)

            del x, x_grad_cam, x_backprop
            scoreflow.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    x_inlier_attrs = torch.stack(x_inlier_attrs)
    x_ood_attrs = torch.stack(x_ood_attrs)

    np.savez_compressed(
        f"{workdir}/msma/gradcam/{experiment_name}_results.npz",
        **{"ood": x_ood_attrs, "inliers": x_inlier_attrs},
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from sade.configs.ve import biggan_config

    workdir = sys.argv[1]

    config = biggan_config.get_config()
    config.fp16 = False
    experiment = config.eval.experiment
    experiment.train = "abcd-val"  # The dataset used for training MSMA
    experiment.inlier = "abcd-test"
    experiment.ood = "lesion_load_20-enhanced"
    config.msma.density_model = "GMM"

    run(config, workdir)
