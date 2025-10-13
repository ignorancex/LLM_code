import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from thingsvision import get_extractor, get_extractor_from_model

from group_cc import ModelWrapper
from imnet_val import get_imnet_val_acts
from tuning_curve import compare_tuning_curves, saveTopN


#   Get SAE latent acts on all imnet vals.  Compute the most "lively" (least dead) latents: those with highest % > 0 act tuning curves. 
def k_lively_latents(extractor, valdir, module_name, savedir, k=32):
    all_images, _, unrolled_acts, _, _, _ = \
        get_imnet_val_acts(extractor, '0', valdir, sort_acts=False)

    all_acts = np.transpose(np.array(unrolled_acts))
    pos_percs = np.sum(all_acts > 0, axis=-1) / all_acts.shape[-1]

    vals, top_latents = torch.topk(torch.abs(torch.tensor(pos_percs) - 0.25), k=k, largest=False)
    print(vals)
    print(top_latents)

    if len(all_acts.shape) == 2:
        Path(savedir).mkdir(parents=True, exist_ok=True)
        # for i in top_latents:
        for i in range(128):
            unrolled_act = all_acts[i].tolist()
            all_ord_list = np.arange(len(all_images)).tolist()
            all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))

            saveTopN(all_images, all_ord_sorted, f"{module_name}_neuron{i}", path=savedir, save_inh=True)

            np.save(os.path.join(savedir, f"{module_name}_unit{i}_unrolled_act.npy"),
                    np.array(unrolled_act))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_act_list.npy"),
                    np.array(list(all_act_list)))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_ord_sorted.npy"),
                    np.array(list(all_ord_sorted)))


if __name__ == "__main__":
    device_str = 'cuda:0'
    valdir = '/media/andrelongon/DATA/imagenet/val'
    sae_name = 'top32_10aux_8exp_sae_weights_100ep'
    layer_name = 'layer3.5.prerelu_out'
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights='IMAGENET1K_V2')
    model = ModelWrapper(model, 8, device, use_gcc=True)
    states = model.state_dict()

    layer_dirs = torch.load(f'/media/andrelongon/DATA/resnet_sae_exp/ckpts/{layer_name}/{sae_name}.pth')['W_dec']
    # layer_dirs = torch.transpose(layer_dirs, 0, 1)
    states['map.weight'] = layer_dirs
    model.load_state_dict(states)
    model = nn.Sequential(model)
    extractor = get_extractor_from_model(model=model, device=device_str, backend='pt')

    top_latents = k_lively_latents(extractor, valdir, sae_name, f'/media/andrelongon/DATA/resnet_sae_exp/tuning_curves/{layer_name}/{sae_name}', k=256)