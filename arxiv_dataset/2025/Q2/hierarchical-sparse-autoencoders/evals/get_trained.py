import json
import jax
import jax.numpy as jnp
import equinox as eqx
import treescope
import sys
import torch
import numpy as np
from collections.abc import MutableMapping

model_code_path = "" # e.g. "../sae"
model_checkpoint_path = "" # Where are the model checkpoints as downloaded using the download_weights code e.g. ../hsae_models

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def flatten(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            if eqx.is_array(value):
                items.append((new_key, np.array(value)))
            else:
                items.append((new_key, value))
    return dict(items)

def get_moe(name, dtype, device, layer, step, top_only):
    sys.path.append(model_code_path)
    with jax.default_device(jax.devices("cpu")[0]):
        from run_moe_eqx_utils import (
            get_model,
            get_lr_schedule,
            get_optimizer,
            regularizer_warmup_fn,
            get_restore_vals,
            restore_state,
        )

    restore_from = f"{model_checkpoint_path}/{name}"
    # i.e. latest step, if you want a specific step enter a multiple of 1500
    restore_step = step

    config = load_config(f'{model_checkpoint_path}/{name}.json')
    subspace_dim = config['subspace_dim']
    num_experts = config['num_experts']
    atoms_per_subspace = config['atoms_per_subspace']
    k = config['k']
    bias = config['bias']
    save_checkpoints = config['save_checkpoints']
    input_dim = 2304
    with jax.default_device(jax.devices("cpu")[0]):
        mngr, restored = get_restore_vals(save_checkpoints, restore_from, restore_step)
        key = jax.random.PRNGKey(0)
        model, hyperparameters = get_model(input_dim, subspace_dim, atoms_per_subspace, num_experts, k, bias, key)
        restored.model["bias"] = restored.model["bias"]#*np.sqrt(input_dim)
    from torch_moe import MixtureOfExperts_v2 as moe_torch
    torch_model = moe_torch(d_in=input_dim,
                            subspace_dim=subspace_dim,
                            atoms_per_subspace=atoms_per_subspace,
                            d_sae=num_experts,
                            model_name="gemma-2-2b",
                            hook_layer=layer,
                            k=k,
                           device=device,
                           dtype=torch.float32,
                            top_only=top_only)
    pt_state_dict = torch_model.state_dict()
    np_state_dict = flatten(restored.model)
    pt_state_dict['top_level_autoencoder.encoder'] = torch.from_numpy(np_state_dict['top_level_autoencoder.encoder']).to(device)
    pt_state_dict['top_level_autoencoder.decoder'] = torch.from_numpy(np_state_dict['top_level_autoencoder.decoder']).to(device)
    pt_state_dict['W_down'] = torch.from_numpy(np_state_dict['W_down']).to(device)
    pt_state_dict['W_up'] = torch.from_numpy(np_state_dict['W_up']).to(device)
    pt_state_dict['encoder_weights'] = torch.from_numpy(np_state_dict['encoder_weights']).to(device)
    pt_state_dict['decoder_weights'] = torch.from_numpy(np_state_dict['decoder_weights']).to(device)
    if bias:
        pt_state_dict['b_dec'] = torch.from_numpy(np_state_dict['bias']).to(device)
    torch_model.load_state_dict(pt_state_dict)
    torch_model = torch_model.to(device)
    return torch_model

def get_trained(type, name, layer, dtype, device, step=None, top_only=False):
    if type == 'MOE':
        return get_moe(name, dtype, device, layer, step, top_only)
