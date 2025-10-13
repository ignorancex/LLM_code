import json
import sys
import jax

model_code_path = "" # e.g. "../sae"
model_checkpoint_path = "" # Where are the model checkpoints as downloaded using the download_weights code e.g. ../hsae_models
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_moe(name,step=None):
    sys.path.append(model_code_path)
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
    restore_step = None

    config = load_config(f'{model_checkpoint_path}/{name}.json')
    subspace_dim = config['subspace_dim']
    num_experts = config['num_experts']
    atoms_per_subspace = config['atoms_per_subspace']
    k = config['k']
    warmup_steps = config['warmup_steps']
    bias = config['bias']
    save_checkpoints = config['save_checkpoints']
    input_dim = 2304

    restore_step = step
    mngr, restored = get_restore_vals(save_checkpoints, restore_from, restore_step)
    key = jax.random.PRNGKey(0)
    model, hyperparameters = get_model(input_dim, subspace_dim, atoms_per_subspace, num_experts, k, bias, key)

    hyperparameters = restored.hyperparameters
    return restore_state(model,restored.model)
