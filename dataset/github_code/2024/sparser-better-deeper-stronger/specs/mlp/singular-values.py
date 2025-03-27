from main import parse_args_default
from data_handling.utils import combine_with_defaults
from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

base_config = {
    "batch_size": 128,
    "test_batch_size": 128,
    "multiplier": 1,
    "epochs": 100,
    "lr": 0.01,
    "seed": 17,
    "optimizer": "sgd",
    "data": "mnist",
    "model" : 'mlp',
    "l2": 1.0e-4,
    'bench': False,
    'scaled': False,
    'sparse': True,
    'fix': False,
    'sparse_init': 'erk',
    'growth': 'random',
    'death': 'magnitude',
    'redistribution': 'none',
    'death_rate': 0.0,
    'density': 0.05,
    'update_frequency': 100,
    'decay_schedule': 'cosine',
    'use_wandb': False,
    'save_locally': False,
    'tag': name,
    'sigma_w': 1.00,
    'sigma_b': 0.0,
    'q_star': 1,
    "activation": 'tanh',
    'depth': 7,
    'width': 100,
    'no_cuda': False,
}

base_config = combine_with_defaults(
    base_config, defaults=vars(parse_args_default([]))
)


params_grid = [
    { # Sparse baselines
        "seed": [1, 2, 3, 4, 5],
        "density": [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "scaled": [False],
        "fix": [True],
        "sparse": [True],
        "init_type": ['orthogonal'],
        'record_jacobian': [True],
        'sparse_init': ['uniform', 'snip_direct', 'erk', 'grasp_direct', 'synflow_direct', 
                        'uniform_AI', 'snip_direct_AI', 'erk_AI', 'grasp_direct_AI', 'synflow_direct_AI'],
        'activation': ['linear', 'tanh', 'hard_tanh', 'relu'],
    },
    { # Our Mlp
        "seed": [1, 2, 3, 4, 5],
        "density": [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "scaled": [False],
        "fix": [True],
        "sparse": [True],
        "init_type": ['orthogonal'],
        'record_jacobian': [True],
        'sparse_init': ['uniform_EI', 'snip_direct_EI', 'erk_EI', 'grasp_direct_EI', 'synflow_direct_EI'],
        'activation': ['linear', 'tanh', 'hard_tanh', 'relu'],
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="ortho",
    script="python mrun.py",
    python_path="",
    exclude=[
        ".pytest_cache",
        "__pycache__",
        "checkpoints",
        "out",
        "singularity",
        ".vagrant",
        "notebooks",
        "Vagrantfile",
        "results",
        "download"
    ],
    tags=[name],
    base_config=base_config,
    params_grid=params_grid,
)