from main import parse_args_default
from data_handling.utils import combine_with_defaults
from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

base_config = {
    "batch_size": 128,
    "test_batch_size": 128,
    "multiplier": 1,
    "epochs": 100,
    "lr": 0.0001,
    "seed": 17,
    "optimizer": "sgd",
    "data": "mnist",
    "model": 'mlp',
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
    'density': 0.1,
    'update_frequency': 100,
    'decay_schedule': 'cosine',
    'use_wandb': False,
    'save_locally': False,
    'tag': name,
    'sigma_w': 1.0247,
    'sigma_b': 0.00448,
    'q_star': 1,
    "activation": 'tanh',
    'depth': 1000,
    'width': 128,
    'no_cuda': False,
    'AI_iters': 1000,
}

base_config = combine_with_defaults(
    base_config, defaults=vars(parse_args_default([]))
)

params_grid = [
    {  # Dense Conv baseline
        "seed": [1, 2, 3, 4, 5],
        "density": [1.0],
        "scaled": [False],
        "fix": [True],
        "sparse": [False],
        "init_type": ['orthogonal'],
        "lr": [0.0001],
    },
    {  # Sparse large conv
        "seed": [1, 2, 3, 4, 5],
        "density": [0.125],
        "scaled": [False],
        "fix": [True],
        "sparse": [True],
        "init_type": ['orthogonal'],
        'sparse_init': ['synflow_direct', 'uniform', 'erk', 'snip_direct', 'grasp_direct', 'synflow_direct_AI',
                        'uniform_AI', 'erk_AI', 'snip_direct_AI', 'grasp_direct_AI', 'synflow_direct_EI', 'uniform_EI',
                        'erk_EI', 'snip_direct_EI', 'grasp_direct_EI'],
        "lr": [0.0001],
    },
    {  # Sparse SAO
        "seed": [1, 2, 3, 4, 5],
        "degree": [16],
        "depth": [1000],
        "density": [0.125],
        "scaled": [False],
        "fix": [True],
        "sparse": [True],
        "init_type": ['default'],  # we use default for SAO, will be overriden by DeltaSAO init
        'sparse_init': ['SAO'],
        "lr": [0.0001],
        "channel_width": [128]
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="ortho",
    script="./scripts/mrun_scipy.sh",
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