_base_ = [
    './hood.py',
]

dataset_type = 'HoodDataset'

data = dict(
    train=dict(
        type=dataset_type,),
    val=dict(
        type=dataset_type,),
    test=dict(
        rollout=True,
        type=dataset_type,
        env_cfg=dict(
            base=dict(
                external=0,
                noise_scale=0.0,
                separate_arms=True,
                random_betas=False,))),
)