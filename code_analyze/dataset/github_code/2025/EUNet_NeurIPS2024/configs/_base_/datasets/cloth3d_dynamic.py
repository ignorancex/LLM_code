_base_ = [
    './cloth3d.py',
]

dataset_type = 'Cloth3DDynamicDataset'

data = dict(
    test=dict(
        type=dataset_type,
        env_cfg=dict(
            meta_path='data/cloth3d/entry_test.txt',),),
)