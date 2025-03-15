_base_ = [
    './meta_cloth.py',
]

dataset_type = 'MetaClothDynamicDataset'

data = dict(
    train=dict(
        type=dataset_type,
        env_cfg=dict(
            dynamic=True,
            rollout=False,
            meta_path='data/meta_cloth/entry_train_meta.txt', # path to the "seq\tframe_num"
            ),),
    val=dict(
        type=dataset_type,
        env_cfg=dict(
            dynamic=True,
            rollout=False,
            meta_path='data/meta_cloth/entry_test_meta.txt', # path to the "seq\tframe_num"
            ),),
    test=dict(
        type=dataset_type,
        env_cfg=dict(
            dynamic=True,
            rollout=True,
            meta_path='data/meta_cloth/entry_test_meta.txt', # path to the "seq\tframe_num"
            ),),
)