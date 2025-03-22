_base_ = [
    '../_base_/models/selfsup_hoodmgn_learned_pnet.py',
    '../_base_/datasets/cloth3d_dynamic.py',
    '../_base_/schedules/adam_hood.py',
    '../_base_/default_runtime.py'
]

# Custom model
model = dict(
    backbone=dict(
        state_dim=3,),
    decode_head=dict(
        accuracy=[
            dict(type='CollisionAccuracy', acc_name='acc_collision_dynamic'),
            dict(type='L2Accuracy', reduction='mean', acc_name='acc_l2_dynamic'),],),
    test_cfg=dict(
        by_epoch=False,
        step_max_interval=0,
        step=1,),
    selfsup_potential_cfg=dict(
        init_cfg=dict(type='Pretrained', checkpoint="work_dirs/eunet/material/latest.pth"),),
    )

