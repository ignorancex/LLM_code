_base_ = [
    '../_base_/models/selfsup_hoodmgn_learned_pnet.py',
    '../_base_/datasets/hood_dynamic.py',
    '../_base_/schedules/adam_hood.py',
    '../_base_/default_runtime.py'
]


step = 5
step_max_interval = 5000
checkpoint_interval = 2000

model = dict(
    train_cfg=dict(
        by_epoch=False,
        step_max_interval=step_max_interval,
        step=step,),
    test_cfg=dict(
        by_epoch=False,
        step_max_interval=step_max_interval,
        step=step,),
    selfsup_potential_cfg=dict(
        init_cfg=dict(type='Pretrained', checkpoint="work_dirs/eunet/material/latest.pth"),),
    )

# Custom dataset
data = dict(
    train=dict(
        env_cfg=dict(
            base=dict(attr_option='random'),
            noise_std=0.0,
            omit_frame=step,
            step=step,),),
    val=dict(
        val_seq=2,
        env_cfg=dict(
            base=dict(attr_option='random'),
            omit_frame=step,
            step=step,),),
    test=dict(
        env_cfg=dict(
            # Can overwrite the option for inference. This is not to compare with ground truth
            base=dict(attr_option='0'),
            rollout=True,
            collision=True,
            step=1),),)


evaluation = dict(by_epoch=False, interval=int(150000))
checkpoint_config = dict(by_epoch=False, interval=int(checkpoint_interval), max_keep_ckpts=100)
