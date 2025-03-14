model = dict(
    type='SelfsupDynamic',
    processor_cfg=[
        dict(type='DynamicDGLProcessor'),],
    dynamic_cfg=dict(
        type='DynamicContactProcessor',
        radius=0.03,
        group_cfg=dict(
            max_radius=None,
            min_radius=0.0,
            sample_num=1,
            use_xyz=True,
            normalize_xyz=False,
            return_grouped_xyz=False,
            return_grouped_idx=True,
            return_unique_cnt=False,),
        ),
    collision_pushup=2e-3,
    backbone=dict(
        type='MeshGraphNet',
        state_dim=6,
        gravity_dim=6,
        num_frames=2,
        position_dim=3,
        embed_dims=128,
        num_encoder_layers=15,
        num_fcs=3,
        dropout=0.0,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='LN'),
        dt=1.0/30,
        ),
    decode_head=dict(
        type='AccDecoder',
        in_channels=128,
        out_channels=3,
        dt=1.0/30,
        loss_decode=[
            dict(type='ExternalLoss', reduction='sum', loss_weight=2e-2, loss_name='loss_external_dynamic'),
            dict(type='InertiaLoss', reduction='sum', loss_weight=2e-2, loss_name='loss_inertia_dynamic'),
            dict(type='GravityLoss', reduction='sum', loss_weight=2e-2, loss_name='loss_gravity_dynamic'),
            dict(type='PotentialPriorLoss', reduction='sum', loss_weight=2e-2, loss_name='loss_potential_prior_dynamic'),
            dict(
                type='ContactLoss',
                thresh=2e-3,
                radius=None,
                min_radius=0.0,
                sample_num=1,
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False,
                reduction='sum', loss_weight=1.0, loss_name='loss_contact_dynamic',
                weight_start=5e5, weight_max=5e7, start_rampup_iteration=50000, n_rampup_iterations=100000),
            ],
        ),
    selfsup_potential_cfg=dict(
        type='EUNetSimulator',
        train_cfg=dict(warmup=100),
        backbone=dict(
            type='EnergyPotential',
            pre_norm=True,
            dt=1/30,
            attr_dim=5,
            state_dim=3,
            position_dim=3,
            num_force_layers=4,
            embed_dims=128,
            dropout=0.0,
            eps=1e-7,
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='LN'),
            num_fcs=3,
            dissipate_sigma=0.5,
            ),
        decode_head=dict(
            type='PotentialDecoder',
            out_channels=1, # 1 for scalar
            in_channels=128,
            position_dim=3,
            dt=1/30,
            init_cfg=None,
            eps=1e-7,
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='LN'),
            norm_acc_steps=10000,
            dissipate_sigma=0.5,
            loss_decode=[
                dict(type='MSELoss',
                    reduction='sum',
                    loss_weight=1.0,
                    loss_name='loss_elenergy'),
                dict(type='CmpLoss',
                    reduction='sum',
                    loss_weight=1e6,
                    min_diff=0.0,
                    scalar=1.0,
                    loss_name='loss_elcmp'),
                ],
            accuracy=[
                dict(type='MSEAccuracy', reduction='sum', acc_name='acc_elenergy'),
                dict(type='CompareAccuracy', reduction='mean', acc_name='acc_elcmp'),
                ],
            ),
    )
)