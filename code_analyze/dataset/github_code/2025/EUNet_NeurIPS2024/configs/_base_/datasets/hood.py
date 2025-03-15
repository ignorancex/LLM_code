env_cfg = dict(
    wholeseq=False,
	base=dict(
        smpl_model='smpl/SMPL_FEMALE.pkl',
		smpl_segm='data/cloth3d/smpl/segm_per_v_overlap.pkl',
        garment_dict_file='garments_dict.pkl',
        split_path='datasplits/train.csv',

        aux_data='data/hood_data/aux_data',
        data_root='data/hood_data',
        orig_data_root='vto_dataset/smpl_parameters',

        # Settings from HOOD
        noise_scale=3e-3,
        lookup_steps=5,
        pinned_verts=True,
        wholeseq=False,
        random_betas=True,
        use_betas_for_restpos=False,
        betas_scale=3.,
        restpos_scale_min=1.,
        restpos_scale_max=1.,
        n_coarse_levels=0,
        separate_arms=False,
        zero_betas=False,
        button_edges=False,
        external=9.8,
		dt=1.0/30,
		fps=30,
        mass_scalar=1.0,
		),
	# Frame settings
	omit_frame=1,
    init_frame=1,
	history=1,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
	train=dict(
		phase='train',
		env_cfg=env_cfg,),
	val=dict(
		phase='val',
		env_cfg=env_cfg,),
	test=dict(
		phase='test', 
		env_cfg=env_cfg,),
)