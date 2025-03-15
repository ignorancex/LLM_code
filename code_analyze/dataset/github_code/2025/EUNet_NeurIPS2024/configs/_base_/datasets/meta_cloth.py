env_cfg = dict(
	clothenv_base=dict(
		root_dir = 'data/meta_cloth', # root path to the garment
		generated_dir = 'data/meta_cloth_generated', # path to store generated data,
		garment_dir = 'data/meta_cloth/mesh', # Path to the original garment path.
		split_meta = 'data/meta_cloth/train_val_test.json',
        mesh_name='mesh_484.obj',
        meta_name='meta_484.json',
		dt=1.0/30,
		fps=30,
		),
	# Frame settings
	omit_frame=1,
    init_frame=2,
	history=0,
	# Noise
	noise_std=0.0,
	step=3, # Autoregressive step
	dynamic=True,
    rollout=False,
	max_frame=30,
	start_frame=0,
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
	train=dict(
		phase='train',
		env_cfg=env_cfg,),
	val=dict(
		phase='test',
		env_cfg=env_cfg,),
	test=dict(
		phase='test', 
		env_cfg=env_cfg,),
)