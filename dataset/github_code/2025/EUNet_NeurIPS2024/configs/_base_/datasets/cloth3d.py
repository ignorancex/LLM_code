env_cfg = dict(
    clothenv_base=dict(
        root_dir = 'data/cloth3d/train',
        smpl_dir = 'data/cloth3d/smpl',
        generated_dir = 'data/cloth3d/generated_train',
        smpl_segm='data/cloth3d/smpl/segm_per_v_overlap.pkl',
        dt=1.0/30, # Only for computing, the real one is controled by fps
        fps=30,),
    # Frame settings
    max_frame=80,
    start_frame=10,
	omit_frame=1,
    init_frame=2, # historical reason here
	history=1,
	# Model behavior related
	step=1,
	dynamic=True,
    rollout=True,
    collision=True,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    test=dict(
        phase='test', 
        env_cfg=env_cfg,),
)