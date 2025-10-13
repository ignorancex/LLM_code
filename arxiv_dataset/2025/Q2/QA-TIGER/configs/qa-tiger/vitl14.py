config = dict(
	type='qa-tiger',
	seed=713,
	epochs=15,
	num_labels=42,
	log_interval=100,
	output_dir= './runs/music_avqa/qa-tiger_clip_vitl14@336px',
	pretrained_weight="base",
	data=dict(
		root='./data',
		img_size=336,
		batch_size=32,
		eval_batch_size=32,
		num_workers=16,
		frame_sample_rate=1, 

		audios_dir='./raw_audios',
		frames_dir='./raw_frames',
		train_annot='./annots/music_avqa/music_avqa_train.json',
		valid_annot='./annots/music_avqa/music_avqa_val.json',
		test_annot='./annots/music_avqa/music_avqa_test.json',
		test_annots=None,
		ans_quelen='./annots/music_avqa/answer2idx.json',
		
		# precomputed features
		quest_feat=None,
		audio_feat='./feats/vggish',
		video_feat='./feats/frame_ViT-L14@336px',
		patch_feat='./feats/visual_tome14',
		prompt_feat=None,
	),

	hyper_params=dict(
		gpus='0',
		model_type="QA-TIGER_ViTL14@336px",
		model=dict(
			d_model=512,
			video_dim=768,
			patch_dim=1024,
			quest_dim=512,
			audio_dim=128,
			topK=7,
			num_experts=7,
			encoder_type='ViT-L/14@336px',
		),
		optim=dict(
			lr=1e-4,
			encoder_lr=None,
			min_lr=1e-7,
			weight_decay=0,
			betas=(0.95, 0.999)
		),
		sched=dict(
			name='StepLR',
			mode='min',
			gamma=0.1,
			step_size=8,
			factor=0.5,
			patience=5,
			verbose=True,	
			warmup_epochs=2,
		),
	)
)