config = dict(
	type='question_aware',
	seed=713,
	epochs=15,
	num_labels=42,
	log_interval=100,
	output_dir= './runs/music_avqa/tspm_clip_vitl14@336px',
	pretrained_weight="base",
	data=dict(
		root='./data',
		img_size=224,
		batch_size=32,
		eval_batch_size=32,
		num_workers=16,
		frame_sample_rate=1,

		audios_dir='./audios',
		frames_dir='./frames',
		train_annot='./annots/music_avqa/music_avqa_train.json',
		valid_annot='./annots/music_avqa/music_avqa_val.json',
		test_annot='./annots/music_avqa/music_avqa_test.json',
		test_annots=None,
		ans_quelen='./annots/music_avqa/answer2idx.json',
		
		# precomputed features
		audio_feat='./feats/vggish',
		quest_feat='./feats/qst_ViT-L14@336px',
		prompt_feat='./feats/qaPrompt_ViT-L14@336px',
		patch_feat='./feats/visual_tome14',
		video_feat='./feats/frame_ViT-L14@336px',
	),

	hyper_params=dict(
		gpus='0',
		model_type="TSPM_CLIP_ViT-B/32",

		model=dict(
            topK=10,
            avq_cross_attn=False,
            audio_dim=128,
            vis_dim=768,
            patch_dim=1024,
            qst_dim=768,
            hidden_size=512,
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