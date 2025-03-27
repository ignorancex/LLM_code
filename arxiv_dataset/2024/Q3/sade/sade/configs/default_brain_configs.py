import math

import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 3
    training.n_iters = 250001
    training.snapshot_freq = 10000
    training.log_freq = 100
    training.eval_freq = 500

    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.sampling_freq = 10000
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    # Pretrain options
    training.load_pretrain = False
    training.pretrained_checkpoint = "/path/to/weights/"
    training.grad_accum_factor = 1

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 10
    evaluate.end_ckpt = 10
    evaluate.batch_size = 32
    evaluate.enable_sampling = True
    evaluate.num_samples = 8
    evaluate.enable_loss = False
    evaluate.ood_eval = False
    evaluate.sample_size = 32
    evaluate.checkpoint_num = -1

    experiment = config.eval.experiment = ml_collections.ConfigDict()
    experiment.id = "default"
    experiment.train = "abcd-val"  # The dataset used for training MSMA
    experiment.inlier = "abcd-test"
    experiment.ood = "tumor"
    experiment.flow_checkpoint_path = "/path/to/weights/"

    # msma
    config.msma = msma = ml_collections.ConfigDict()
    msma.max_timestep = 1.0
    msma.min_timestep = 0.01  # Ignore first x% of sigmas
    msma.n_timesteps = 20  # Number of discrete timesteps to evaluate
    msma.schedule = "geometric"  # Timestep schedule that dictates which sigma to sample
    msma.checkpoint = -1  # ckpt number for score norms, defaults to latest (-1)
    msma.skip_inliers = False  # skip computing score norms for inliers
    msma.expectation_iters = -1
    msma.denoise = False
    msma.l2_normed = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "ABCD"
    # data.ood_ds = "lesion"  # "IBIS"
    data.image_size = (176, 208, 160)  # For generating images
    data.spacing_pix_dim = 1.0
    data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"
    data.splits_dir = "/ASD/ahsan_projects/Developer/braintypicality-scripts/split-keys/"
    data.cache_rate = 0.0
    data.num_channels = 2

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 545.0  # For medres
    model.sigma_min = 0.06
    model.num_scales = 1000
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.blocks_down = (1, 2, 2, 4)
    model.blocks_up = (1, 1, 1)
    model.resblock_pp = False
    model.dilation = 1
    model.jit = False

    # flow-model
    config.flow = flow = ml_collections.ConfigDict()
    flow.num_blocks = 4
    flow.patch_batch_size = 8
    flow.context_embedding_size = 128
    flow.use_global_context = True
    flow.global_embedding_size = 512
    flow.training_fast_mode = False

    # Config for patch sizes
    flow.local_patch_config = ml_collections.ConfigDict()
    flow.local_patch_config.kernel_size = 3
    flow.local_patch_config.padding = 1
    flow.local_patch_config.stride = 1

    # Config for larger receptive fields outputting gobal context
    flow.global_patch_config = ml_collections.ConfigDict()
    flow.global_patch_config.kernel_size = 11
    flow.global_patch_config.padding = 2
    flow.global_patch_config.stride = 4

    # Flow training configs
    flow.lr = 3e-4
    flow.training_kimg = 100
    flow.ema_halflife_kimg = 50
    flow.ema_rampup_ratio = 0.01
    flow.patches_per_train_step = 256
    flow.log_tensorboard = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.scheduler = "skip"
    optim.lr = 3e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    # finetuning
    config.finetuning = finetuning = ml_collections.ConfigDict()
    finetuning.n_iters = 5000
    finetuning.n_fast_steps = 20
    finetuning.outer_step_size = 0.1
    finetuning.fp16 = False
    finoptim = finetuning.optim = ml_collections.ConfigDict()
    finoptim.weight_decay = 0.0
    finoptim.optimizer = "Adam"
    finoptim.scheduler = "skip"
    finoptim.lr = 3e-4
    finoptim.beta1 = 0.0
    finoptim.eps = 1e-8
    finoptim.warmup = 1
    finoptim.grad_clip = 0

    config.seed = 42
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.fp16 = False

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(
        optim_weight_decay={
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-1),
        },
        optim_optimizer={"values": ["Adam", "Adamax", "AdamW"]},
        # optim_lr={
        #     "distribution": "log_uniform",
        #     "min": math.log(1e-5),
        #     "max": math.log(1e-2),
        # },
        model_time_embedding_sz={"values": [128, 256]},
        model_attention_heads={"values": [1, 0]},
        # model_embedding_type={"values": ["fourier", "positional"]},
        optim_warmup={"values": [5000]},
        optim_scheduler={"values": ["skip"]},
        training_n_iters={"value": 50001},
        training_log_freq={"value": 50},
        training_eval_freq={"value": 100},
        training_snapshot_freq={"value": 100000},
        training_snapshot_freq_for_preemption={"value": 100000},
    )

    sweep.parameters = param_dict
    sweep.method = "bayes"
    sweep.metric = dict(name="val_loss")

    return config
