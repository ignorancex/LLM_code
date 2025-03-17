import os

from sade.configs.default_brain_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.likelihood_weighting = False
    training.reduce_mean = True
    training.batch_size = 4
    training.n_iters = 20
    training.log_freq = 2
    training.eval_freq = 5
    training.snapshot_freq_for_preemption = 10
    training.sampling_freq = 10
    training.load_pretrain = False
    training.pretrain_dir = "workdir/test/pretrain/"

    data = config.data
    data.image_size = (48, 64, 40)
    data.spacing_pix_dim = 4.0
    data.num_channels = 2
    data.cache_rate = 0.0

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    dir_path = os.path.join(cur_dir, "..", "..", "..", "tests", "dummy_data")
    data.dir_path = os.path.abspath(dir_path)
    data.splits_dir = data.dir_path

    evaluate = config.eval
    evaluate.sample_size = 8
    evaluate.batch_size = 8

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.warmup = 1000
    optim.scheduler = "skip"

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.probability_flow = False
    sampling.snr = 0.17
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    # model
    model = config.model
    model.num_scales = 10
    model.name = "ncsnpp3d"
    model.resblock_type = "biggan"
    model.act = "memswish"
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nf = 8
    model.norm_num_groups = 2
    model.blocks_down = (1, 2, 1)
    model.blocks_up = (1, 1)
    model.time_embedding_sz = 32
    model.init_scale = 0.0
    model.num_scales = 10
    model.conv_size = 3
    model.self_attention = False
    model.dropout = 0.0
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = False

    return config
