"""Training NCSN++ on Brains with VE SDE.
   Keeping it consistent with CelebaHQ config from Song
"""
from sade.configs.default_brain_configs import get_default_configs
from sade.configs.ve.biggan_config import get_config as inner_config


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.reduce_mean = True
    training.batch_size = 4
    training.sampling_freq = 10000
    training.n_iters = 1_500_001
    training.pretrain_dir = (
        "/ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/checkpoints-meta/"
    )

    data = config.data
    data.image_size = (176, 208, 160)

    data.spacing_pix_dim = 1.0
    data.num_channels = 2
    data.cache_rate = 0.0
    data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"
    data.splits_dir = "/ASD/ahsan_projects/Developer/braintypicality-scripts/split-keys/"
    data.ood_ds = "lesion_load_20"

    evaluate = config.eval
    evaluate.sample_size = 2
    evaluate.batch_size = 4

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 1e-4
    optim.warmup = 10_000
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
    model.name = "single-matryoshka"
    model.resblock_type = "biggan"
    model.act = "memswish"
    model.ema_rate = 0.9999
    model.nf = 16
    model.time_embedding_sz = 64
    model.num_scales = 2000
    model.embedding_type = "fourier"
    model.fourier_scale = 2.0
    model.learnable_embedding = True
    model.trainable_inner_model = True

    model.sigma_max = 1508
    model.sigma_min = 0.09

    config.inner_model = inner_config()

    return config
