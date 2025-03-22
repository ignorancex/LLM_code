import ml_collections

from sade.configs.ve.single_matryoshka_config import get_config as get_default_config


def get_config():
    config = get_default_config()

    training = config.training
    training.batch_size = 2
    training.log_freq = 5
    training.pretrained_checkpoint = "/ASD/ahsan_projects/braintypicality/workdir/hres/f2-1e4/checkpoints/checkpoint_130.pth"

    eval = config.eval
    eval.batch_size = 2
    experiment = eval.experiment
    experiment.train = "abcd-val"
    experiment.inlier = "abcd-test"
    experiment.ood = "lesion_load_20-enhanced"

    # msma config
    msma = config.msma
    msma.max_timestep = 0.8
    msma.min_timestep = 0.01

    # flow-model
    flow = config.flow
    flow.base_distribution = "gaussian_mixture"
    flow.num_blocks = 20
    flow.context_embedding_size = 128
    flow.use_global_context = True
    flow.global_embedding_size = 512
    flow.input_norm = False

    flow.patch_batch_size = 2048 # very conservative - could be 16384
    flow.patches_per_train_step = 16384
    flow.training_kimg = 200
    flow.training_fast_mode = False

    # Config for patch sizes
    flow.local_patch_config = ml_collections.ConfigDict()
    flow.local_patch_config.kernel_size = 5
    flow.local_patch_config.padding = 2
    flow.local_patch_config.stride = 1

    # Config for larger receptive fields outputting gobal context
    flow.global_patch_config = ml_collections.ConfigDict()
    flow.global_patch_config.kernel_size = 17
    flow.global_patch_config.padding = 0
    flow.global_patch_config.stride = 8

    return config
