import ml_collections
from sade.configs.ve.biggan_config import get_config as get_default_config


def get_config():
    config = get_default_config()

    config.training.batch_size = 1
    config.training.log_freq = 10

    # flow-model
    flow = config.flow
    flow.num_blocks = 20
    flow.context_embedding_size = 128
    flow.use_global_context = True
    flow.global_embedding_size = 512


    flow.patch_batch_size = 32
    flow.patches_per_train_step = 2
    flow.training_kimg = 1

    return config
