import torch


class DENOISER_CONST:
    WEIGHT_DECAY_KEY = "weight_decay"
    LR_KEY = "lr"
    OPTIM_KEY = "optimizer"
    RANDOM_SEED_KEY = "random_seed"
    DEVICE_KEY = "device"
    NOISE_TYPE_KEY = "noise_type"
    NOISE_PARAMS_KEY = "noise_params"
    EPOCHS_KEY = "epochs"

    SGD_OPTIM = "SGD"
    ADAM_OPTIM = "ADAM"
    ADAMW_OPTIM = "ADAMW"
    OPTIMS = [SGD_OPTIM, ADAM_OPTIM, ADAMW_OPTIM]

    DEFAULT_RANDOM_SEED = 2024
    DEFAULT_DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    DEFAULT_EPOCHS = 10
    DEFAULT_LR = 0.001
    DEFAULT_WEIGHT_DECAY = 0.0005
    DEFAULT_OPTIM = ADAM_OPTIM

    GAUSSIAN_NOISE = "GAUSSIAN"
    SALT_AND_PEPPER_NOISE = "SALT_AND_PEPPER"
    NOISES = [GAUSSIAN_NOISE, SALT_AND_PEPPER_NOISE]

    DEFAULT_GAUSSIAN_NOISE_PARAMS = (0.001, 0.0)
    DEFAULT_SALT_AND_PEPPER_NOISE_PARAMS = (0.1, 0.9)
    DEFAULT_NOISE_TYPE = GAUSSIAN_NOISE
