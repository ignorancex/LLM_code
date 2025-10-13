import numpy as np
import torch

from wyckoff_generation.common.args_and_config import get_config
from wyckoff_generation.common.registry import registry
from wyckoff_generation.common.utils import setup_imports


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config):
    """
    Main function for training or testing
    :param config: dictionary with all parameters to be used for training/testing
    :return:
    """
    # Set all random seeds
    seed_all(config["seed"])

    # Figure out what device to use
    if config["cuda"]:
        device = torch.device("cuda")

        # For reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    config["device"] = device

    # setup imports to enable instantiating objects
    setup_imports()
    # initialize runner
    runner = registry.get_runner_class(config["mode"])(config)

    # run
    runner.run()


if __name__ == "__main__":
    conf = get_config()
    main(conf)
