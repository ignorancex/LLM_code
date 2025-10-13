import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import LOGGER, seed_everything

if len(LOGGER.handlers) > 1:
    LOGGER.handlers = []


def test(cfg: DictConfig):
    """
    Test a model on a given dataset split.

    Parameters
    ----------
    cfg: DictConfig
        The configuration to use for testing.
    """

    if cfg.get("seed"):
        LOGGER.info(f"Seed set to {cfg.seed}")
        seed_everything(cfg.seed)

    LOGGER.info(f"Instantiating datamoudle <{cfg.test.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.test.datamodule)

    if cfg.test.split == "val":
        val_loader = datamodule.get_dataloader("val")
    else:
        val_loader = datamodule.get_dataloader("test")

    LOGGER.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    LOGGER.info(f"Instantiating model weights from <{cfg.test.weights}>")
    model.load_state_dict(torch.load(cfg.test.weights))

    LOGGER.info(f"Instantiating testers <{cfg.test.tester._target_}>")
    tester = hydra.utils.instantiate(cfg.test.tester)

    LOGGER.info("Starting test!")
    tester.run_test(model, val_loader, torch.device(cfg.test.device))


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(
    cfg: DictConfig,
):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    test(cfg)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    main()
