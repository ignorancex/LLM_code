from typing import Optional

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import LOGGER

if len(LOGGER.handlers) > 1:
    LOGGER.handlers = []


def export(cfg: DictConfig):
    """
    Export TSM_RGB model to ONNX format.
    """
    LOGGER.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    LOGGER.info(f"Instantiating model weights from <{cfg.test.weights}>")
    model.load_state_dict(torch.load(cfg.test.weights))
    # Input to the model
    batch_size = 16
    time_len = 16
    height = 244
    width = 244
    x = (
        torch.randn(batch_size, 3, height, width, requires_grad=True),
        torch.tensor([time_len], requires_grad=False),
    )
    _ = model(x[0], x[1])

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "path_to_save",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input", "input_2"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "height",
                3: "width",
            },  # variable length axes
            "input_2": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time_len", 3: "height", 4: "width"},
        },
    )


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name=None,
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    export(cfg)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    main()
