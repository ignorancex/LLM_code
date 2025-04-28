import torch
from pytorch_lightning.cli import LightningCLI

from datasets import EEGDataset
from models import EEGCodec


def cli_main():
    cli = LightningCLI(
        EEGCodec,
        EEGDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cli_main()
