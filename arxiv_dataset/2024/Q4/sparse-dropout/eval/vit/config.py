from pathlib import Path

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config():
    config = ConfigDict()

    config.seed = 0
    config.fabric = dict(
        accelerator="auto",
        precision="16-mixed",
    )
    config.wandb = dict(
        project="flash-dropout-vit",
        notes=placeholder(str),
        mode="online",
    )

    config.model = dict(
        patch_size=(2, 2),
        block_size=(8, 8),
        n_embed=1024,
        n_head=8,
        n_layers=2,
        dropout=dict(
            p=0.0,
            variant="vanilla",
            block_size=(128, 128),
        ),
    )

    config.optimizer = dict(
        lr=1e-4,
    )

    config.train = dict(
        max_epochs=50,
        eval_every=200,
        log_every=50,
        early_stop=dict(
            monitor="Valiation accuracy",
            patience=5,
            mode="max",
        ),
    )

    config.data = dict(
        name="mnist",
        train_batch_size=64,
        val_batch_size=64,
        train_size=8192,
        val_size=4096,
    )

    return config
