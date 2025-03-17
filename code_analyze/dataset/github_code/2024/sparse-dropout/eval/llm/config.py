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
        project="flash-dropout-llm",
        notes=placeholder(str),
        mode="online",
    )

    config.model = dict(
        context_length=128,
        n_layer=4,
        n_head=8,
        n_embed=1024,
        dropout=dict(
            p=0.0,
            variant="vanilla",
            block_size=(128, 128),
        ),
    )

    config.optimizer = dict(
        lr=3e-4,
        weight_decay=1e-1,
    )

    config.train = dict(
        max_epochs=100,
        eval_every=200,
        log_every=50,
        early_stop=dict(
            monitor="Valiation loss",
            patience=5,
            mode="min",
        ),
    )

    config.data = dict(
        name="shakespeare",
        length=config.model.get_ref("context_length"),
        cache_dir=Path("data", "shakespeare"),
        train_batch_size=32,
        val_batch_size=64,
        train_size=4096,
        val_size=1024,
    )

    return config
