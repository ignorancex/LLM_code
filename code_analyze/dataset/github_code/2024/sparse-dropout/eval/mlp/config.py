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
        project="flash-dropout-mlp",
        notes=placeholder(str),
        mode="online",
    )

    config.model = dict(
        num_layers=2,
        hidden_dim=1024,
        output_dim=10,
        dropout=dict(
            variant="blockwise[cuda]",
            p=0.0,
            block_size=(128, 128),
        ),
    )

    config.optimizer = dict(
        lr=1e-3,
    )

    config.train = dict(
        max_epochs=100,
        eval_every=50,
        log_every=10,
        early_stop=dict(
            monitor="Valiation accuracy",
            patience=5,
            mode="max",
        ),
    )

    config.data = dict(
        name="mnist",
        train_batch_size=1024,
        val_batch_size=1024,
        train_size=16384,
        val_size=4096,
    )

    return config
