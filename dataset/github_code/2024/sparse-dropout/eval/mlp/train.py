import lightning as L
import torch
import torch.nn.functional as F
import wandb
from absl import app
from lightning.pytorch.callbacks import EarlyStopping
from ml_collections import config_flags
from torch.optim import Adam

from eval.mlp.dataset import load_data_module
from eval.mlp.model import BasicNet
from eval.utils import CudaTimer, global_metrics

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")


def main(_):
    config = _CONFIG.value

    wandb.login()
    wandb.init(**config.wandb, config=config.to_dict())

    L.seed_everything(config.seed)

    fabric = L.Fabric(**config.fabric)
    fabric.launch()

    # Instantiate objects
    dm = load_data_module(**config.data)
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = BasicNet(dm.train_samlpe[0], **config.model)
    model = fabric.setup_module(model)

    optimizer = Adam(model.parameters(), **config.optimizer)
    optimizer = fabric.setup_optimizers(optimizer)

    # Train
    stopper = EarlyStopping(**config.train.early_stop)

    step = 0
    for epoch in range(config.train.max_epochs):
        for x, label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with CudaTimer() as fwd_timer:
                preds = model(x)
            loss = F.cross_entropy(preds, label)
            with CudaTimer() as bwd_timer:
                fabric.backward(loss)
            optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(preds, -1) == label

            global_metrics.log(
                loss=loss.detach(),
                correct=correct.detach(),
                fwd_timer=fwd_timer,
                bwd_timer=bwd_timer,
            )
            step += 1

            if step % config.train.log_every == 0:
                train_loss, train_acc, fwd_timer, bwd_timer = global_metrics.collect(
                    "loss", "correct", "fwd_timer", "bwd_timer"
                )
                global_metrics.clear()

                fwd_time = [timer.elapsed() for timer in fwd_timer]
                bwd_time = [timer.elapsed() for timer in bwd_timer]
                wandb.log(
                    {
                        "train/loss": torch.tensor(train_loss).mean().item(),
                        "train/acc": torch.concat(train_acc).float().mean().item(),
                        "fwd_time": torch.tensor(fwd_time).mean().item(),
                        "bwd_time": torch.tensor(bwd_time).mean().item(),
                        "step": step,
                        "epoch": epoch,
                    },
                )

            if step % config.train.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for x, label in val_loader:
                        preds = model(x)
                        correct = torch.argmax(preds, -1) == label
                        loss = F.cross_entropy(preds, label)
                        global_metrics.log(val_loss=loss, val_correct=correct)
                model.train()

                val_loss, val_acc = global_metrics.collect("val_loss", "val_correct")
                val_loss = torch.tensor(val_loss).mean()
                val_acc = torch.concat(val_acc).float().mean()
                wandb.log(
                    {
                        "val/loss": val_loss.item(),
                        "val/acc": val_acc.item(),
                        "step": step,
                        "epoch": epoch,
                    },
                )

                should_stop, reason = stopper._evaluate_stopping_criteria(val_acc)
                if should_stop:
                    print(f"Early stopping: {reason}")
                    dm.teardown("fit")
                    exit()

    dm.teardown("fit")


if __name__ == "__main__":
    app.run(main)
