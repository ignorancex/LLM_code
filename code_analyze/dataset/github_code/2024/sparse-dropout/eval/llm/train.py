import lightning as L
import torch
import wandb
from absl import app
from lightning.pytorch.callbacks import EarlyStopping
from ml_collections import config_flags

from eval.llm.dataset import load_data_module
from eval.llm.model import GPT
from eval.utils import CudaTimer, global_metrics

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")


def main(_):
    config = _CONFIG.value
    print(config)

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

    model = GPT(**config.model, vocab_size=dm.vocab_size)
    model = fabric.setup_module(model)

    optimizer = model.configure_optimizers(**config.optimizer)
    optimizer = fabric.setup_optimizers(optimizer)

    # Train
    stopper = EarlyStopping(**config.train.early_stop)

    step = 0
    for epoch in range(config.train.max_epochs):
        for (x,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with CudaTimer() as fwd_timer:
                logits, loss = model(x)
            with CudaTimer() as bwd_timer:
                fabric.backward(loss)
            optimizer.step()

            global_metrics.log(
                loss=loss.detach(),
                fwd_timer=fwd_timer,
                bwd_timer=bwd_timer,
            )
            step += 1

            if step % config.train.log_every == 0:
                train_loss, fwd_timer, bwd_timer = global_metrics.collect(
                    "loss", "fwd_timer", "bwd_timer"
                )
                fwd_time = [timer.elapsed() for timer in fwd_timer]
                bwd_time = [timer.elapsed() for timer in bwd_timer]
                train_loss = torch.tensor(train_loss).mean().item()
                fwd_time = torch.tensor(fwd_time).mean().item()
                bwd_time = torch.tensor(bwd_time).mean().item()
                mfu = model.estimate_mfu(
                    dm.train_batch_size, (fwd_time + bwd_time) / 1000
                )
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "mfu": mfu,
                        "fwd_time": fwd_time,
                        "bwd_time": bwd_time,
                        "step": step,
                        "epoch": epoch,
                    },
                )

            if step % config.train.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for (x,) in val_loader:
                        logits, loss = model(x)
                        global_metrics.log(val_loss=loss)
                model.train()

                (val_loss,) = global_metrics.collect("val_loss")
                val_loss = torch.tensor(val_loss).mean()
                wandb.log(
                    {
                        "val/loss": val_loss.item(),
                        "step": step,
                        "epoch": epoch,
                    },
                )

                should_stop, reason = stopper._evaluate_stopping_criteria(val_loss)
                if should_stop:
                    print(f"Early stopping: {reason}")
                    dm.teardown("fit")
                    exit()

    dm.teardown("fit")


if __name__ == "__main__":
    app.run(main)
