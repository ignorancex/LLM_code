import os
import wandb
import argparse
import socket
from .trainer import Trainer
from .strainer import Strainer
from .configs import RunConfig, TrainerConfig, StrainerConfig
from .data_processing import DataProcess
from pathlib import Path
from datetime import datetime
from pydantic_yaml import parse_yaml_file_as

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=Path)
parser.add_argument("--data_path", type=Path)
parser.add_argument("--save_path", type=Path)
parser.add_argument("--name", type=str)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        print("No Internet Connection Available")
        return False


def main() -> None:
    wandb_enabled = (
        os.getenv("WANDB_PROJECT") is not None and os.getenv("WANDB_GROUP") is not None
    )
    wandb_project: str = str(os.getenv("WANDB_PROJECT"))
    wandb_group: str = str(os.getenv("WANDB_GROUP"))
    cfg: RunConfig = parse_yaml_file_as(RunConfig, args.cfg)
    data_path: Path = args.data_path
    name: str = args.name
    fname: str = name + "_" if name else ""
    resume: bool = args.resume
    sp: Path | None = args.save_path
    id = f"{fname}{datetime.strftime(datetime.now(), '%d_%m_%y_%H_%M_%S')}"
    resumeid = None
    if sp is None:
        save_path = PROJECT_ROOT / "outputs" / "runs" / id
    else:
        save_path = sp
        if resume and save_path.exists():
            resumeid = save_path.name  # wandb id of original training run
        elif resume:
            raise ValueError("Save path must contain a training state to resume")

    if wandb_enabled:
        os.makedirs(save_path, exist_ok=True)
        mode = "online" if internet() else "offline"
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            dir=str(save_path),
            config=cfg.model_dump(),
            resume=resume,
            id=resumeid if resume else id,
            name=id,
            mode=mode,
        )
    if isinstance(cfg.trainer_cfg, TrainerConfig):
        pipeline = DataProcess(cfg, data_path)
        trainer = Trainer(
            cfg, data_pipeline=pipeline, save_path=save_path, resume=resume
        )
        trainer.train()
    elif isinstance(cfg.trainer_cfg, StrainerConfig):
        pipeline = DataProcess(cfg, data_path, True)
        strainer = Strainer(
            cfg,
            data_pipeline=pipeline,
            save_path=save_path,
            resume=resume,
        )
        strainer.train()


if __name__ == "__main__":
    main()
