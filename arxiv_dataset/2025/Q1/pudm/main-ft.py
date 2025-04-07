import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DDIMPipeline
from diffusers.utils import make_image_grid

import dataloaders
import settings
from models import losses, trainers

# constants for generation
NUM_INFERENCE_STEPS = 50

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="CIFAR10_vehicles")
    parser.add_argument("--algorithm", type=str, default="Unsupervised")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    # hyperparameters
    config_name = args.config_name
    algorithm = args.algorithm
    beta = args.beta
    seed = args.seed

    if algorithm != "PU":
        beta = 0.0

    # config
    config = settings.load(name=config_name)

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # key
    key = f"main_ft_{config_name}_{algorithm}_{beta}_{seed}"

    # dataset
    train_loader, test_loader = dataloaders.load(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        normal_class=config.normal_class,
        n_unlabeled_normal=config.n_unlabeled_normal,
        n_unlabeled_sensitive=config.n_unlabeled_sensitive,
        n_labeled_sensitive=config.n_labeled_sensitive,
        n_test=config.n_test,
    )

    # load pre-trained model
    if config.dataset_name == "CIFAR10":
        model_id = "google/ddpm-cifar10-32"
    else:
        model_id = "google/ddpm-celebahq-256"

    pretrained_pipeline = DDIMPipeline.from_pretrained(model_id)
    model = pretrained_pipeline.components["unet"]
    noise_scheduler = pretrained_pipeline.components["scheduler"]

    # loss
    criterion = losses.load(algorithm=algorithm, beta=beta)

    # output directories
    os.makedirs("outputs/", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/images/", exist_ok=True)

    # Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="outputs/logs",
    )

    # Device
    print("Device:", accelerator.device)

    # train
    print("Fine-Tuning:")
    trainer = trainers.Trainer(accelerator=accelerator, project_name=key)
    trainer.fit(
        model=model,
        noise_scheduler=noise_scheduler,
        criterion=criterion,
        data_loader=train_loader,
        num_train_timesteps=config.num_train_timesteps,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        num_warmup_steps=config.num_warmup_steps,
    )
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler).to(accelerator.device)  # type: ignore

    # save pipeline
    os.makedirs(f"checkpoints/{key}", exist_ok=True)
    pipeline.save_pretrained(f"checkpoints/{key}")

    # save images
    print("save images:")
    with torch.no_grad():
        overview = pipeline(batch_size=64, num_inference_steps=NUM_INFERENCE_STEPS).images
        image_grid = make_image_grid(overview, rows=8, cols=8)
        image_grid.save(f"outputs/images/{key}.png")
