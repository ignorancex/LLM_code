import argparse
import os
import random

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPTokenizer

from models import losses, trainers


class MiddelAgedMan(Dataset):
    def __init__(self, root: str, clip_tokenizer: CLIPTokenizer):
        self.image_folder = datasets.ImageFolder(
            root=root,
            transform=transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
        )

        self.clip_tokenizer = clip_tokenizer
        self.captions = []

        for path, label in self.image_folder.samples:
            class_name = os.path.basename(os.path.dirname(path))
            if class_name == "0_unlabeled":
                caption = "a photo of a middle aged man"
            else:
                caption = "a photo of Brad Pitt"

            self.captions.append(caption)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        caption = self.captions[idx]

        inputs = self.clip_tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        return {
            "image": image,
            "label": label,
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
        }


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="Unsupervised")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    # hyperparameters
    algorithm = args.algorithm
    beta = args.beta if algorithm == "PU" else 0.0
    seed = args.seed
    num_epochs = 1000
    learning_rate = 1e-5 if algorithm == "PU" else 1e-4

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # key
    key = f"main_sd_{algorithm}_{beta}_{seed}"

    # load stable diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    pretrained_pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    vae = pretrained_pipeline.components["vae"]
    text_encoder = pretrained_pipeline.components["text_encoder"]
    tokenizer = pretrained_pipeline.components["tokenizer"]
    unet = pretrained_pipeline.components["unet"]
    noise_scheduler = pretrained_pipeline.components["scheduler"]

    # dataset
    train_loader = DataLoader(
        MiddelAgedMan(root="./datasets/middle_aged_man100", clip_tokenizer=tokenizer),
        batch_size=16,
        shuffle=True,
        num_workers=12,
    )

    # loss
    criterion = losses.load(algorithm=algorithm, beta=beta)

    # output directories
    os.makedirs("outputs/", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/images/", exist_ok=True)

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="outputs/logs",
    )

    # Device
    print("Device:", accelerator.device)

    # train
    print("Fine-Tuning:")
    trainer = trainers.StableDiffusionTrainer(accelerator=accelerator, project_name=key)
    trainer.fit(
        model=unet,
        noise_scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        criterion=criterion,
        data_loader=train_loader,
        num_train_timesteps=1000,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        num_warmup_steps=500,
    )
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=pretrained_pipeline.components["safety_checker"],
        feature_extractor=pretrained_pipeline.components["feature_extractor"],
    ).to(accelerator.device)  # type: ignore

    # save pipeline
    os.makedirs(f"checkpoints/{key}", exist_ok=True)
    pipeline.save_pretrained(f"checkpoints/{key}")

    # save images
    prompts = {"brad_pitt": "a photo of Brad Pitt", "middle_aged_man": "a photo of a middle aged man"}

    with torch.no_grad():
        for k, v in prompts.items():
            overview = pipeline(v, num_images_per_prompt=16).images
            image_grid = make_image_grid(overview, rows=4, cols=4)
            image_grid.save(f"outputs/images/{key}_{k}.png")
