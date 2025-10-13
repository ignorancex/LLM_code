import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type,
    ):
        
        file_name = [
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/coffee.png",
            "assets/clock.jpg",
            "assets/book.jpg",
            "assets/monalisa.jpg",
            "assets/oranges.jpg",
            "assets/penguin.jpg",
            "assets/vase.jpg",
            "assets/room_corner.jpg",
        ]
        
        test_instruction = [
            "Make the image look like it's from an ancient Egyptian mural.",
            'get rid of the coffee bean.',
            'remove the cup.',
            "Change it to look like it's in the style of an impasto painting.",
            "Make this photo look like a comic book",
            "Give this the look of a traditional Japanese woodblock print.",
            'delete the woman',
            "Change the image into a watercolor painting.",
            "Make it black and white.",
            "Make it pop art.",
            'the sofa is leather, and the wall is black',
        ]
        
        pl_module.flux_fill_pipe.transformer.eval()
        for i, name in enumerate(file_name):
            test_image = Image.open(name)
            combined_image = Image.new('RGB', (test_image.size[0] * 2, test_image.size[1]))
            combined_image.paste(test_image, (0, 0))
            combined_image.paste(test_image, (test_image.size[0], 0))
            
            mask = Image.new('L', combined_image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([test_image.size[0], 0, test_image.size[0] * 2, test_image.size[1]], fill=255)
            if condition_type == 'edit_n':
                prompt_ = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left. \n " + test_instruction[i]
            else:
                prompt_ = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but " + test_instruction[i]
                
            image = pl_module.flux_fill_pipe(
                prompt=prompt_,
                image=combined_image,
                height=512,
                width=1024,
                mask_image=mask,
                guidance_scale=50,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(666)
            ).images[0]
            image.save(os.path.join(save_path, f'flux-fill-test-{self.total_steps}-{i}-{condition_type}.jpg'))
        
        pl_module.flux_fill_pipe.transformer.train()
        