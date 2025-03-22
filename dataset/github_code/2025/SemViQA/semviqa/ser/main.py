import logging
import os
import pandas as pd
import numpy as np
import torch
import gc
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from transformers import default_data_collator
from torch.utils.data import DataLoader
import argparse
import time
from safetensors.torch import load_model
from transformers import AutoTokenizer

from .qatc_model import QATCConfig, QATCForQuestionAnswering
from .data_utils import load_data 

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {active_params}")

def load_models(args):
    config = QATCConfig(
        model_name=args.model_name,
        freeze_text_encoder=args.freeze_text_encoder,
        alpha=args.alpha,
        beta=args.beta
    )
    
    model = QATCForQuestionAnswering(config)
    
    if args.is_load_weight:
        print("loading train weight")
        if "safetensors" in args.weight_model:
            load_model(model, f"{args.weight_model}")
        else:
            model.load_state_dict(torch.load(args.weight_model), strict=False)
    
    count_parameters(model)
    return model, config

def setting_optimizer(config, model):
    optimizer_cls = torch.optim.AdamW
    return optimizer_cls(model.parameters(), lr=config.learning_rate)

def main(args):
    logger = get_logger(__name__, log_level="INFO")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    model, config = load_models(args)
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    optimizer = setting_optimizer(args, model)
    train_dataset, test_dataset = load_data(args, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.train_batch_size
    )
    
    eval_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size
    )

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=args.max_lr, cycle_momentum=False)

    if args.is_load_weight: 
        print("loading scheduler weight")
        lr_scheduler.load_state_dict(torch.load(args.weight_scheduler))

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    progress_bar = tqdm(desc="Steps", disable=not accelerator.is_local_main_process)
 
    best_acc = 0.0

    print("Starting training...")
    global_step = 1
    cnt = 0
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(accelerator.device)
            batch['Tagging'] = batch['Tagging'].to(torch.float32)
            output = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                start_positions=batch['start_positions'], 
                end_positions=batch['end_positions'],
                tagging_labels=batch['Tagging']
            )

            loss = output.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            logs = {"step": f"{step}/{len(train_dataloader)}", "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(logs)
            global_step += 1
            
            if global_step % args.max_iter == 0:
                train_loss = round(train_loss / args.max_iter, 4)

                print({
                        'global_step': global_step,
                        'Train loss': train_loss, 
                        "epoch": epoch,
                    }) 
                train_loss = 0.0

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss}")

        model.eval()
        eval_loss = 0.0
        predictions, true_positions = [], []
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                for key in batch:
                    batch[key] = batch[key].to(accelerator.device)
                batch['Tagging'] = batch['Tagging'].to(torch.float32)
                output = model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions'],
                    tagging_labels=batch['Tagging']
                )

                loss = output.loss
                eval_loss += loss.item()

                start_preds = torch.argmax(output.start_logits, axis=1).cpu().detach().tolist()
                end_preds = torch.argmax(output.end_logits, axis=1).cpu().detach().tolist()
                start_true = batch['start_positions'].flatten().cpu().tolist()
                end_true = batch['end_positions'].flatten().cpu().tolist()

                predictions.extend(list(zip(start_preds, end_preds)))
                true_positions.extend(list(zip(start_true, end_true)))
            
            progress_bar.update(1)

        eval_loss /= len(eval_dataloader)
        accuracy = np.mean([p == t for p, t in zip(predictions, true_positions)])

        print(f"Epoch {epoch+1} - Eval Loss: {eval_loss} - Accuracy: {accuracy}")

        if accuracy > best_acc:
            save_path = os.path.join(args.output_dir, f"best {args.name}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # accelerator.save_state(save_path)
            model.save_pretrained(save_path)
            torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
            torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.bin"))
            tokenizer.save_pretrained(save_path)
            config.save_pretrained(save_path)
            best_acc = accuracy
            print("Save model best acc at epoch", epoch)
            cnt = 0
        else:
            cnt += 1
        if cnt == args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print("Training completed in:", time.time() - start_time, "seconds.")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VNFASHIONDIFF")
    parser.add_argument("--name", type=str, default="QATC", help="Name of the model")
    parser.add_argument("--model_name", type=str, default="/kaggle/input/model-base", help="Path to the base model") 
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/", help="Output directory")
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--train_batch_size", type=int, default=12, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_epoch_eval", type=int, default=1, help="Number of epochs to evaluate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_8bit_adam", type=int, default=None, help="Use 8-bit Adam optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--max_lr", type=float, default=0.00003, help="Maximum learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting destination (e.g., 'wandb')")
    
    # Data related arguments
    parser.add_argument("--train_data", type=str, default='train.csv', help="Path to training data")
    parser.add_argument("--eval_data", type=str, default='test.csv', help="Path to evaluation data")
    
    # Others
    parser.add_argument("--freeze_text_encoder", type=int, default=0, help="Whether to freeze the text encoder")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha")
    parser.add_argument("--is_load_weight", type=int, default=0, help="Load weights from pre-trained model")
    parser.add_argument("--weight_model", type=str, default="/kaggle/input/weight-QACT/pytorch_model.bin", help="Path to model weights")
    parser.add_argument("--weight_optimizer", type=str, default="/kaggle/input/weight-QACT/optimizer.bin", help="Path to optimizer weights")
    parser.add_argument("--weight_scheduler", type=str, default="/kaggle/input/weight-QACT/scheduler.bin", help="Path to scheduler weights")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--is_train", type=int, default=1, help="Set to True if training")
    parser.add_argument("--is_eval", type=int, default=1, help="Set to Eval")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    main(args)
