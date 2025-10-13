import os
import uuid
import json
import numpy as np
import torch
import wandb
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ["WANDB_MODE"] = "offline"
from tqdm import tqdm
from datetime import datetime as dt
from accelerate import Accelerator

from src.utils.model_io import load_checkpoint, save_checkpoint, save_weights
from src.utils.utils import RunningAverage, RunningAverageDict, compute_errors

from src.dataloader.nyu import NYUV2
from src.dataloader.hypersim import HYPERSIM
from src.dataloader.zju import ZJU
from src.loss import SILogLoss, CharbonnierLoss

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

logging = True

from src.models.depthor import Depthor
# from src.models.depthor_s import Depthor

def main_worker(args):
    # set up accelerator and model
    accelerator = Accelerator()
    model = Depthor(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    model.set_extra_param()

    # set up wandb
    if logging and accelerator.is_main_process:
        project = 'depthor' + f"-{args.dataset}"
        run_id = f"{dt.now().strftime('%d-%h_%H')}-nodebs{args.bs}-tep{args.epochs}-{uuid.uuid4()}"
        name = f"{args.name}_{run_id}"
        args.run_id = run_id
        tags = args.tags.split(',') if args.tags != '' else None
        wandb.init(project=project, name=name, config=args, dir='./', tags=tags, notes=args.notes)
        with open(f'{wandb.run.dir}/run_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # resume training
    if args.resume != '':
        model, optimizer_state_dict, epoch = load_checkpoint(args.resume, model)
        args.epoch = epoch + 1
        args.last_epoch = epoch
    else:
        args.epoch = 0
        args.last_epoch = -1
        optimizer_state_dict = None

    # set up optimizer
    params = [{"params": model.get_lr_params(), "lr": args.lr}]
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # set up dataloader
    if args.dataset == 'nyu':
        train_loader = NYUV2(args, 'train').data
    elif args.dataset == 'hypersim':
        train_loader = HYPERSIM(args, 'train').data
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    if args.dataset_eval == 'nyu':
        test_loader = NYUV2(args, 'online_eval').data
    elif args.dataset_eval == 'hypersim':
        test_loader = HYPERSIM(args, 'online_eval').data
    elif args.dataset_eval == 'zju':
        test_loader = ZJU(args, 'online_eval').data
    else:
        raise NotImplementedError(f"Dataset {args.dataset_eval} not implemented.")

    # set up scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                              steps_per_epoch=len(train_loader),
                                              cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                                              last_epoch=args.last_epoch,
                                              div_factor=args.div_factor, final_div_factor=args.final_div_factor)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    test_loader = accelerator.prepare(test_loader)
    train(model, optimizer, train_loader, scheduler, test_loader, accelerator, args)


def train(model, optimizer, train_loader, scheduler, test_loader, accelerator, args, ):
    log_loss = SILogLoss()
    mse_loss = CharbonnierLoss()

    model.train()

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    for epoch in range(args.epoch, args.epochs):
        if accelerator.is_main_process and logging:
            wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), disable=not accelerator.is_main_process,
                             desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train", total=len(train_loader)):
            optimizer.zero_grad()

            input_data = batch
            depth = batch['depth']

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            depth_0, pred = model(input_data)
            mask = (depth > args.min_depth) & (depth < args.max_depth) & (pred > 0)

            log = log_loss(pred, depth, mask=mask.to(torch.bool), interpolate=False)
            mse = mse_loss(pred, depth, mask=mask.to(torch.bool))

            loss = log
            # loss = mse

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if logging and step % 5 == 0 and accelerator.is_main_process:
                wandb.log({f"Train/{log_loss.name}": log.detach().item()}, step=step)
                wandb.log({f"Train/{mse_loss.name}": mse.detach().item()}, step=step)

            step += 1
            scheduler.step()

            if step % args.validate_every == 0:
                model.eval()
                with (torch.no_grad()):
                    val_log = RunningAverage()
                    val_mse = RunningAverage()
                    metrics = RunningAverageDict()

                    for i, batch in tqdm(enumerate(test_loader), disable=not accelerator.is_main_process,
                                         desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Validation", total=len(test_loader)):

                        depth = batch['depth']
                        input_data = batch

                        if 'has_valid_depth' in batch and not batch['has_valid_depth']:
                            continue

                        depth_0, pred = model(input_data)
                        mask = (depth > args.min_depth) & (depth < args.max_depth) & (pred > 0)

                        log = log_loss(pred, depth, mask=mask.to(torch.bool), interpolate=False)
                        mse = mse_loss(pred, depth, mask=mask.to(torch.bool))

                        # Append local metrics
                        if mask.any():
                            val_log.append(log.detach().item())
                            val_mse.append(mse.detach().item())

                        # Logging first batch's prediction and ground truth
                        if logging and i == 0 and accelerator.is_main_process:
                            wandb_pred = wandb.Image(pred[0, 0].cpu(), caption=f"epoch:{epoch}")
                            wandb_depth = wandb.Image(depth[0, 0].cpu(), caption=f"epoch:{epoch}")
                            wandb.log({"pred": wandb_pred})
                            wandb.log({"depth": wandb_depth})

                        # Post-processing for metrics
                        pred = pred.squeeze().cpu().numpy()
                        pred[pred < args.min_depth_eval] = args.min_depth_eval
                        pred[pred > args.max_depth_eval] = args.max_depth_eval
                        pred[np.isinf(pred)] = args.max_depth_eval
                        pred[np.isnan(pred)] = args.min_depth_eval

                        gt_depth = depth.squeeze().cpu().numpy()
                        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
                        if valid_mask.any():
                            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask], gt_depth, pred, args.max_depth_eval))

                    # Convert metrics to CUDA tensors for gathering
                    gathered_val_log = accelerator.gather_for_metrics(torch.tensor([val_log.get_value()], dtype=torch.float32, device=accelerator.device)).mean().item()
                    gathered_val_mse = accelerator.gather_for_metrics(torch.tensor([val_mse.get_value()], dtype=torch.float32, device=accelerator.device)).mean().item()

                    metrics_dict = metrics.get_value()
                    gathered_metrics = {}
                    for key, value in metrics_dict.items():
                        tensor_value = torch.tensor([value], dtype=torch.float32, device=accelerator.device)  # Ensure it's a CUDA tensor
                        gathered_value = accelerator.gather_for_metrics(tensor_value)
                        gathered_metrics[key] = gathered_value.mean().item()  # Compute the mean for each metric

                    # Calculate and print final metrics only on the main process
                    if logging and accelerator.is_main_process:
                        print("Validated Metrics:", {key: value for key, value in gathered_metrics.items()})

                        wandb.log({f"Test/{log_loss.name}": gathered_val_log}, step=step)
                        wandb.log({f"Test/{mse_loss.name}": gathered_val_mse}, step=step)
                        wandb.log({f"Metrics/{k}": v for k, v in gathered_metrics.items()}, step=step)
                        save_checkpoint(model, optimizer, epoch, fpath=f"/data/depthor/checkpoints/{args.name}_{args.run_id}_latest.pt")
                        save_weights(model.module, fpath=f"/data/depthor/weights/{args.name}_{args.run_id}_latest.pt")
                        if gathered_metrics['rmse'] < best_loss:
                            save_checkpoint(model, optimizer, epoch, fpath=f"/data/depthor/checkpoints/{args.name}_{args.run_id}_best.pt")
                            save_weights(model.module, fpath=f"/data/depthor/weights/{args.name}_{args.run_id}_best.pt")
                            best_loss = gathered_metrics['rmse']
                        # if gathered_metrics['mae'] < best_loss:
                        #     save_checkpoint(model, optimizer, epoch, fpath=f"/data/depthor/checkpoints/{args.name}_{args.run_id}_best.pt")
                        #     save_weights(model.module, fpath=f"/data/depthor/weights/{args.name}_{args.run_id}_best.pt")
                        #     best_loss = gathered_metrics['mae']
                model.train()
    accelerator.end_training()
    return model


if __name__ == '__main__':

    from src.config import args

    if args.no_logging:
        globals()['logging'] = False

    main_worker(args)
