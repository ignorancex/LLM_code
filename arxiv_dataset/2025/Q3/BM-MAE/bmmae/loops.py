from collections import defaultdict

import numpy as np
import torch
import tqdm
import wandb
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import average_precision_score, roc_auc_score
from sksurv.metrics import concordance_index_censored
from .utils import patchify, visualize_multimodal_3d


def loop_pretrain(
    model,
    modalities,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    epochs,
    device,
    norm_pix_loss,
    wandb_logging,
):
    best_loss = 1000000
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # add lr to metrics
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        total_modality_loss = {modality: 0.0 for modality in modalities}
        for i, batch_data in enumerate(tqdm.tqdm(train_loader, desc="Training...")):
            inputs = {modality: batch_data[modality].to(device) for modality in modalities}
            optimizer.zero_grad()
            outputs, mask = model(inputs)
            modality_loss = {}
            for modality in modalities:
                target = patchify(inputs[modality])
                if norm_pix_loss:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, keepdim=True)
                    target = (target - mean) / ((var + 1e-6) ** 0.5)

                temp_loss = (outputs[modality] - target) ** 2
                temp_loss = temp_loss.mean(-1)
                temp_loss = (temp_loss * mask[modality]).sum() / mask[modality].sum()
                modality_loss[modality] = temp_loss

            loss = sum(modality_loss.values())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs["flair"].shape[0]
            for modality in modality_loss:
                total_modality_loss[modality] += modality_loss[modality].item() * batch_data[modality].shape[0]

        lr_scheduler.step()

        epoch_loss /= len(train_loader.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        for modality in total_modality_loss:
            total_modality_loss[modality] /= len(train_loader.dataset)
            print(f"{modality} loss: {total_modality_loss[modality]:.4f}")
            metrics[f"train/{modality}_loss"].append(total_modality_loss[modality])
        metrics["train/loss"].append(epoch_loss)

        # enter val loop on first and last epochs and every 10 epochs
        if epoch == 0 or epoch == epochs - 1 or epoch % 1 == 0:
            val_loss = 0.0
            total_modality_loss = {modality: 0.0 for modality in modalities}
            model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(tqdm.tqdm(val_loader, desc="Validation...")):
                    inputs = {modality: batch_data[modality].to(device) for modality in modalities}

                    outputs, mask = model(inputs)

                    modality_loss = {}
                    for modality in modalities:
                        target = patchify(inputs[modality])
                        if norm_pix_loss:
                            mean = target.mean(dim=-1, keepdim=True)
                            var = target.var(dim=-1, keepdim=True)
                            target = (target - mean) / ((var + 1e-6) ** 0.5)
                        temp_loss = (outputs[modality] - target) ** 2
                        temp_loss = temp_loss.mean(-1)
                        temp_loss = (temp_loss * mask[modality]).sum() / mask[modality].sum()
                        modality_loss[modality] = temp_loss


                    loss = sum(modality_loss.values())
                    val_loss += loss.item() * inputs["flair"].shape[0]
                    for modality in modality_loss:
                        total_modality_loss[modality] += modality_loss[modality].item() * batch_data[modality].shape[0]

                    if i == 0:
                        fig = visualize_multimodal_3d(outputs, inputs, mask)
                        metrics["Val/plot"].append(fig)

                val_loss /= len(val_loader.dataset)
                print(f"Validation loss: {val_loss:.4f}")
                for modality in total_modality_loss:
                    total_modality_loss[modality] /= len(val_loader.dataset)
                    print(f"{modality} loss: {total_modality_loss[modality]:.4f}")
                    metrics[f"val/{modality}_loss"].append(total_modality_loss[modality])
                metrics["val/loss"].append(val_loss)

                
                best_loss = metrics["val/loss"][-1]
                torch.save(
                    model.state_dict(),
                    f"pretrained_models/bmmae.pth",
                )

        if wandb_logging:
            print("Logging to wandb")
            wandb.log({k: v[-1] for k, v in metrics.items()})
            

    return model, metrics


def loop_classification(epochs, model, modalities, optimizer, lr_scheduler, train_dl, val_dl, device, wandb_logging):
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # add lr to metrics
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        for i, batch_data in enumerate(tqdm.tqdm(train_dl, desc="Training...")):
            inputs = {modality: batch_data[modality].to(device) for modality in modalities}
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.shape[0]

        lr_scheduler.step()
        epoch_loss /= len(train_dl.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        metrics["train/loss"].append(epoch_loss)

        val_loss = 0.0
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm.tqdm(val_dl, desc="Validation...")):
                inputs = {modality: batch_data[modality].to(device) for modality in modalities}
                labels = batch_data["label"].to(device)
                outputs = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(1), labels)
                val_loss += loss.item() * labels.size(0)
                preds.append(outputs)
                targets.append(labels)

            val_loss /= len(val_dl.dataset)
            print(f"Validation loss: {val_loss:.4f}")
            metrics["val/loss"].append(val_loss)
            # roc_auc and ap
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            roc_auc = roc_auc_score(targets.cpu().numpy(), torch.sigmoid(preds).cpu().numpy())
            ap = average_precision_score(targets.cpu().numpy(), torch.sigmoid(preds).cpu().numpy())
            metrics["val/roc_auc"].append(roc_auc)
            metrics["val/ap"].append(ap)
            print(f"roc_auc: {roc_auc:.4f} ap: {ap:.4f}")

        if wandb_logging:
            wandb.log({k: v[-1] for k, v in metrics.items()})

    return model, metrics


def loop_segmentation(
    epochs, model, loss_fn, opt, scheduler, train_dl, valid_dl, device, path_to_save, wandb_logging
):
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # if last epoch compute hd95
        if epoch == epochs - 1:
            hd95_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)
        # log learning rate
        metrics["lr"].append(opt.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")

        print("-" * 10)
        model.train()
        epoch_loss = 0.0

        for i, batch_data in enumerate(tqdm.tqdm(train_dl, desc="Training...")):
            inputs, mask = (batch_data["image"].to(device), batch_data["label"].to(device))

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, mask)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * inputs.shape[0]

        scheduler.step()
        epoch_loss /= len(train_dl.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        metrics["train/loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

            val_loss = 0.0
            for batch_data in tqdm.tqdm(valid_dl, desc="Validation..."):
                inputs, mask = (batch_data["image"].to(device), batch_data["label"].to(device))
                outputs = model(inputs)
                loss = loss_fn(outputs, mask)
                val_loss += loss.item() * inputs.size(0)
                try:
                    outputs = outputs.as_tensor()
                except:
                    pass
                # put everything on cpu
                outputs = outputs.cpu()
                mask = mask.cpu()
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                dice_metric(y_pred=outputs, y=mask)
                if epoch == epochs - 1:
                    hd95_metric(y_pred=outputs, y=mask)

            dice = dice_metric.aggregate("none").numpy()

            mean_dice = np.nanmean(dice, axis=0)

            mean_dice_tc = mean_dice[0]
            mean_dice_wt = mean_dice[1]
            mean_dice_et = mean_dice[2]

            std_dice = np.nanstd(dice, axis=0)

            std_dice_tc = std_dice[0]
            std_dice_wt = std_dice[1]
            std_dice_et = std_dice[2]

            metrics["Val/TC Dice"].append(mean_dice_tc)
            metrics["Val/TC Dice std"].append(std_dice_tc)
            metrics["Val/WT Dice"].append(mean_dice_wt)
            metrics["Val/WT Dice std"].append(std_dice_wt)
            metrics["Val/ET Dice"].append(mean_dice_et)
            metrics["Val/ET Dice std"].append(std_dice_et)
            val_loss /= len(valid_dl.dataset)
            metrics["Val/loss"].append(val_loss)
            # extend to combine both dict lw_metrics dict
            dice_metric.reset()
            if epoch == epochs - 1:
                hd95 = hd95_metric.aggregate("none").numpy()
                mean_hd95 = np.nanmean(hd95, axis=0)

                mean_hd95_tc = mean_hd95[0]
                mean_hd95_wt = mean_hd95[1]
                mean_hd95_et = mean_hd95[2]

                std_hd95 = np.nanstd(hd95, axis=0)

                std_hd95_tc = std_hd95[0]
                std_hd95_wt = std_hd95[1]
                std_hd95_et = std_hd95[2]

                metrics["Val/TC HD95"].append(mean_hd95_tc)
                metrics["Val/TC HD95 std"].append(std_hd95_tc)
                metrics["Val/WT HD95"].append(mean_hd95_wt)
                metrics["Val/WT HD95 std"].append(std_hd95_wt)
                metrics["Val/ET HD95"].append(mean_hd95_et)
                metrics["Val/ET HD95 std"].append(std_hd95_et)
                hd95_metric.reset()
                

            if wandb_logging:
                wandb.log({k: v[-1] for k, v in metrics.items()})

            if path_to_save is not None:
                torch.save(
                    model.state_dict(),
                    path_to_save,
                )

            print(
                f"Validation loss: {val_loss:.4f}\n"
                f"TC Dice: {mean_dice_tc:.4f} +/- {std_dice_tc:.4f}\n "
                f"WT Dice: {mean_dice_wt:.4f} +/- {std_dice_wt:.4f}\n "
                f"ET Dice: {mean_dice_et:.4f} +/- {std_dice_et:.4f}\n"
            )

    return model, metrics

def loop_survival(epochs, model, modalities, optimizer, lr_scheduler, train_dl, val_dl, device, wandb_logging):
    from pycox.models.loss import NLLLogistiHazardLoss
    loss_fn = NLLLogistiHazardLoss()
    metrics = defaultdict(list)
    for epoch in range(epochs):
        # add lr to metrics
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        model.train()
        epoch_loss = 0.0
        for i, batch_data in enumerate(tqdm.tqdm(train_dl, desc="Training...")):
            inputs = {modality: batch_data[modality].to(device) for modality in modalities}
            time_idx = batch_data["time_idx"].to(device)
            event = batch_data["event"].to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, time_idx, event)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * time_idx.shape[0]

        lr_scheduler.step()
        epoch_loss /= len(train_dl.dataset)
        print(f"Training loss: {epoch_loss:.4f}")
        metrics["train/loss"].append(epoch_loss)

        val_loss = 0.0
        model.eval()
        times = []
        events = []
        risks = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm.tqdm(val_dl, desc="Validation...")):
                inputs = {modality: batch_data[modality].to(device) for modality in modalities}
                time_idx = batch_data["time_idx"].to(device)
                event = batch_data["event"].to(device).float()
                outputs = model(inputs)
                loss = loss_fn(outputs, time_idx, event)
                val_loss += loss.item() * time_idx.size(0)
                times.append(batch_data["time"].numpy())
                events.append(batch_data["event"].numpy())
                risks.append(outputs.cpu().numpy().sum(axis=1)) 

            val_loss /= len(val_dl.dataset)
            print(f"Validation loss: {val_loss:.4f}")
            metrics["val/loss"].append(val_loss)
            # roc_auc and ap
            times = np.concatenate(times)
            events = np.concatenate(events)
            risks = np.concatenate(risks)
            c_index = concordance_index_censored(events.astype(bool), times, risks)[0]
            metrics["val/c_index"].append(c_index)
            print(f"c_index: {c_index:.4f}")

        if wandb_logging:
            wandb.log({k: v[-1] for k, v in metrics.items()})

    if modalities == ["t1", "t1ce", "t2", "flair"]:
        # create a df with the risks and times and events
        import pandas as pd
        risk_df = pd.DataFrame({"risk": risks, "time_loop": times, "event_loop": events})
        metrics["risk_df"] = risk_df

    return model, metrics
