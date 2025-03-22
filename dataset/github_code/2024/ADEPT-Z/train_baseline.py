"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:32:44
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from typing import Callable, Dict, Iterable, Optional

import mlflow
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder
from core.models.layers.super_mesh import SuperZeroCRLayer, SuperZeroDCLayer
from core.optimizer.utils import Converter


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    grad_scaler: Optional[Callable] = None,
    teacher: Optional[nn.Module] = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)

    class_meter = AverageMeter("ce")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)

    data_counter = 0
    correct = 0

    total_data = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        data_counter += data.shape[0]

        target = target.to(device, non_blocking=True)

        with amp.autocast(enabled=grad_scaler._enabled):
            output = model(data)
            output = output.to(device, non_blocking=True)
            class_loss = criterion(output, target)
            class_meter.update(class_loss.item())
            loss = class_loss

            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                aux_loss = 0
                if name == "kl_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher_scores = teacher(data).data.detach()
                    T = 2
                    p = F.log_softmax(output / T, dim=1)
                    q = F.softmax(teacher_scores / T, dim=1)
                    aux_loss = F.kl_div(p, q, reduction="batchmean") * (T**2) * weight
                elif name == "mse_distill" and teacher is not None:
                    with torch.no_grad():
                        teacher(data).data.detach()
                    teacher_hiddens = [
                        m._recorded_hidden
                        for m in teacher.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]
                    student_hiddens = [
                        m._recorded_hidden
                        for m in model.modules()
                        if hasattr(m, "_recorded_hidden")
                    ]

                    aux_loss = weight * sum(
                        F.mse_loss(h1, h2)
                        for h1, h2 in zip(teacher_hiddens, student_hiddens)
                    )
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        if configs.run.grad_clip:
            max_grad_value = float(configs.run.max_grad_value)

            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    if p.grad.is_complex():
                        # Clip the real part
                        p.grad.real.clamp_(-max_grad_value, max_grad_value)
                        # Clip the imaginary part
                        p.grad.imag.clamp_(-max_grad_value, max_grad_value)
                    else:
                        p.grad.clamp_(-max_grad_value, max_grad_value)

        grad_scaler.step(optimizer)
        grad_scaler.update()
        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} class Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                class_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    avg_class_loss = class_meter.avg
    accuracy = 100.0 * correct.to(torch.float32) / total_data
    lg.info(
        f"Train class Loss: {avg_class_loss:.4e}, Accuracy: {correct}/{total_data} ({accuracy:.2f}%)"
    )
    mlflow.log_metrics(
        {
            "train_class": avg_class_loss,
            "train_acc": accuracy.item(),
            "lr": get_learning_rate(optimizer),
        },
        step=epoch,
    )


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("ce")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(validation_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    lg.info(
        f"\nValidation set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    mlflow.log_metrics(
        {"val_loss": class_meter.avg, "val_acc": accuracy.item()}, step=epoch
    )


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    fp16: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    correct = 0
    class_meter = AverageMeter("mse")
    with amp.autocast(enabled=fp16):
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(data)

                val_loss = criterion(output, target)
                class_meter.update(val_loss.item())
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()

    loss_vector.append(class_meter.avg)
    accuracy = 100.0 * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        f"\nTest set: Average loss: {class_meter.avg:.4e}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    mlflow.log_metrics(
        {"test_loss": class_meter.avg, "test_acc": accuracy.item()}, step=epoch
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    train_loader, validation_loader, test_loader = builder.make_dataloader(
        splits=["train", "valid", "test"]
    )

    if (
        configs.run.do_distill
        and configs.teacher is not None
        and os.path.exists(configs.teacher.checkpoint)
    ):
        teacher = builder.make_model(device, model_cfg=configs.teacher)
        load_model(teacher, path=configs.teacher.checkpoint)
        teacher.eval()
        lg.info(f"Load teacher model from {configs.teacher.checkpoint}")
    else:
        teacher = None

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state)
        if int(configs.run.deterministic)
        else None,
    )

    converter = Converter(model.super_layer)

    gene_file = configs.gene.file_path

    with open(gene_file, "r") as file:
        gene = yaml.safe_load(file)

    lg.info(f"Current Gene for training: {gene}")

    if isinstance(gene, str):
        gene = eval(gene)

    solution = converter.gene2solution(gene)
    lg.info(f"Current Solution for training: {solution}")

    optimizer = builder.make_optimizer(
        model.get_parameter_groups(
            weight_decay=float(configs.optimizer.weight_decay), lr=configs.optimizer.lr
        ),
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )

    _, evaluator = builder.make_search_engine(
        name=configs.evo_search.name,
        model=model,
        calibration_loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    val_accuracies = []
    test_accuracies = []

    lg.info(configs)

    k = 8

    # Start training each model
    lg.info("Start evaluating solution. \n")
    lg.info(f"{solution} \n")

    # Reset phi and sigma
    model.reset_parameters(int(configs.run.random_state))

    # Fix the current arch solution into the model.
    model.fix_arch_solution(solution)

    area = evaluator.cost_predictor._evaluate_area(solution)
    power = evaluator.cost_predictor._evaluate_power(solution)
    latency = evaluator.cost_predictor._evaluate_latency(solution)
    cd = k * (2 * k - 1) / (latency * area * 1e-6)
    ee = k * (2 * k - 1) / (latency * power * 1e-3)

    lg.info(f"Area for solution: {area}")
    lg.info(f"Power for solution: {power}")
    lg.info(f"Power for solution: {latency}")
    lg.info(f"Compute Density for solution: {cd} TOPS/(mm^2)")
    lg.info(f"Energy Efficiency for solution: {ee} TOPS/W")

    # exit(0)
    # Make auxiliary criterions
    aux_criterions = dict()
    if configs.aux_criterion is not None:
        for name, config in configs.aux_criterion.items():
            if float(config.weight) > 0:
                try:
                    fn = builder.make_criterion(name, cfg=config)
                except NotImplementedError:
                    fn = name
                aux_criterions[name] = [fn, float(config.weight)]
    print(aux_criterions)

    # Make model saver
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=True,
        truncate=2,
        metric_name="acc",
        format="{:.2f}",
    )

    # Make grad_scaler
    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))

    # Define the location of checkpoints
    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    # Set the experiment
    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    # Loss vector, accuracy vector
    lossv, accv, acct = [0], [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )

        if teacher is not None:
            test(
                teacher,
                validation_loader,
                0,
                criterion,
                [],
                [],
                device,
                fp16=grad_scaler._enabled,
            )
            lg.info("Map teacher to student...")
            if hasattr(model, "load_from_teacher"):
                with amp.autocast(grad_scaler._enabled):
                    model.load_from_teacher(teacher)

        # For loop to train the model for multiple epochs
        for epoch in range(1, int(configs.run.n_epochs) + 1):
            # if no noise-aware training
            model.set_phase_noise(0)
            # set dc and cr noise
            for m in model.super_layer.super_layers_all:
                if isinstance(m, SuperZeroDCLayer):
                    m.set_dc_noise(noise_std=0)
                if isinstance(m, SuperZeroCRLayer):
                    m.set_cr_noise(tr_noise_std=0, phase_noise_std=0)

            # if noise-aware training
            # model.set_phase_noise(float(configs.evo_search.robustness.phase_noise_std))
            # # set dc, cr noise
            # for m in model.super_layer.super_layers_all:
            #     if isinstance(m, SuperZeroDCLayer):
            #         m.set_dc_noise(noise_std = float(configs.evo_search.robustness.dc_noise_std))
            #     if isinstance(m,SuperZeroCRLayer):
            #         m.set_cr_noise(tr_noise_std=float(configs.evo_search.robustness.cr_tr_noise_std),
            #                         phase_noise_std=float(configs.evo_search.robustness.cr_phase_noise_std)*(np.pi)/180)

            # Training process
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                device,
                grad_scaler=grad_scaler,
                teacher=teacher,
            )

            # Validation Process
            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,  # validation accuracy
                    device,
                    fp16=grad_scaler._enabled,
                )

            # Noise-free test
            test(
                model,
                test_loader,
                epoch,
                criterion,
                lossv if validation_loader is None else [],
                # accv if validation_loader is None else [],
                acct,
                device,
                fp16=grad_scaler._enabled,
            )

            # Enable all the noises when getting the test accuracy
            # Set phase noise
            model.set_phase_noise(float(configs.evo_search.robustness.phase_noise_std))
            # Set dc, cr noise
            for m in model.super_layer.super_layers_all:
                if isinstance(m, SuperZeroDCLayer):
                    m.set_dc_noise(
                        noise_std=float(configs.evo_search.robustness.dc_noise_std)
                    )
                if isinstance(m, SuperZeroCRLayer):
                    m.set_cr_noise(
                        tr_noise_std=float(
                            configs.evo_search.robustness.cr_tr_noise_std
                        ),
                        phase_noise_std=float(
                            configs.evo_search.robustness.cr_phase_noise_std
                        )
                        * (np.pi)
                        / 180,
                    )

            # Repeat getting noisy accuracy for 5 times, choose the average value as the final noisy accuracy
            temp_acct = []
            for _ in range(5):
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    lossv if validation_loader is None else [],
                    # accv if validation_loader is None else [],
                    temp_acct,
                    device,
                    fp16=grad_scaler._enabled,
                )
            # Calculate the average test accuracy and std of 5 tests
            avg_acct = np.mean(temp_acct)
            std_acct = np.std(temp_acct)
            lg.info(f"Average test accuracy: {avg_acct}")
            lg.info(f"The standard deviation of test accuracies: {std_acct}")
            acct.append(avg_acct)

            saver.save_model(
                model,
                accv[-1],
                epoch=epoch,
                path=checkpoint,
                save_model=False,
                print_msg=True,
            )

        # validation/test accuracy of all epochs for this solution
        val_data = [x.item() for x in accv[1:]]
        test_data = acct[2::2]

        val_accuracies.append(val_data)
        test_accuracies.append(test_data)
        lg.info(f"Current validation accuracies are: {val_data} \n")
        lg.info(f"Current test accuracies are: {test_data} \n")

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")

    mlflow.end_run()

    lg.info(f"Estimations of area are: {area} \n")
    lg.info(f"Estimations of power are: {power} \n")
    lg.info(f"Estimations of latency are: {latency} \n")
    lg.info(f"Estimations of Compute Density are: {cd} \n")
    lg.info(f"Estimations of Energy Efficiency are: {ee} \n")
    lg.info("Finished.")


if __name__ == "__main__":
    main()
