"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:32:33
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import mlflow
import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import set_torch_deterministic

from core import builder
from core.optimizer.base import EvolutionarySearchBase
from core.optimizer.nsga2 import NSGA2
from core.optimizer.utils import Evaluator


def search(
    model: nn.Module,
    es_engine: EvolutionarySearchBase,
    evaluator: Evaluator,
    args,
    checkpoint=None,
) -> None:
    model.eval()
    es_engine.initialize_population()
    genes = es_engine.ask(to_solution=False)
    for gene in genes:
        print(gene, "\n")

    best_score_values = []
    avg_score_values = []
    best_acc_values = []
    avg_acc_values = []
    best_area_values = []
    avg_area_values = []
    best_power_values = []
    avg_power_values = []
    best_latency_values = []
    avg_latency_values = []
    for k in range(int(args.evo_search.n_iterations)):
        if isinstance(es_engine, NSGA2):
            es_engine.generate_offsprings()
        solutions = es_engine.ask()
        genes = es_engine.ask(to_solution=False)
        (
            scores,
            best_solution_cost_dict,
            best_solution_score,
            acc_values,
            area_values,
            power_values,
            latency_values,
        ) = evaluator.evaluate_all(
            genes, solutions, None, k, args.evo_search.population_size
        )

        es_engine.tell(scores)
        data_dict = {"Best score": es_engine.best_score}
        data_dict.update(best_solution_cost_dict)
        lg.info(f"ES iteration {k}, data_dict: {data_dict}")
        lg.info(f"Scores: {scores}")
        lg.info(f"Acc: {acc_values}")
        lg.info(f"Area: {area_values}")
        lg.info(f"Power:{power_values}")
        lg.info(f"Latency:{latency_values}")

        mlflow.log_metrics(
            {
                "search_score": es_engine.best_score,
            },
            step=k,
        )
        assert best_solution_score == es_engine.best_score

        # avg_acc = np.mean([val.cpu().numpy() for val in acc_values])
        avg_acc = np.mean(acc_values)
        avg_area = np.mean(area_values)
        avg_power = np.mean(power_values)
        avg_latency = np.mean(latency_values)
        # avg_score = np.mean([score.cpu().numpy() for score in scores])
        avg_score = np.mean(scores)
        avg_score_values.append(avg_score)
        avg_acc_values.append(avg_acc)
        avg_area_values.append(avg_area)
        avg_power_values.append(avg_power)
        avg_latency_values.append(avg_latency)
        best_score_values.append(best_solution_score)
        best_acc_values.append(best_solution_cost_dict["accuracy"])
        best_area_values.append(best_solution_cost_dict["area"])
        best_power_values.append(best_solution_cost_dict["power"])
        best_latency_values.append(best_solution_cost_dict["latency"])

        es_engine.dump_solution_to_file(
            best_sol=es_engine.best_solution,
            best_score=best_solution_score,
            cost_dict=best_solution_cost_dict,
            filename=checkpoint,
        )
        lg.info("\n\n")
        # store the model and solution after every iteration
    lg.info(f"ES iteration Done. Best solution core = {best_solution_score}")
    # eval the best solution
    (
        scores,
        best_solution_cost_dict,
        best_solution_score,
        acc_values,
        area_values,
        power_values,
        latency_values,
    ) = evaluator.evaluate_all([es_engine.best_gene], [es_engine.best_solution])
    data_dict = {"Best score": es_engine.best_score}
    data_dict.update(best_solution_cost_dict)
    lg.info(f"SOLUTION: {es_engine.best_solution}")
    avg_values_pack = (
        avg_score_values,
        avg_acc_values,
        avg_area_values,
        avg_power_values,
        avg_latency_values,
    )
    best_values_pack = (
        best_acc_values,
        best_area_values,
        best_power_values,
        best_latency_values,
    )
    return (
        es_engine.best_solution,
        es_engine.best_score,
        best_solution_cost_dict,
        best_score_values,
        avg_values_pack,
        best_values_pack,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    # print(configs)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    _, validation_loader, _ = builder.make_dataloader(splits=["valid"])

    model = builder.make_model(
        device,
        model_cfg=configs.model,
        random_state=int(configs.run.random_state)
        if int(configs.run.deterministic)
        else None,
    )

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )

    es_engine, evaluator = builder.make_search_engine(
        name=configs.evo_search.name,
        model=model,
        calibration_loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.yaml"

    lg.info(f"Current checkpoint: {checkpoint}")

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

    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
            # f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({socket.gethostname()})"
        )
        lg.info(configs)

        search(
            model=model,
            es_engine=es_engine,
            evaluator=evaluator,
            args=configs,
            checkpoint=checkpoint,
        )

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")

    mlflow.end_run()


if __name__ == "__main__":
    main()
