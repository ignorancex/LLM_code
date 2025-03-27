"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-10 20:34:02
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:50:36
"""

#!/usr/bin/env python
# coding=UTF-8
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from typing import Dict

import mlflow
import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.plot import plt, set_ms
from pyutils.torch_train import set_torch_deterministic

set_ms()
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

from core import builder
from core.optimizer.base import EvolutionarySearchBase
from core.optimizer.nsga2 import NSGA2
from core.optimizer.utils import Converter, CosineDecay, Evaluator


def plot_pareto_front(objective_values, objective_values_0, figure_save_path):
    objective_values = np.array(objective_values)
    objective_values_0 = np.array(objective_values_0)

    x = objective_values[:, 0]  # Acc
    y = objective_values[:, 1]  # Compute_Density
    z = objective_values[:, 2]  # Energy_Efficiency

    x_0 = objective_values_0[:, 0]  # Acc
    y_0 = objective_values_0[:, 1]  # Compute_Density
    z_0 = objective_values_0[:, 2]  # Energy_Efficiency

    # add 2 baseline points to the graph:
    mzi_objectives = [96.185, 0.025297404651095055, 0.4445337884699598]
    mzi_objectives_x = [mzi_objectives[0], mzi_objectives[1]]
    mzi_objectives_y = [mzi_objectives[0], mzi_objectives[2]]
    mzi_objectives_z = [mzi_objectives[1], mzi_objectives[2]]

    butterfly_objectives = [95.06, 0.46527204747410833, 21.84202971542177]

    butterfly_objectives_x = [butterfly_objectives[0], butterfly_objectives[1]]
    butterfly_objectives_y = [butterfly_objectives[0], butterfly_objectives[2]]
    butterfly_objectives_z = [butterfly_objectives[1], butterfly_objectives[2]]

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(x_0, y_0, z_0, c="red", marker="o", label="1st iteration")
    ax1.scatter(x, y, z, c="blue", marker="o", label="40th iteration")
    ax1.scatter(*mzi_objectives, color="black", s=100, label="MZI")
    ax1.scatter(*butterfly_objectives, color="gold", s=100, label="Butterfly")
    ax1.set_title("Original 3D Scatter Plot")
    ax1.set_xlabel("Accuracy")
    ax1.set_ylabel("Compute_Density")
    ax1.set_zlabel("Energy_Efficiency")

    ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.scatter(
        x_0, y_0, np.min(z_0) - 1, c="red", marker="o", alpha=0.3, label="1st iteration"
    )
    ax2.scatter(
        x, y, np.min(z) - 1, c="blue", marker="o", alpha=0.3, label="40th iteration"
    )
    ax2.scatter(*mzi_objectives_x, color="black", s=100, label="MZI")
    ax2.scatter(*butterfly_objectives_x, color="gold", s=100, label="Butterfly")
    ax2.set_title("Projection on Z Plane")
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Compute_Density")

    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.scatter(x_0, z_0, c="red", marker="o", alpha=0.3, label="1st iteration")
    ax3.scatter(x, z, c="blue", marker="o", alpha=0.3, label="40th iteration")
    ax3.scatter(*mzi_objectives_y, color="black", s=100, label="MZI")
    ax3.scatter(*butterfly_objectives_y, color="gold", s=100, label="Butterfly")
    ax3.set_title("Projection on Y Plane")
    ax3.set_xlabel("Accuracy")
    ax3.set_ylabel("Energy_Efficiency")

    ax3.legend()

    ax4 = fig.add_subplot(224)
    ax4.scatter(y_0, z_0, c="red", marker="o", alpha=0.3, label="1st iteration")
    ax4.scatter(y, z, c="blue", marker="o", alpha=0.3, label="40th iteration")
    ax4.scatter(*mzi_objectives_z, color="black", s=100, label="MZI")
    ax4.scatter(*butterfly_objectives_z, color="gold", s=100, label="Butterfly")
    ax4.set_title("Projection on X Plane")
    ax4.set_xlabel("Compute_Density")
    ax4.set_ylabel("Energy_Efficiency")

    ax4.legend()

    plt.tight_layout()

    fig.savefig(fname=figure_save_path, dpi=300)


def search(
    model: nn.Module,
    es_engine: EvolutionarySearchBase,
    evaluator: Evaluator,
    converter: Converter,
    params: Dict,
    scheduler_dict: Dict[str, CosineDecay],
    figure_save_path,
    args,
    checkpoint=None,
) -> None:
    model.eval()
    es_engine.initialize_population()
    genes = es_engine.ask(to_solution=False)
    solutions = es_engine.ask()
    lg.info("Initial Population:")
    for gene in genes:
        lg.info(f"{gene}")

    objective_values = {}
    objectives_by_front = {}

    # evaluate the initialize population and save the objectives
    (score_iter0, eval_results_iter0) = evaluator.evaluate_all(model, genes, solutions)
    lg.info("Evaluation results for the initial population:")
    lg.info(f"Scores: {score_iter0}")

    # collect all solutions on pareto_front 0
    pareto_front0 = []

    for k in range(int(args.evo_search.n_iterations)):
        lg.info(f"Iteration:{k}")

        # update parameters for mutation/crossover
        for name, scheduler in scheduler_dict.items():
            scheduler.step()
            params[name] = scheduler.get_dr(death_rate=scheduler_dict[name].init_value)
            lg.info(f"{name} updated to {params[name]}")

        es_engine.set_parameters(params=params)

        # update mutation/crossover operators
        if k < args.evo_search.n_global_search:
            # global search
            mutation_ops_dc = ["op1_dc", "op2_dc", "op3_dc", "op4_dc"]
            mutation_ops_cr = ["op1_cr", "op2_cr", "op3_cr"]
        else:
            # local search
            mutation_ops_dc = ["op1_dc", "op2_dc", "op3_dc"]
            mutation_ops_cr = ["op1_cr", "op2_cr"]
            final_params = {
                "mutation_rate_dc": 0.01,
                "mutation_rate_cr": 0.01,
                "mutation_rate_block": 0,
                "crossover_rate_dc": 0.1,
                "cr_split_ratio": 0.05,
                "crossover_rate_block": 0,
            }
            es_engine.set_parameters(params=final_params)

        es_engine.set_operators(
            mutation_ops_dc=mutation_ops_dc, mutation_ops_cr=mutation_ops_cr
        )

        scores = []
        selected_results = {}

        if isinstance(es_engine, NSGA2):
            es_engine.generate_offsprings()

        # get all genes and solutions in current population
        solutions = es_engine.ask()
        genes = es_engine.ask(to_solution=False)
        lg.info(f"current number of genes in the population is {len(genes)}")

        # In multiobjective evaluation, we evaluate scores and results for all genes/solutions in current population
        (scores, eval_results) = evaluator.evaluate_all(model, genes, solutions)
        # print("scores:", scores)

        lg.info("Evaluation for all genes is completed.")
        # select current population based on the scores
        es_engine.tell(scores)
        # after this step, the population is selected

        selected_genes = []
        for front_index, solutions in es_engine.pareto_fronts.items():
            if front_index == 0:
                for uid, solution_info in solutions.items():
                    selected_genes.append(solution_info["solution"])
        selected_solutions = [converter.gene2solution(gene) for gene in selected_genes]

        if k == int(args.evo_search.n_iterations) - 1:
            lg.info("Final set of Gene:")
            for gene in selected_genes:
                lg.info(f"{gene}")
            lg.info("End")

        # print("Length of selected_solutions:", len(selected_solutions))

        selected_results = {
            key: eval_results[key] for key in selected_solutions if key in eval_results
        }

        # print("Length of selected_results:", len(selected_results))

        es_engine.dump_solution_to_file(best_sol=selected_results, filename=checkpoint)

        for front_index, solutions in es_engine.pareto_fronts.items():
            objective_values[front_index] = []
            for uid, solution_info in solutions.items():
                objective_values[front_index].append(solution_info["objectives"])

        objective_values[0] = sorted(
            objective_values[0], key=lambda x: x[0], reverse=True
        )

        for front_index, objectives_list in objective_values.items():
            lg.info(f"Pareto Front {front_index}")
            lg.info(f"{objectives_list}")
            if front_index == 0:
                pareto_front0.append(objectives_list)

        lg.info(f"Pareto front 0 after {k} iteration: ")
        for i in range(len(pareto_front0)):
            lg.info(f"{pareto_front0[i]}")

        plot_pareto_front(
            objective_values=pareto_front0[-1],
            objective_values_0=pareto_front0[0],
            figure_save_path=figure_save_path,
        )
        lg.info("Pareto front plotted.")

    for i in range(len(pareto_front0)):
        print(f"Iteration {i}, the pareto front 0 is: ")
        print(pareto_front0[i])


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

    params = {
        "mutation_rate_dc": configs.evo_search.mutation_rate_dc,
        "mutation_rate_cr": configs.evo_search.mutation_rate_cr,
        "mutation_rate_block": configs.evo_search.mutation_rate_block,
        "crossover_rate_dc": configs.evo_search.crossover_rate_dc,
        "cr_split_ratio": configs.evo_search.crossover_cr_split_ratio,
        "crossover_rate_block": configs.evo_search.crossover_rate_block,
    }

    schedulers = {
        "mutation_rate_dc": CosineDecay(
            death_rate=configs.evo_search.mutation_rate_dc,
            T_max=configs.evo_search.n_global_search,
            eta_min=0.01,
        ),
        "mutation_rate_cr": CosineDecay(
            death_rate=configs.evo_search.mutation_rate_cr,
            T_max=configs.evo_search.n_global_search,
            eta_min=0.01,
        ),
        "mutation_rate_block": CosineDecay(
            death_rate=configs.evo_search.mutation_rate_block,
            T_max=configs.evo_search.n_global_search,
            eta_min=0,
        ),
        "crossover_rate_dc": CosineDecay(
            death_rate=configs.evo_search.crossover_rate_dc,
            T_max=configs.evo_search.n_global_search,
            eta_min=0.1,
        ),
        "cr_split_ratio": CosineDecay(
            death_rate=configs.evo_search.crossover_cr_split_ratio,
            T_max=configs.evo_search.n_global_search,
            eta_min=0.05,
        ),
        "crossover_rate_block": CosineDecay(
            death_rate=configs.evo_search.crossover_rate_block,
            T_max=configs.evo_search.n_global_search,
            eta_min=0,
        ),
    }

    converter = Converter(model.super_layer)

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.yaml"

    # figure_save_path = "./figures/Search_results_8.png"
    figure_save_path = "./figures/Search_results_16.png"

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
        )
        lg.info(configs)

        search(
            model=model,
            es_engine=es_engine,
            evaluator=evaluator,
            converter=converter,
            params=params,
            scheduler_dict=schedulers,
            figure_save_path=figure_save_path,
            args=configs,
            checkpoint=checkpoint,
        )

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")

    mlflow.end_run()


def test_plot_pareto_front():
    objective_values = np.array(
        [
            [100.148, 0.4025864112041174, 21.982739179945252],
            [100.086, 0.4025962144157178, 21.99348719064612],
            [99.967, 0.4025962144157178, 21.99348719064612],
            [99.854, 0.4060334634181596, 21.998628375865216],
            [99.69200000000001, 0.4025994822590138, 21.998628375865216],
            [99.634, 0.4025994822590138, 21.998628375865216],
            [99.583, 0.41596150150173333, 22.011018285493094],
            [99.55, 0.42337462648328394, 22.011018285493094],
            [99.51, 0.42717014906336603, 22.00624783177668],
            [99.064, 0.4381727336210598, 22.018088003567588],
            [99.021, 0.4341649253618673, 22.02251409164717],
            [98.968, 0.4361506935256798, 22.018088003567588],
            [98.91, 0.4361545287897971, 22.02251409164717],
            [98.879, 0.47746219969307413, 21.887566906790674],
            [98.114, 0.4798834465277117, 21.903735337071268],
            [97.721, 0.448461549219608, 22.00399817486451],
            [97.63, 0.44849398991469397, 22.018088003567588],
            [97.53, 0.45464432968765295, 21.911472171328214],
            [97.529, 0.44638780575647746, 22.018088003567588],
            [97.522, 0.4463918231731398, 22.018088003567588],
            [97.291, 0.4485061109626761, 22.013529284480658],
            [97.242, 0.4485101665987759, 22.018088003567588],
            [97.178, 0.4704960591344944, 21.94025211202431],
            [96.958, 0.454890337458636, 21.98316437293588],
            [96.923, 0.45490702558850554, 21.999017611840017],
            [96.87, 0.45494457835795177, 22.02251409164717],
            [96.785, 0.45494457835795177, 22.02251409164717],
            [96.663, 0.4777595953863234, 21.993888066766996],
            [96.535, 0.4778086916576129, 22.018088003567588],
            [96.389, 0.4824372774833438, 21.94025211202431],
            [96.385, 0.4826000084797492, 21.98860515638208],
            [96.311, 0.48489831893417346, 21.94025211202431],
            [96.243, 0.4730604419827363, 22.018088003567588],
            [96.201, 0.47072078919650534, 22.018088003567588],
            [96.187, 0.4778086916576129, 22.018088003567588],
            [96.047, 0.47306345409173256, 22.02251409164717],
            [96.014, 0.474441770187087, 22.02484377266139],
            [96.006, 0.474446308436007, 22.02907321019635],
            [95.914, 0.47781329454763555, 22.02251409164717],
            [95.901, 0.47781329454763555, 22.02251409164717],
            [95.664, 0.4778117645371522, 22.02251409164717],
            [95.556, 0.4825968737039818, 21.993888066766996],
            [95.506, 0.48018542530586555, 22.00399817486451],
            [95.467, 0.48260626500538184, 21.965847770492875],
            [95.386, 0.4801900741017317, 22.00399817486451],
            [95.356, 0.47542499630270796, 22.02251409164717],
            [95.252, 0.4826047041457636, 21.993888066766996],
            [95.201, 0.48020867018535435, 22.013529284480658],
            [95.112, 0.4840723318835822, 22.020487517413162],
            [94.996, 0.4850642915165041, 21.965847770492875],
            [94.644, 0.4850832670464441, 21.993888066766996],
            [94.205, 0.48511015601702057, 22.018088003567588],
            [93.903, 0.48511647779647704, 22.02251409164717],
            [93.851, 0.48511647779647704, 22.02251409164717],
            [93.695, 0.4875714479434061, 22.013529284480658],
            [93.576, 0.48511647779647704, 22.02251409164717],
        ]
    )
    objective_values_0 = np.array(
        [
            [96.185, 0.46527204747410833, 21.84202971542177],
            [95.06, 0.025297404651095055, 0.4445337884699598],
            [94.015, 0.10447122959211651, 5.590251559033435],
            [95.111, 0.12656420210643968, 7.0507983037884525],
            [95.503, 0.16756392445287757, 9.072154220020474],
            [93.674, 0.09817055567549925, 5.213805647455652],
            [94.165, 0.2698960025187217, 14.657123568578207],
            [96.026, 0.2661217666111463, 14.683762119666662],
            [95.78, 0.24460764276224736, 13.808939142705297],
            [95.528, 0.25106761187008386, 13.769146758365222],
            [95.77, 0.25173675675400126, 13.769146758365222],
            [94.117, 0.1292655155714457, 7.086523305595606],
            [95.533, 0.15043841321888612, 8.370316924059827],
            [94.068, 0.11216063932121705, 6.0274559575788205],
            [94.508, 0.11995436485779142, 6.40479728013885],
            [93.71, 0.13133576680839076, 7.083377717993188],
            [94.84, 0.1421029527401899, 7.701842358830909],
            [91.797, 0.1497806793614929, 8.349221244155144],
            [94.481, 0.17732373875577123, 9.809832652057592],
            [93.765, 0.10475790603009148, 5.644374170935059],
            [94.257, 0.13918019118896618, 7.730812642730298],
            [93.792, 0.15231147852045407, 8.41629573658777],
            [94.539, 0.1289020636979269, 7.099454094009191],
            [95.139, 0.14325523101787893, 7.777279205828466],
            [94.541, 0.2510731036556807, 13.785753702791952],
            [94.948, 0.1758759241711169, 9.802235919220196],
            [95.772, 0.203617641800311, 11.496547117241304],
            [91.836, 0.11104482195470705, 5.934671837983441],
            [94.772, 0.22033501178030027, 12.173601073762534],
            [95.802, 0.4042350282567946, 21.91088502606754],
            [95.559, 0.10942211464956239, 6.0241470263770465],
            [93.823, 0.13925766117184374, 7.670291444209988],
            [95.289, 0.18835717772416138, 10.3090291771292],
            [93.168, 0.19155132658423246, 10.902019290845226],
            [96.215, 0.26171934590949103, 14.683762119666662],
            [95.486, 0.2301531859328762, 12.957272289193446],
            [94.607, 0.1815255554363229, 10.37151660578818],
            [95.59, 0.21981995685434164, 12.213085076354822],
            [95.076, 0.14859740944665376, 8.351100875064109],
            [95.497, 0.1990054743148072, 10.919505363143639],
            [95.134, 0.1766660722613189, 9.874533624764075],
            [95.959, 0.25414324454002185, 13.790064062616294],
            [95.96, 0.18470209101833054, 10.359698090576083],
            [94.799, 0.20493911870295223, 11.496547117241304],
            [95.249, 0.17063872941499342, 9.849708458980722],
            [91.78, 0.11973432431933255, 6.6247531620698155],
            [95.519, 0.32181276758261235, 16.938799842018305],
            [96.158, 0.26172301828491146, 14.675136445662673],
            [96.045, 0.11553010714635874, 6.189663345253998],
            [94.72, 0.32801426893813196, 16.979308360398722],
        ]
    )
    figure_save_path = "./figures/Search_results_16_new.png"
    plot_pareto_front(
        objective_values=objective_values,
        objective_values_0=objective_values_0,
        figure_save_path=figure_save_path,
    )


if __name__ == "__main__":
    main()
    # test_plot_pareto_front()
