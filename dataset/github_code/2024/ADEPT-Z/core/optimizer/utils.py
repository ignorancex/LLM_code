from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from multiprocessing import Manager, Pool
from typing import Union

import joblib
import numpy as np
import torch
import torch.optim as optim
from pyutils.general import logger as lg

from core.acc_proxy.expressivity_score import ParallelExpressivityScoreEvaluator
from core.acc_proxy.robust_score import compute_exp_error_score
from core.acc_proxy.uniformity_score import compute_uniformity_score_js

__all__ = ["Converter", "Evaluator", "combination_sum"]


def get_num_param(model, gene):
    # count the number of 1 in each DC array
    num_active_block = gene[0]
    count_ones = 0
    for i in range(1, num_active_block):
        for j in gene[i][0]:
            if j == 1:
                count_ones += 1

    # For each 1 in DC array, total #param minus by 1
    # except for the last block
    # for the first block after sigma, the PS array is not counted
    num_param = model.super_layer.n_waveguides * (num_active_block - 1) - count_ones
    max_num_param = model.super_layer.n_waveguides * (model.super_layer.n_blocks - 1)

    return num_param / max_num_param


# extract the number of DC and CR in each block of gene
def extract_device_counts(model, gene):
    """
    Extracts the counts of DC and CR devices from a given genotype.

    Parameters:
    - genotype: A list containing mixed elements where elements of interest are lists with two numpy arrays.

    Returns:
    - Two lists: the first containing counts of DC devices, the second containing counts of CR devices.
    """
    # dc_port_candidates = model.super_layer_config["dc_port_candidates"]
    dc_port_candidates = [2, 3, 4, 6, 8]

    # def count_DC_devices(gene_DC):
    #     # Count elements that are not equal to 1
    #     DC_devices_count = np.sum(gene_DC != 1)
    #     return DC_devices_count
    def count_DC_devices(gene, dc_port_candidates):
        count_dc = {key: 0 for key in dc_port_candidates}
        # dc_port_candidates:[2,3,4,6,8]
        for elem in gene[1:]:
            for num in elem[0]:
                if num in dc_port_candidates:
                    count_dc[num] += 1
        # print(count_dc)
        return count_dc

    def count_CR_devices(gene_CR):  # get the number of crossings
        nums = 0
        n = len(gene_CR)
        gene_cr_copy = gene_CR.copy()
        for i in range(n):
            for j in range(0, n - i - 1):
                if gene_cr_copy[j] > gene_cr_copy[j + 1]:
                    gene_cr_copy[j], gene_cr_copy[j + 1] = (
                        gene_cr_copy[j + 1],
                        gene_cr_copy[j],
                    )
                    nums += 1
        return nums

    # dc_counts = []
    cr_counts = []

    dc_counts = count_DC_devices(gene=gene, dc_port_candidates=dc_port_candidates)

    for item in gene:
        if isinstance(item, list) and len(item) == 2:
            # dc_count = count_DC_devices(item[0])
            cr_count = count_CR_devices(item[1])
            # dc_counts.append(dc_count)
            cr_counts.append(cr_count)

    return dc_counts, sum(cr_counts)


def get_feature(gene, model, expressivity_score_evaluator):
    num_param = get_num_param(gene=gene, model=model)

    num_DC_dict, num_CR = extract_device_counts(model=model, gene=gene)
    sorted_DC_dict = dict(sorted(num_DC_dict.items()))
    sorted_DC_array = np.array(list(sorted_DC_dict.values()))
    k = model.super_layer_config["n_waveguides"]

    max_num_DC = np.array([k // i for i in sorted_DC_dict.keys()]) * gene[0]
    max_num_CR = k * (k - 1) / 2 * gene[0]

    required_length = model.super_layer_config["n_blocks"]

    # normalize number of DC and CR
    num_DC_normalized = sorted_DC_array / max_num_DC
    num_DC_normalized_dict = dict(zip(sorted_DC_dict.keys(), num_DC_normalized))
    num_dc2 = num_DC_normalized_dict[2]
    num_dc3 = num_DC_normalized_dict[3]
    num_dc4 = num_DC_normalized_dict[4]
    num_dc6 = num_DC_normalized_dict[6]
    num_dc8 = num_DC_normalized_dict[8]

    num_CR_normalized = num_CR / max_num_CR

    num_samples = 200
    super_ps_layer = model.super_layer.build_ps_layers(num_samples, 1)
    sigma = torch.randn(
        1,
        num_samples,
        model.super_layer.n_waveguides,
        dtype=torch.cfloat,
        device="cuda:0",
    )
    robust_score = compute_exp_error_score(
        super_layer=model.super_layer,
        super_ps_layer=super_ps_layer,
        sigma=sigma,
        num_samples=num_samples,
        phase_noise_std=0.02,
        dc_noise_std=0.01,
        cr_phase_noise_std=1 * (np.pi / 180),
        cr_tr_noise_std=0.01,
    )

    converter = Converter(super_layer=model.super_layer)
    solution = converter.gene2solution(gene)
    model.fix_arch_solution(solution)

    expressivity_score = expressivity_score_evaluator.compute_expressivity_score(
        model=model, num_samples=200, num_steps=150
    )

    uniformity_score = compute_uniformity_score_js(model=model)

    # features = np.concatenate((np.array([num_param]), np.array([num_dc2]), np.array([num_dc3]), np.array([num_dc4]), \
    #                            np.array([num_dc6]), np.array([num_dc8]), np.array([num_CR_normalized]), \
    #                            np.array([robust_score]), np.array([expressivity_score]), np.array([uniformity_score])), axis=0)
    features = np.concatenate(
        (
            np.array([num_param]),
            np.array([num_CR_normalized]),
            np.array([robust_score]),
            np.array([expressivity_score]),
            np.array([uniformity_score]),
            np.array([num_dc2]),
            np.array([num_dc3]),
            np.array([num_dc4]),
            np.array([num_dc6]),
            np.array([num_dc8]),
        ),
        axis=0,
    )
    return features


def get_feature_all(genes, model, expressivity_score_evaluator):
    solutions = []
    k = model.super_layer_config["n_waveguides"]
    num_samples = 200

    # lists used to save different features
    num_params = []
    num_CRs_normalized = []
    robust_scores = []
    uniformity_scores = []
    num_dc2s, num_dc3s, num_dc4s, num_dc6s, num_dc8s = [], [], [], [], []

    super_ps_layer = model.super_layer.build_ps_layers(num_samples, 1)
    sigma = torch.randn(
        1,
        num_samples,
        model.super_layer.n_waveguides,
        dtype=torch.cfloat,
        device="cuda:0",
    )
    converter = Converter(super_layer=model.super_layer)

    for gene in genes:
        num_params.append(get_num_param(gene=gene, model=model))
        num_DC_dict, num_CR = extract_device_counts(model=model, gene=gene)
        sorted_DC_dict = dict(sorted(num_DC_dict.items()))
        sorted_DC_array = np.array(list(sorted_DC_dict.values()))

        max_num_DC = np.array([k // i for i in sorted_DC_dict.keys()]) * gene[0]
        max_num_CR = k * (k - 1) / 2 * gene[0]

        # normalize number of DC and CR
        num_DC_normalized = sorted_DC_array / max_num_DC
        num_DC_normalized_dict = dict(zip(sorted_DC_dict.keys(), num_DC_normalized))
        num_dc2 = num_DC_normalized_dict[2]
        num_dc3 = num_DC_normalized_dict[3]
        num_dc4 = num_DC_normalized_dict[4]
        num_dc6 = num_DC_normalized_dict[6]
        num_dc8 = num_DC_normalized_dict[8]

        num_dc2s.append(num_dc2)
        num_dc3s.append(num_dc3)
        num_dc4s.append(num_dc4)
        num_dc6s.append(num_dc6)
        num_dc8s.append(num_dc8)

        num_CRs_normalized.append(num_CR / max_num_CR)

        robust_scores.append(
            compute_exp_error_score(
                super_layer=model.super_layer,
                super_ps_layer=super_ps_layer,
                sigma=sigma,
                num_samples=num_samples,
                phase_noise_std=0.02,
                dc_noise_std=0.01,
                cr_phase_noise_std=1 * (np.pi / 180),
                cr_tr_noise_std=0.01,
            )
        )

        solution = converter.gene2solution(gene)
        solutions.append(solution)

        model.fix_arch_solution(solution)
        uniformity_scores.append(compute_uniformity_score_js(model=model))

    lg.info("features except expressivity scores finished calculating")

    expressivity_scores = expressivity_score_evaluator.compute_expressivity_score(
        arch_sols=solutions, num_samples=200, num_steps=150
    )

    features = []
    for (
        num_param,
        expressivity_score,
        num_CR_normalized,
        robust_score,
        uniformity_score,
        num_dc2,
        num_dc3,
        num_dc4,
        num_dc6,
        num_dc8,
    ) in zip(
        num_params,
        expressivity_scores,
        num_CRs_normalized,
        robust_scores,
        uniformity_scores,
        num_dc2s,
        num_dc3s,
        num_dc4s,
        num_dc6s,
        num_dc8s,
    ):
        features.append(
            np.concatenate(
                (
                    np.array([num_param]),
                    np.array([num_CR_normalized]),
                    np.array([robust_score]),
                    np.array([expressivity_score]),
                    np.array([uniformity_score]),
                    np.array([num_dc2]),
                    np.array([num_dc3]),
                    np.array([num_dc4]),
                    np.array([num_dc6]),
                    np.array([num_dc8]),
                ),
                axis=0,
            )
        )
    return features  # [num_populations]


class Converter(object):
    def __init__(self, super_layer):
        self.super_layer = super_layer
        self.arch_sol = OrderedDict()
        for name in self.super_layer.name2layer_map:
            self.arch_sol[name] = None
        self.max_n_blocks = self.super_layer.arch["n_blocks"]

    def solution2gene(self, solution: Union[str, tuple]) -> list:
        # solution to hierarchical gene
        if isinstance(solution, str):
            solution = eval(solution)

        if isinstance(solution, tuple):
            solution = OrderedDict(solution)

        solution = list(solution.values())

        assert len(solution) % 2 == 0, lg.error(
            "Solution length must be a multiple of 2 for paired DC, CR"
        )
        valid_layers = []
        invalid_layers = []
        # put valid layers to the front
        for sol in solution:
            if sol[0]:
                valid_layers.append(sol[1])
            else:
                invalid_layers.append(sol[1])
        arch_gene = [len(valid_layers) // 2]
        solution = valid_layers + invalid_layers
        for dc, cr in zip(solution[::2], solution[1::2]):
            arch_gene.append([np.array(dc), np.array(cr)])

        return arch_gene

    def gene2solution(self, gene: list, to_string: bool = True) -> str:
        """Solution: {"dc_0": [1, np.array([1,1,2,2,1])], "cr_0": [1, np.array([0,3,2,1])], "dc_1": [0, np.array(1,1,1,1,1,1,1)], "cr_1": [0, np.array(0,1,2,3)]}
            # {layer_name: [valid_bit, cfg], ...}
        Args:
            gene (list): _description_
            to_string (bool, optional): _description_. Defaults to True.

        Raises:
            NotImplementedError: _description_

        Returns:
            str: _description_
        """
        new_arch_sol = OrderedDict()
        n_blocks = gene[0]
        n_blocks_V = int(n_blocks) // 2
        max_n_blocks_V = self.max_n_blocks // 2
        n_blocks_V_dummy = max_n_blocks_V - n_blocks_V

        n_blocks_U = n_blocks - n_blocks // 2
        max_n_blocks_U = self.max_n_blocks - max_n_blocks_V
        n_blocks_U_dummy = max_n_blocks_U - n_blocks_U
        gene = gene[1:]
        # reorder gene according to U and V partition. The it will be aligned with arch_sol names
        n_layers_per_blocks = self.super_layer.n_layers_per_block
        gene = (
            gene[:n_blocks_V]
            + gene[n_blocks : n_blocks + n_blocks_V_dummy]
            + gene[n_blocks_V:n_blocks]
            + gene[n_blocks + n_blocks_V_dummy :]
        )
        valid_flags = (
            [1] * n_blocks_V * n_layers_per_blocks
            + [0] * n_blocks_V_dummy * n_layers_per_blocks
            + [1] * n_blocks_U * n_layers_per_blocks
            + [0] * n_blocks_U_dummy * n_layers_per_blocks
        )
        flat_gene = list(chain(*gene))
        assert len(self.arch_sol) == len(flat_gene) == len(valid_flags), lg.error(
            f"Length of arch_sol ({len(self.arch_sol)}) must be equal to flat gene ({len(flat_gene)}) and valid flags ({len(valid_flags)})"
        )
        for block_name, cfg, valid in zip(self.arch_sol, flat_gene, valid_flags):
            if isinstance(cfg, np.ndarray):
                cfg = cfg.tolist()
            new_arch_sol[block_name] = [valid, cfg]

        # convert dict to key-value tuples or string to make it hashable

        new_arch_sol = tuple(new_arch_sol.items())
        if to_string:
            new_arch_sol = str(new_arch_sol)
        return new_arch_sol

    def get_gene_choice(self):
        flattened_search_space = []
        for name, _ in self.arch_sol:
            layer = self.super_layer.name2layer_map[name]
            flattened_search_space.extend(layer.arch_spaces)
        return flattened_search_space


class Evaluator(object):
    def __init__(
        self,
        args,
        acc_predictor,
        cost_predictor,
        robustness_predictor,
        score_mode: str = "compute_density",
        multiobj: bool = False,
        num_procs: int = 1,
    ):
        self.args = args
        self.num_procs = num_procs
        if num_procs > 1:
            mgr = Manager()
            self.solution_lib = mgr.dict()
        else:
            self.solution_lib = {}
        self.acc_predictor = acc_predictor
        self.cost_predictor = cost_predictor
        self.robustness_predictor = robustness_predictor
        self.score_mode = score_mode
        self.multiobj = multiobj

        # prepare two MLP prediction models and scalers, used to predict ideal and noisy test acc
        model_file_path = (
            "./checkpoint/random_forest/MLP_regression_model_noisy_16.joblib"
        )

        self.prediction_model_and_scaler = joblib.load(model_file_path)
        self.prediction_model = self.prediction_model_and_scaler["model"]
        self.prediction_scaler = self.prediction_model_and_scaler["scaler"]

        model_file_path_ideal = (
            "./checkpoint/random_forest/MLP_regression_model_16.joblib"
        )
        self.prediction_model_and_scaler_ideal = joblib.load(model_file_path_ideal)
        self.prediction_model_ideal = self.prediction_model_and_scaler_ideal["model"]
        self.prediction_scaler_ideal = self.prediction_model_and_scaler_ideal["scaler"]

        # prepare the expressivityscore Evaluator
        checkpoint_path = (
            "./checkpoint/mnist/cnn/train_16_MZI/SuperOCNN__acc-98.77_epoch-90.pt"
        )
        solution_path = "./configs/mnist/genes/MZI_solution_16.txt"
        self.expressivity_score_evaluator = ParallelExpressivityScoreEvaluator(
            checkpoint_path=checkpoint_path, solution_path=solution_path
        )

    def evaluate_cost(self, arch_sol: str, keys: list = ["area", "power", "latency"]):
        """return a cost dict {"key1": value, "key2": value,...}

        Args:
            arch_sol (str): _description_
            keys (list, optional): _description_. Defaults to ["area", "power", "latency"].

        Returns:
            _type_: _description_
        """
        if isinstance(arch_sol, str):
            arch_sol = eval(arch_sol)

        cost_dict = self.cost_predictor(arch_sol)
        return cost_dict

    @lru_cache(maxsize=8)
    def evaluate_acc(self, solution):
        acc = self.acc_predictor(solution)
        return acc

    def evaluate_score(self, cost_dict: dict) -> float:
        # treat accuracy proxy, compute density, energy efficiency as objective
        # latency, robustness as constraint
        # accuracy = cost_dict["accuracy"]
        area = cost_dict["area"]
        power = cost_dict["power"]
        latency = cost_dict["latency"]

        k = self.args.matrix_size

        # for matrix multiplication between a kxk weight matrix and a kx1 input vector, the total number of operations is: k*(k+k-1)
        # k outputs, each output needs k multiplications and k-1 additions
        # the time it takes to finish one matrix multiplication is the latency(unit: ps)
        # The total number of operations in one second is: (2k^2-k)/latency TOPS

        ## 12/11: remove normalization of compute density and energy efficiency, no need to do that for NSGA-II
        compute_density = k * (2 * k - 1) / (latency * area * 1e-6)  # unit: TOPS/(mm^2)
        # print(f"Compute Density: {compute_density} TOPS/(mm^2)")
        energy_efficiency = k * (2 * k - 1) / (latency * power * 1e-3)  # unit: TOPS/W
        # print(f"Energy Efficiency: {energy_efficiency} TOPS/(W)")

        if self.multiobj:
            if self.score_mode == "compute_efficiency":
                # score = (accuracy, compute_density)
                score = (compute_density,)
            elif self.score_mode == "energy_efficiency":
                score = (energy_efficiency,)
            elif self.score_mode == "compute_density.energy_efficiency":
                score = (compute_density, energy_efficiency)
            else:
                raise NotImplementedError
        else:
            if self.score_mode == "compute_efficiency":
                # score = accuracy * compute_density
                score = compute_density
            elif self.score_mode == "energy_efficiency":
                # score = accuracy * energy_efficiency
                score = energy_efficiency
            elif self.score_mode == "compute_density.energy_efficiency":
                # score = (accuracy**2) * (compute_density**0.2) * (energy_efficiency**0.2)
                score = (compute_density**0.2) * (energy_efficiency**0.2)
            else:
                raise NotImplementedError

        return score

    def _evaluate_accuracy_all(
        self,
        genes,
        model,
        prediction_model,
        prediction_model_ideal,
        prediction_scaler,
        prediction_scaler_ideal,
        expressivity_score_evaluator,
    ):
        lg.info("Evalution of accuracy for all genes started.")

        features_all = get_feature_all(
            genes=genes,
            model=model,
            expressivity_score_evaluator=expressivity_score_evaluator,
        )

        lg.info("Finish getting features for all genes.")

        accuracy_list = []
        for features in features_all:
            # lg.info(f"Features for ideal prediction: {features}")

            inputs_ideal = prediction_scaler_ideal.transform([features])

            # lg.info(f"Features for ideal prediction: {inputs_ideal}")

            predicted_ideal_acc = float(
                round(prediction_model_ideal.predict(inputs_ideal)[0], 4)
            )

            # lg.info(f"predicted ideal_acc: {predicted_ideal_acc}")

            predicted_ideal_acc_array = np.array([predicted_ideal_acc]) * 5 + 95

            # lg.info(f"predicted ideal_acc_array: {predicted_ideal_acc_array}")

            # lg.info("Prediction of ideal accuracy get.")

            features_noisy = np.concatenate((features, predicted_ideal_acc_array))

            # lg.info(f"Features for noisy prediction: {features_noisy}")

            inputs_noisy = prediction_scaler.transform([features_noisy])

            # lg.info(f"Input_Noisy: {inputs_noisy}")

            accuracy = (
                float(round(prediction_model.predict(inputs_noisy)[0], 4)) * 10 + 93
            )

            # lg.info(f"Noisy Accuracy Prediction: {accuracy}")

            accuracy_list.append(accuracy)
            # lg.info("Prediction of noisy accuracy get.")

        lg.info("Finish getting accuracy list including all genes.")
        return accuracy_list

    def _evaluate(self, gene, solution):
        fingerprint = solution
        if fingerprint in self.solution_lib:
            """circuit has been simulated before"""
            pass
        else:
            # accuracy = self.evaluate_acc(solution)
            # print("accuracy:", accuracy)
            # lg.info("_evaluate function begins")
            cost_keys = self.score_mode.split(".")
            keys = []
            if "compute_density" in cost_keys:
                keys.append("area")

            if "energy_efficiency" in cost_keys:
                keys.append("power")

            keys.append("latency")

            keys.append("robustness")

            cost = self.evaluate_cost(fingerprint, keys=keys)

            tmp_dict = {}  # {"accuracy": accuracy}
            tmp_dict.update(cost)

            if "robustness" in keys:
                robustness = self.robustness_predictor(solution)
                tmp_dict.update({"robustness": robustness})

            self.solution_lib[fingerprint] = tmp_dict
        cost_dict = self.solution_lib[fingerprint]

        # here the score is a tuple of scores, but do not have accuracy results
        score = self.evaluate_score(cost_dict)
        # lg.info("Getting score tuples inside _evaluate funcition")
        return score, solution, gene, cost_dict  # here accuracy is missing

    def evaluate_all(self, model, genes, solutions):
        scores = []
        assert len(genes) == len(solutions)

        tasks = tuple(zip(genes, solutions))

        if self.num_procs > 1:
            pool = Pool(self.num_procs)
            eval_result = pool.starmap(self._evaluate, tasks)
        else:
            eval_result = {}
            accuracy_list = self._evaluate_accuracy_all(
                genes,
                model,
                self.prediction_model,
                self.prediction_model_ideal,
                self.prediction_scaler,
                self.prediction_scaler_ideal,
                self.expressivity_score_evaluator,
            )

            lg.info("_evaluate_accuracy_all function finished.")

            for i, (gene, solution) in enumerate(tasks):
                score, cost_dict = (
                    self._evaluate(gene, solution)[0],
                    self._evaluate(gene, solution)[3],
                )

                ## add accuracy value to the score tuple and the cost_dict
                score = (accuracy_list[i],) + score
                cost_dict["accuracy"] = accuracy_list[i]
                eval_result[solution] = (score, cost_dict)

            scores = [i[0] for i in eval_result.values()]

        return (scores, eval_result)


def combination_sum(candidates, target):
    """
    given candidate integers as a list and a target integer,\
    find all unique combinations that sum to the target using dynamic programming
    """
    # Sort the candidates to help avoid duplicates in the result
    candidates.sort()

    # Initialize a 2D DP table to store combinations for each target value
    dp = [[] for _ in range(target + 1)]

    # For a target of 0, there is one empty combination
    dp[0] = [[]]

    # Loop through the candidates and build up the DP table
    for candidate in candidates:
        for i in range(candidate, target + 1):
            for combo in dp[i - candidate]:
                dp[i].append(combo + [candidate])

    # Return the combinations for the target value
    return dp[target]


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.init_value = death_rate
        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]["lr"]

    def reset(self, new_max_lr=None):
        if new_max_lr is not None:
            self.sgd.param_groups[0]["lr"] = new_max_lr
        self.cosine_stepper.last_epoch = -1


class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.initial_dr = death_rate  # Store initial learning rate
        self.current_dr = death_rate  # Current learning rate
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1
        if self.steps % self.frequency == 0:
            self.current_dr *= self.factor  # Update learning rate

    def get_dr(self):
        return self.current_dr  # Return the current learning rate
