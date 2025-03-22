import csv
import os
import sys

import numpy as np

from core.acc_proxy.expressivity_score import ExpressivityScoreEvaluator
from core.acc_proxy.robust_score import compute_exp_error_score
from core.acc_proxy.uniformity_score import compute_uniformity_score_js
from core.optimizer.utils import Converter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# prepare gene features and test accuracies for regression
class RegressorBase:
    # initiate model and genes
    def __init__(self, model, gene_file_path):
        self.genes = []
        self.model = model
        self.ideal_test_acc = []
        self.noisy_test_acc = []

        with open(gene_file_path, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                self.genes.append(row[1])
                self.ideal_test_acc.append((float(row[2]) - 95) / 5)
                self.noisy_test_acc.append((float(row[3]) - 93) / 10)

    # modify the length of gene
    def process_gene(self, gene, required_length):
        dummy_block = [
            np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        ]
        required_extention = required_length - len(gene[1:])

        # extend the gene
        for i in range(required_extention):
            gene.append(dummy_block)
        return gene

    # get the number of parameters of the gene(normalized)
    def get_num_param(self, model, gene):
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

        # normalize num_param
        # the maximum number: all DC array filled with DC devices, only the first PS array after sigma is not counted
        max_num_param = model.super_layer.n_waveguides * (
            model.super_layer.n_blocks - 1
        )
        return num_param / max_num_param

    # get the number of DC/CR device counts(normalized)
    def get_device_counts(self, model, gene):
        """
        Extracts the counts of DC and CR devices from a given genotype.

        Parameters:
        - genotype: A list containing mixed elements where elements of interest are lists with two numpy arrays.

        Returns:
        - Two lists: the first containing counts of DC devices, the second containing counts of CR devices.
        """
        dc_port_candidates = model.super_layer_config["dc_port_candidates"]

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

        cr_counts = []

        dc_counts = count_DC_devices(gene=gene, dc_port_candidates=dc_port_candidates)

        for item in gene:
            if isinstance(item, list) and len(item) == 2:
                # dc_count = count_DC_devices(item[0])
                cr_count = count_CR_devices(item[1])
                # dc_counts.append(dc_count)
                cr_counts.append(cr_count)

        return dc_counts, sum(cr_counts)

    # get the features for one gene
    def get_features(self, model, gene, expressivity_score_evaluator):
        # print(gene)
        num_param = self.get_num_param(gene=gene, model=model)

        num_DC_dict, num_CR = self.get_device_counts(model=model, gene=gene)
        sorted_DC_dict = dict(sorted(num_DC_dict.items()))
        sorted_DC_array = np.array(list(sorted_DC_dict.values()))
        k = model.super_layer_config["n_waveguides"]

        max_num_DC = np.array([k // i for i in sorted_DC_dict.keys()]) * gene[0]
        max_num_CR = k * (k - 1) / 2 * gene[0]

        # normalize number of DC and CR
        num_DC_normalized = sorted_DC_array / max_num_DC
        num_DC_normalized_dict = dict(zip(sorted_DC_dict.keys(), num_DC_normalized))
        num_CR_normalized = num_CR / max_num_CR
        # exit(0)

        converter = Converter(super_layer=model.super_layer)
        solution = converter.gene2solution(gene)
        model.fix_arch_solution(solution)
        # print(model.super_layer)
        robust_score = compute_exp_error_score(
            super_layer=model.super_layer,
            num_samples=200,
            phase_noise_std=0.02,
            dc_noise_std=0.01,
            cr_phase_noise_std=1 * (np.pi / 180),
            cr_tr_noise_std=0.01,
        )

        expressivity_score = expressivity_score_evaluator.compute_expressivity_score(
            model=model
        )

        uniformity_score = compute_uniformity_score_js(model=model)

        features = {
            "gene": gene,
            "num_param": num_param,
            "cr_device_counts": num_CR_normalized,
            "robust_score": robust_score,
            "expressivity_score": expressivity_score,
            "uniformity_score": uniformity_score,
        }

        features.update(num_DC_normalized_dict)

        return features

    # get features for all genes, and save the feature list of genes to a csv file
    def save_features_to_csv(self, output_file_path, checkpoint_path, solution_path):
        self.genes = [gene.replace("array", "np.array") for gene in self.genes]
        first_write = True

        expressivity_score_evaluator = ExpressivityScoreEvaluator(
            checkpoint_path=checkpoint_path, solution_path=solution_path
        )

        with open(output_file_path, mode="w", newline="", encoding="utf-8") as file:
            for gene in self.genes:
                if isinstance(gene, str):
                    gene = eval(gene)
                processed_gene = self.process_gene(
                    gene, required_length=self.model.super_layer_config["n_blocks"]
                )
                feature = self.get_features(
                    gene=processed_gene,
                    model=self.model,
                    expressivity_score_evaluator=expressivity_score_evaluator,
                )
                # print(feature)

                if first_write:
                    writer = csv.DictWriter(file, fieldnames=feature.keys())
                    writer.writeheader()
                    first_write = False

                writer.writerow(feature)
