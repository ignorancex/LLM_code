import random

import torch
import numpy as np
import wandb

from utils.metrics import AverageMeterSet


class ValidationMetricsCalculator:
    def __init__(self, similarity_matrix: torch.tensor, attribute_matching_matrix: torch.tensor,
                 ref_attribute_matching_matrix: torch.tensor, top_k: tuple):
        self.top_k = top_k
        self.attribute_matching_matrix = attribute_matching_matrix
        self.ref_attribute_matching_matrix = ref_attribute_matching_matrix
        self.num_query_features = similarity_matrix.size(0)
        self.num_test_features = similarity_matrix.size(0)
        self.similarity_matrix = similarity_matrix
        self.top_scores = torch.zeros(self.num_query_features, max(top_k))
        self.most_similar_idx = torch.zeros(self.num_query_features, max(top_k))
        self.recall_results = {}
        self.recall_positive_queries_idxs = {k: [] for k in top_k}
        self.top_scores_calculated = False

    def __call__(self):
        # Filter query_feat == target_feat
        assert self.similarity_matrix.shape == self.ref_attribute_matching_matrix.shape

        self.similarity_matrix[(self.ref_attribute_matching_matrix == True)] = self.similarity_matrix.min()
        return self._calculate_recall_at_k()

    def _calculate_recall_at_k(self) -> dict:
        """
        :return: one metric for each k in top_k use recall_@k as key, average of multi-k as avg_recall_@{k1},{k2}...,{kn}
        """
        average_meter_set = AverageMeterSet()
        self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))
        self.top_scores_calculated = True
        # calculate in numpy
        topk_attribute_matching = np.array(np.take_along_axis(self.attribute_matching_matrix.numpy(), self.most_similar_idx.numpy(),
                                                     axis=1))
        for k in self.top_k:
            query_matched_vector = topk_attribute_matching[:, :k].sum(axis=1).astype(bool)
            self.recall_positive_queries_idxs[k] = list(np.where(query_matched_vector > 0)[0])
            num_correct = query_matched_vector.sum()
            num_samples = len(query_matched_vector)
            average_meter_set.update('recall_@{}'.format(k), num_correct, n=num_samples)
        recall_results = average_meter_set.averages()
        return recall_results

    @staticmethod
    def _multiple_index_from_attribute_list(attribute_list, indices):
        attributes = []
        for idx in indices:
            attributes.append(attribute_list[idx.item()])
        return attributes



