from tqdm import tqdm
import numpy as np
from scipy import stats
from enum import Enum
from dataclasses import dataclass
from typing import List


@dataclass
class Params:
    nn: List[int]
    nsm_k: int


class Algorithm(Enum):
    NeighborhoodStabilityMeasure = 'nsm'

    def __str__(self):
        return self.value

    def scores(self, index, db: np.ndarray, queries: np.ndarray, params: Params):
        if self == self.NeighborhoodStabilityMeasure:
            return self._nsm(index, db, queries, params)

    def correlation(self, gt_scores: np.ndarray, predicted_scores: np.ndarray):
        return stats.spearmanr(gt_scores, predicted_scores)

    def _nsm(self, index, db: np.ndarray, queries: np.ndarray, params: Params):
        nn_max = max(params.nn)
        
        print(f'Searching the index for {nn_max} nearest neighbors for {queries.shape[0]} queries...')
        _, neighborhoods = index.search(queries, nn_max)

        current_nprobe = index.nprobe
        index.nprobe = params.nsm_k * 10

        db_neighbors = np.zeros((db.shape[0], params.nsm_k + 1), dtype=int)
        unique_neighbors = np.unique(neighborhoods.reshape([-1]))
        print(f'Searching for {params.nsm_k} neighbor(s) of {len(unique_neighbors)} points...')
        _, db_neighbors[unique_neighbors] = index.search(db[unique_neighbors], params.nsm_k + 1)

        db_nn = np.zeros((db.shape[0], params.nsm_k), dtype=int)
        for i, db_neighbor in enumerate(db_neighbors):
            j = 0
            for id in db_neighbor:
                if id == i:
                    continue
                db_nn[i, j] = id
                j += 1
                if j >= params.nsm_k:
                    break
        db_neighbors = db_nn

        stabilities = []
        for nn in tqdm(params.nn, disable=len(params.nn) == 1):
            nn_stabilities = []
            for neighborhood in tqdm(neighborhoods, disable=len(params.nn) > 1):
                neighborhood_set = set(neighborhood[:nn].tolist())
                score = np.zeros(len(neighborhood[:nn]), dtype=int)
                for i, neighbors in enumerate(db_neighbors[neighborhood[:nn]]):
                    score[i] = len(set(neighbors) & neighborhood_set)
                nn_stabilities.append(np.sum(score) / float(len(score) * params.nsm_k))
            stabilities.append(nn_stabilities)

        index.nprobe = current_nprobe
        return np.array(stabilities)
