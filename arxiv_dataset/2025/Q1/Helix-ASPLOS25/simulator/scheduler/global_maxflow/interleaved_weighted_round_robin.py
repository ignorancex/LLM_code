# 2023.01.25 Yixuan Mei

import math

from typing import List


class IWRR:
    def __init__(self, capacities: List[float], initial_loads: List[float]) -> None:
        """
        Interleaved weighted round-robin.

        :param capacities: capacity of each candidate
        :param initial_loads: initial load of each candidate
        """
        assert len(capacities) == len(initial_loads), "Shape mismatch in IWRR"
        self.capacities: List[float] = capacities
        self.loads: List[float] = initial_loads

    def update_loads(self, workload: float, index: int) -> None:
        """
        Add workload / capacity to the load of candidate with given index. This function is called
        when for some reason (e.g., follow history path) a candidate is required to execute some workload.

        :param workload: amount of workload
        :param index: index of the candidate to update
        :return: None
        """
        assert index < len(self.capacities) and not self.capacities[index] == 0, "Can not update loads!"
        self.loads[index] += workload / self.capacities[index]

    def choose_one(self, workload: float, mask: List[bool] or None) -> int:
        """
        Interleaved weighted round-robin: choose one candidate.

        :param workload: the amount of workload
        :param mask: whether each candidate may be considered
        :return: index of the selected candidate
        """
        # generate the mask and check shape
        if mask is None:
            mask = [True for _ in range(len(self.capacities))]
        assert len(mask) == len(self.capacities), "Shape mismatch in IWRR"
        assert any(mask), "No candidate can be selected because of all False mask!"

        # IWRR for choosing the candidate
        best_candidate_idx: int = -1
        best_load_after: float = math.inf
        for cur_idx, (cur_capacity, cur_load, cur_mask) in enumerate(zip(self.capacities, self.loads, mask)):
            assert not (cur_mask and cur_capacity == 0), "Zero capacity candidate not masked out!"
            if cur_mask and cur_load + workload / cur_capacity < best_load_after:
                best_candidate_idx = cur_idx
                best_load_after = cur_load + workload / cur_capacity

        # apply load and return
        self.loads[best_candidate_idx] += workload / self.capacities[best_candidate_idx]
        return best_candidate_idx

    def restore_one(self, workload: float, index: int) -> None:
        """
        Restore the change caused by the workload. Used when rejecting a request in offline mode.

        :param workload: the amount of workload
        :param index: index of the candidate to update
        :return: None
        """
        assert index < len(self.capacities) and not self.capacities[index] == 0, "Can not restore loads!"
        self.loads[index] -= workload / self.capacities[index]
