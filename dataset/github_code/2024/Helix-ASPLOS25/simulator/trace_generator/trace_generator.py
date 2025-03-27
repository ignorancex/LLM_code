# 2024.03.25 Yixuan Mei

import random

from typing import List, Tuple

from simulator.trace_generator.arrival_rate_sampler import ArrivalRateSampler, ArrivalRateSource
from simulator.trace_generator.length_sampler import LengthSampler, Dataset


class TraceGenerator:
    def __init__(self, arrival_rate_source: ArrivalRateSource, length_dataset: Dataset,
                 cluster_token_throughput: float, seed: int) -> None:
        """
        Generate traces based on realistic datasets.

        :param arrival_rate_source: source of arrival rate
        :param length_dataset: dataset for input and output length
        :param cluster_token_throughput: average token throughput of the cluster
        :param seed: random seed
        """
        # save parameters
        self.arrival_rate_source: ArrivalRateSource = arrival_rate_source
        self.length_dataset: Dataset = length_dataset
        self.cluster_token_throughput: float = cluster_token_throughput
        self.seed: int = seed
        random.seed(seed)

        # initialize samplers
        self.length_sampler = LengthSampler(dataset=length_dataset, seed=seed)
        avg_token_length: float = self.length_sampler.get_average_length()
        ideal_request_throughput: float = cluster_token_throughput / avg_token_length
        self.arrival_rate_sampler = ArrivalRateSampler(arrival_rate_source=arrival_rate_source,
                                                       target_avg_request_throughput=ideal_request_throughput,
                                                       seed=seed)

    def generate_trace(self, start_time: float, duration: int) -> List[Tuple[float, int, int]]:
        """
        Generate a trace.

        :param start_time: start time (s)
        :param duration: duration (s), must be a multiple of 3
        :return: query arrive time, input length, output length
        """
        assert duration % 3 == 0, "Duration must be a multiple of 3!"

        # generate trace
        trace: List[Tuple[float, int, int]] = []
        prev_interval_diff = 0
        for interval_idx in range(duration // 3):
            # sample arrival rate and arrive time
            raw_arrival_rate = self.arrival_rate_sampler.sample_arrival_rate()
            arrival_rate: int = round(raw_arrival_rate + prev_interval_diff)
            prev_interval_diff = raw_arrival_rate + prev_interval_diff - arrival_rate
            assert -1 < prev_interval_diff < 1, "Diff too large!"

            # get arrive time
            arrive_time_list: List[float] = [start_time + interval_idx * 3 + 3 * random.random() for _ in
                                             range(arrival_rate)]
            arrive_time_list.sort()

            for _, arrive_time in zip(range(arrival_rate), arrive_time_list):
                input_length, output_length = self.length_sampler.sample_length()
                trace.append((arrive_time, input_length, output_length))

        # check and return
        assert not len(trace) == 0, "Empty trace!"
        return trace
