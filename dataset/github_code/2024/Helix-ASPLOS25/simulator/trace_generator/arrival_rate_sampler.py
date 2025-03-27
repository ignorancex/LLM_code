# 2024.03.25 Yixuan Mei

import random
import pickle

from enum import Enum
from typing import List
from pathlib import Path


class ArrivalRateSource(Enum):
    # Azure code: full of spikes
    AzureCode = "ArrivalRateSource.AzureCode"
    # Azure conversation: very smooth
    AzureConv = "ArrivalRateSource.AzureConv"


class ArrivalRateSampler:
    def __init__(self, arrival_rate_source: ArrivalRateSource, target_avg_request_throughput: float, seed: int) -> None:
        """
        Sample the arrival rate from the selected source (rescaled to target average request throughput).

        :param arrival_rate_source: source of arrival rate
        :param target_avg_request_throughput: target average request throughput (per second)
        :param seed: random seed
        :return: None
        """
        # parameters
        self.arrival_rate_source: ArrivalRateSource = arrival_rate_source
        self.target_avg_request_throughput: float = target_avg_request_throughput
        self.seed: int = seed
        random.seed(seed)

        # load the arrival rate data
        cur_abs_path = Path(__file__).parent.absolute()
        if self.arrival_rate_source == ArrivalRateSource.AzureCode:
            with open(cur_abs_path / "arrival_rate/azure_code_arrive_time.pkl", "rb") as file:
                arrival_rate_list: List[float] = pickle.load(file)
        elif self.arrival_rate_source == ArrivalRateSource.AzureConv:
            with open(cur_abs_path / "arrival_rate/azure_conv_arrive_time.pkl", "rb") as file:
                arrival_rate_list: List[float] = pickle.load(file)
        assert len(arrival_rate_list) == 1200, "Arrival rate list length should be 1200 (i.e. 3s interval)!"

        # rescale the arrival rate (here each interval is 3s)
        original_avg_request_throughput: float = sum(arrival_rate_list) / len(arrival_rate_list)
        rescale_factor: float = 3 * target_avg_request_throughput / original_avg_request_throughput
        self.arrival_rate_list: List[float] = [rescale_factor * rate for rate in arrival_rate_list]

    def sample_arrival_rate(self) -> float:
        """
        Sample the arrival rate. (Should be used as a 3s interval.)

        :return: arrival rate
        """
        index: int = random.randint(0, len(self.arrival_rate_list) - 1)
        return self.arrival_rate_list[index]
