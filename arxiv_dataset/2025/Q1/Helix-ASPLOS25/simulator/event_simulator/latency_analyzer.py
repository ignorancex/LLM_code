# 2024.03.22 Yixuan Mei

import os
import matplotlib.pyplot as plt

from typing import Dict, Tuple, TYPE_CHECKING

from simulator.event_simulator.utils import ATOL
from simulator.event_simulator.request import InferenceRequest, RequestLocation, RequestPhase

if TYPE_CHECKING:
    from simulator.event_simulator.query_manager import Query


class RequestLatencyEntry:
    def __init__(self, total: float, compute: float, network: float, request: InferenceRequest) -> None:
        """
        Detailed info of current request's latency

        :param total: total amount of time used on this request
        :param compute: time spent on nodes (compute + queueing)
        :param network: time spent on network
        :param request: a reference to the request
        :return: None
        """
        self.total: float = total
        self.compute: float = compute
        self.network: float = network
        self.request: InferenceRequest = request


class QueryLatencyEntry:
    def __init__(self, total: float, compute: float, network: float, query: "Query") -> None:
        """
        Detailed info of current request's latency

        :param total: total amount of time used on this request
        :param compute: time spent on nodes (compute + queueing)
        :param network: time spent on network
        :param query: a reference to the query
        :return: None
        """
        self.total: float = total
        self.compute: float = compute
        self.network: float = network
        self.query: "Query" = query


class LatencyAnalyzer:
    def __init__(self) -> None:
        """
        Latency analyzer, which stores the latency of all finished requests and queries

        :return: None
        """
        self.request_latency: Dict[int, RequestLatencyEntry] = {}
        self.query_latency: Dict[int, QueryLatencyEntry] = {}

    def add_request(self, request: InferenceRequest) -> None:
        """
        Add a finished request into the latency analyzer

        :param request: the request to add
        :return: None
        """
        # check that request is finished
        assert f"{RequestLocation.SourceNode}" in request.location_history[0][0], "Invalid location history!"
        assert f"{RequestLocation.SinkNode}" in request.location_history[-1][0], "Request not finished!"

        # calculate time
        total_time = request.location_history[-1][1] - request.location_history[0][1]
        compute_time, network_time = 0, 0
        for i in range(1, len(request.location_history)):
            location, arrival_time = request.location_history[i]
            prev_location, prev_arrival_time = request.location_history[i - 1]
            if f"{RequestLocation.Link}" in location:
                assert f"{RequestLocation.SourceNode}" or f"{RequestLocation.ComputeNode}" in prev_location, \
                    "Invalid location history!"
                compute_time += arrival_time - prev_arrival_time
            elif f"{RequestLocation.ComputeNode}" in location or f"{RequestLocation.SinkNode}" in location:
                assert f"{RequestLocation.Link}" in prev_location, "Invalid location history!"
                network_time += arrival_time - prev_arrival_time
            else:
                assert False, "Invalid location history!"
        assert abs(total_time - compute_time - network_time) < ATOL, "Time mismatch!"

        # store
        self.request_latency[request.request_uid] = RequestLatencyEntry(
            total=total_time, compute=compute_time, network=network_time, request=request
        )

    def add_query(self, query: "Query") -> None:
        """
        Add a finished query into the latency analyzer

        :param query: the query to add
        :return: None
        """
        total_time, compute_time, network_time = 0, 0, 0
        for inference_history in query.inference_history:
            cur_request_entry = self.request_latency[inference_history.request_uid]
            total_time += cur_request_entry.total
            compute_time += cur_request_entry.compute
            network_time += cur_request_entry.network
        self.query_latency[query.query_uid] = QueryLatencyEntry(
            total=total_time, compute=compute_time, network=network_time, query=query
        )

    def visualize_request_latency(self, ignore_initialize: bool, save_file_path: str or None = None) \
            -> Tuple[float, float, float]:
        """
        Analyze latency.

        :param ignore_initialize: whether to ignore the initial requests
        :param save_file_path: path to save the plot
        :return: average total time, average compute time, average network time
        """
        total_time_list, compute_time_list, network_time_list = [], [], []
        for request_uid, request_latency_entry in self.request_latency.items():
            if ignore_initialize and request_latency_entry.request.phase == RequestPhase.Initialization:
                continue
            total_time_list.append(request_latency_entry.total)
            compute_time_list.append(request_latency_entry.compute)
            network_time_list.append(request_latency_entry.network)
        avg_total_time = sum(total_time_list) / len(total_time_list)
        avg_compute_time = sum(compute_time_list) / len(compute_time_list)
        avg_network_time = sum(network_time_list) / len(network_time_list)

        # get 99 percentile of total time, compute time, network time
        total_time_list.sort()
        compute_time_list.sort()
        network_time_list.sort()
        percentile_99_total_time = total_time_list[int(len(total_time_list) * 0.99)]
        percentile_99_compute_time = compute_time_list[int(len(compute_time_list) * 0.99)]
        percentile_99_network_time = network_time_list[int(len(network_time_list) * 0.99)]

        # plot a distribution of total time, compute time, network time (in three sub-figures)
        fig, ax = plt.subplots(3, 1, figsize=(12, 12))
        plt.rcParams.update({'font.size': 12})
        # total time
        ax[0].hist(total_time_list, bins=50, color="#98FB98", alpha=0.7)
        ax[0].set_title("Total Time Distribution")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Frequency")
        ax[0].grid(True)
        ax[0].axvline(x=avg_total_time, color="g", linestyle="--", linewidth=5,
                      label=f"Average: {avg_total_time:.2f}")
        ax[0].axvline(x=percentile_99_total_time, color="r", linestyle="--", linewidth=5,
                      label=f"99 Percentile: {percentile_99_total_time:.2f}")
        ax[0].legend()
        # compute time
        ax[1].hist(compute_time_list, bins=50, color="#B0E0E6", alpha=0.7)
        ax[1].set_title("Compute Time Distribution")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Frequency")
        ax[1].grid(True)
        ax[1].axvline(x=avg_compute_time, color="g", linestyle="--", linewidth=5,
                      label=f"Average: {avg_compute_time:.2f}")
        ax[1].axvline(x=percentile_99_compute_time, color="r", linestyle="--", linewidth=5,
                      label=f"99 Percentile: {percentile_99_compute_time:.2f}")
        ax[1].legend()
        # network time
        ax[2].hist(network_time_list, bins=50, color="#F88379", alpha=0.7)
        ax[2].set_title("Network Time Distribution")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("Frequency")
        ax[2].grid(True)
        ax[2].axvline(x=avg_network_time, color="g", linestyle="--", linewidth=5,
                      label=f"Average: {avg_network_time:.2f}")
        ax[2].axvline(x=percentile_99_network_time, color="r", linestyle="--", linewidth=5,
                      label=f"99 Percentile: {percentile_99_network_time:.2f}")
        ax[2].legend()
        plt.tight_layout()
        if save_file_path is not None:
            plt.savefig(save_file_path)
        plt.show()

        return avg_total_time, avg_compute_time, avg_network_time
