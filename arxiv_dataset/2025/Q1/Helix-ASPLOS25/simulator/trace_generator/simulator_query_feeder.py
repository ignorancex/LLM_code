# 2024.04.22 Yixuan Mei

from enum import Enum
from typing import List, Tuple, Optional

from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.query_manager import QueryManager
from simulator.event_simulator.request import RequestPhase
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import GlobalFlowScheduler
from simulator.scheduler.swarm.swarm_scheduler import SwarmScheduler
from simulator.scheduler.naive_scheduler import NaiveScheduler
from simulator.scheduler.shortest_queue.shortest_queue_scheduler import ShortestQueueScheduler
from simulator.trace_generator.trace_generator import ArrivalRateSource, Dataset, TraceGenerator
from simulator.trace_generator.length_sampler import LengthSampler


class OnlineRequestFeeder:
    def __init__(self, cluster_token_throughput: float, start_time: float, duration: float, seed: int = 0):
        """
        Online mode: request will arrive based on real arrival rates (with assigned average).

        :param cluster_token_throughput: token throughput of the cluster
        :param start_time: simulation start time
        :param duration: simulation duration
        :param seed: random seed
        """
        # record the parameters
        self.cluster_token_throughput: float = cluster_token_throughput
        self.start_time: float = start_time
        self.duration: float = duration
        self.seed: int = seed

        # initialize the trace generator and generate the trace
        self.trace_generator = TraceGenerator(arrival_rate_source=ArrivalRateSource.AzureConv,
                                              length_dataset=Dataset.AzureConversation,
                                              cluster_token_throughput=cluster_token_throughput, seed=seed)
        self.trace: List[Tuple[float, int, int]] = self.trace_generator.generate_trace(start_time=start_time,
                                                                                       duration=int(duration))

        # calculate real avg arrival rate
        total_tokens = sum([entry[1] + entry[2] for entry in self.trace])
        print(f"Online Request Feeder: avg token feeding throughput {total_tokens / duration}.")

    def auto_simulate(self, simulator: ClusterSimulator, watch_items: Optional[List[str]] = None,
                      watch_interval: Optional[float] = None):
        """
        Run simulation.

        :param simulator: the cluster simulator, it should be fully initialized
        :param watch_items: items to watch during simulation
        :param watch_interval: watch interval
        :return: None
        """
        query_manager: QueryManager = simulator.query_manager
        for idx, request_arrival in enumerate(self.trace):
            arrive_time, input_length, output_length = request_arrival
            query_manager.issue_query(creation_time=arrive_time,
                                      input_seq_length=input_length,
                                      output_seq_length=output_length)
            if idx + 1 < len(self.trace):
                simulator.simulate(until=self.trace[idx + 1][0],  # next arrival time
                                   watch_items=watch_items, watch_interval=watch_interval)
            else:
                simulator.simulate(watch_items=watch_items, watch_interval=watch_interval)
        assert len(query_manager.queries_on_the_fly) == 0, "Found unfinished queries!"
        assert len(query_manager.finished_queries) == len(self.trace), "Some queries missing!"


class OfflineRequestFeeder:
    def __init__(self, initial_query_count: int, start_time: float, duration: float,
                 stop_at_duration: bool, feed_hwm: float, seed: int = 0) -> None:
        """
        Offline mode: initially we will scatter a fixed number of requests. New requests will only
        arrive when the scheduler request so.
        Note: 1. For MaxFlow scheduling, we launch new queries based on kv cache status. (we will launch
                 more if bottleneck <= self.feed_hwm)
              2. For Swarm and Random (Naive) scheduling, since they can not reject a request when there are
                 too many, we can only set a manual max query count (i.e. initial_query_count).

        :param initial_query_count: initial query count
        :param start_time: simulation start time
        :param duration: simulation duration
        :param stop_at_duration: whether we should stop at duration
        :param feed_hwm: high watermark for feeding new queries
        :param seed: random seed
        :return None
        """
        # save parameters
        self.initial_query_count: int = initial_query_count
        self.start_time: float = start_time
        self.duration: float = duration
        self.stop_at_duration: bool = stop_at_duration
        self.feed_hwm: float = feed_hwm
        self.seed: int = seed

        # since we are in offline mode, we will not generate the trace here
        # we only need a length sampler to generate the initial queries
        self.length_sampler = LengthSampler(dataset=Dataset.AzureConversation, seed=seed)

        # simulator
        self.simulator: Optional[ClusterSimulator] = None

    def auto_simulate(self, simulator: ClusterSimulator, watch_items: Optional[List[str]] = None,
                      watch_interval: Optional[float] = None) -> None:
        """
        Run simulation.

        :param simulator: the cluster simulator, it should be fully initialized
        :param watch_items: items to watch during simulation
        :param watch_interval: watch interval
        :return: None
        """
        query_manager: QueryManager = simulator.query_manager

        # register offline mode
        simulator.register_offline_query_feeder(offline_query_feeder=self)
        self.simulator = simulator

        # first launch the initial queries
        for i in range(self.initial_query_count):
            input_length, output_length = self.length_sampler.sample_length()
            query_manager.issue_query(creation_time=self.start_time + i * 0.1,
                                      input_seq_length=input_length, output_seq_length=output_length)

        # simulate
        if self.stop_at_duration:
            simulator.simulate(until=self.start_time + self.duration,
                               watch_items=watch_items, watch_interval=watch_interval)
        else:
            simulator.simulate(watch_items=watch_items, watch_interval=watch_interval)

    def check_launch_new_query(self, finished_request_type: RequestPhase) -> None:
        """
        Launch a new query when the simulator satisfies the condition.
        Note: this function will be called under two cases:
              1. a query finished prompt phase
              2. a query finished last decode

        :return: None
        """
        # if current time is large than start + duration, just return
        if self.simulator.current_time > self.start_time + self.duration:
            return

        # check whether we can launch more requests into the simulator
        scheduler = self.simulator.scheduler
        if isinstance(scheduler, GlobalFlowScheduler):
            # check kv cache expectation's bottleneck
            # if cluster has space, then we just launch more
            bottleneck = scheduler.core.kv_expectation.bottleneck_usage()
            launch_more = (bottleneck <= self.feed_hwm)
        elif (isinstance(scheduler, SwarmScheduler) or isinstance(scheduler, NaiveScheduler)
              or isinstance(scheduler, ShortestQueueScheduler)):
            if finished_request_type == RequestPhase.Initialization:
                # a query finished its prompt phase, in order to keep the total number of on the fly
                # queries unchanged, we can not launch more.
                launch_more = False
            elif finished_request_type == RequestPhase.Increment:
                # a query finished, can launch another one to replace it
                launch_more = True
            else:
                assert False, "Unknown query type!"
        else:
            assert False, "Unknown scheduler type!"

        # launch the new query
        if launch_more:
            input_length, output_length = self.length_sampler.sample_length()
            self.simulator.query_manager.issue_query(
                creation_time=self.simulator.current_time + 0.1,
                input_seq_length=input_length, output_seq_length=output_length
            )
