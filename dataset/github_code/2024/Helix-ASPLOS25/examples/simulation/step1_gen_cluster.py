# 2024.10.28 Yixuan Mei
from simulator.initial_layout.fake_cluster_generator import FakeClusterGenerator, PartitionedClusterGenerator
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def generate_single(file_name: str):
    """
    This function shows how to generate a cluster configuration file to represent a cluster of
    machines located in a single region.
    """
    generator = FakeClusterGenerator()

    # set the statistics of the cluster
    generator.set_node_statistics(num_compute_nodes=24, avg_degree=23, source_degree=24, sink_degree=24,
                                  node_type_percentage={"A100": 4, "T4": 12, "L4": 8})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)

    # generate the cluster
    generator.generator_fake_cluster(file_name=file_name, seed=0)


def generate_partitioned(file_name: str):
    """
    This function shows how to generate a cluster configuration file to represent a partitioned cluster.
    """
    generator = PartitionedClusterGenerator()

    # set the statistics of the cluster
    generator.add_partition(nodes_list=["A100"] * 4)
    generator.add_partition(nodes_list=["L4"] * 2 + ["T4"] * 8)
    generator.add_partition(nodes_list=["L4"] * 6 + ["T4"] * 4)
    generator.set_network_statistics(
        in_partition_avg_bandwidth=1.25 * gbps, in_partition_var_bandwidth=125 * mbps,
        in_partition_avg_latency=1 * MilliSec, in_partition_var_latency=0,
        cross_partition_avg_bandwidth=12.5 * mbps, cross_partition_var_bandwidth=2.5 * mbps,
        cross_partition_avg_latency=50 * MilliSec, cross_partition_var_latency=10 * MilliSec
    )

    # generate the cluster
    generator.generator_fake_cluster(file_name=file_name, seed=0, create_separate=False)


def main():
    """
    We provide two automatic ways to generate the cluster configuration file. Refer to FakeClusterGenerator
    and PartitionedClusterGenerator for more details. If you have a specific cluster configuration, you can
    also write your own script to generate the cluster configuration file.

    Note: currently, the simulator only supports machines with {"A100", "V100", "L4", "L4x2", "T4", "T4x2",
    "T4x4"} GPUs. You can add more machines by profiling them and add the data to simulator/model_manager.
    """
    generate_single(file_name="./config/single24.ini")
    print("Single cluster configuration file is generated to ./config/single24.ini")
    generate_partitioned(file_name="./config/3cluster24.ini")
    print("Partitioned cluster configuration file is generated to ./config/3cluster24.ini")


if __name__ == '__main__':
    main()
