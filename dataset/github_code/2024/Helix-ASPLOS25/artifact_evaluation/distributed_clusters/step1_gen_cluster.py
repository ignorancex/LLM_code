import os
from simulator.initial_layout.fake_cluster_generator import PartitionedClusterGenerator
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def main():
    generator = PartitionedClusterGenerator()
    generator.add_partition(nodes_list=["A100"] * 4)
    generator.add_partition(nodes_list=["L4"] * 2 + ["T4"] * 8)
    generator.add_partition(nodes_list=["L4"] * 6 + ["T4"] * 4)
    generator.set_network_statistics(
        in_partition_avg_bandwidth=1.25 * gbps, in_partition_var_bandwidth=125 * mbps,
        in_partition_avg_latency=1 * MilliSec, in_partition_var_latency=0,
        cross_partition_avg_bandwidth=12.5 * mbps, cross_partition_var_bandwidth=2.5 * mbps,
        cross_partition_avg_latency=50 * MilliSec, cross_partition_var_latency=10 * MilliSec
    )
    os.makedirs("./config", exist_ok=True)
    generator.generator_fake_cluster(file_name="./config/3cluster24.ini", seed=0,
                                     create_separate=True, separate_path="./config")
    print("Generated a cluster in 3 regions with 24 nodes!")
    print("Check the config file at ./config/3cluster24.ini")


if __name__ == '__main__':
    main()
