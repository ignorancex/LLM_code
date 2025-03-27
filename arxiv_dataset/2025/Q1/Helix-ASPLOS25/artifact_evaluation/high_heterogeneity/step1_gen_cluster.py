# 2024.11.07 Yixuan Mei
import os

from simulator.initial_layout.fake_cluster_generator import FakeClusterGenerator
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def main():
    # initialize a fake cluster generator
    generator = FakeClusterGenerator()
    generator.set_node_statistics(num_compute_nodes=42, avg_degree=41, source_degree=42, sink_degree=42,
                                  node_type_percentage={"A100": 4, "V100": 6, "L4": 8, "L4x2": 4, "T4": 10, "T4x2": 6,
                                                        "T4x4": 4})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)

    # generate the cluster
    os.makedirs("./config", exist_ok=True)
    generator.generator_fake_cluster(file_name="./config/cluster42.ini", seed=0)
    print("Generated cluster configuration file at ./config/cluster42.ini")


if __name__ == '__main__':
    main()
