# 2024.04.22 Yixuan Mei

from simulator.initial_layout.fake_cluster_generator import FakeClusterGenerator
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def generate_cluster24():
    generator = FakeClusterGenerator()
    generator.set_node_statistics(num_compute_nodes=24, avg_degree=23, source_degree=24, sink_degree=24,
                                  node_type_percentage={"A100": 4, "T4": 12, "L4": 8})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)
    generator.generator_fake_cluster(file_name="./config/cluster24.ini", seed=0)


def generate_a100():
    generator = FakeClusterGenerator()
    generator.set_node_statistics(num_compute_nodes=4, avg_degree=3, source_degree=4, sink_degree=4,
                                  node_type_percentage={"A100": 4})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)
    generator.generator_fake_cluster(file_name="./config/a100.ini", seed=0)


def generate_l4():
    generator = FakeClusterGenerator()
    generator.set_node_statistics(num_compute_nodes=8, avg_degree=7, source_degree=8, sink_degree=8,
                                  node_type_percentage={"L4": 8})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)
    generator.generator_fake_cluster(file_name="./config/l4.ini", seed=0)


def generate_t4():
    generator = FakeClusterGenerator()
    generator.set_node_statistics(num_compute_nodes=12, avg_degree=11, source_degree=12, sink_degree=12,
                                  node_type_percentage={"T4": 12})
    generator.set_link_statistics(avg_bandwidth=1 * gbps, var_bandwidth=0,
                                  avg_latency=1 * MilliSec, var_latency=0,
                                  fill_with_slow_link=True,
                                  slow_link_avg_bandwidth=1 * gbps, slow_link_var_bandwidth=0,
                                  slow_link_avg_latency=1 * MilliSec, slow_link_var_latency=0)
    generator.generator_fake_cluster(file_name="./config/t4.ini", seed=0)


def main():
    generate_cluster24()
    print("Generated cluster24.ini")
    generate_a100()
    print("Generated a100.ini")
    generate_l4()
    print("Generated l4.ini")
    generate_t4()
    print("Generated t4.ini")


if __name__ == '__main__':
    main()
