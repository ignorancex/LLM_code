# 2024.04.24 Yixuan Mei
import os.path
from typing import Dict, List, Tuple

from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod
from simulator.event_simulator.request import InferenceRequest, RequestPhase
from simulator.initial_layout.layout_synthesizer import LayoutSynthesizer, LayoutMethod
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import SchedulingMode, KVParameters, SchedulerCore

from llm_sys.utils import to_real


def gen_sys_config(
    host_ip: str,
    type2ips: Dict[str, List[str]],
    # model and machine
    machine_num_dict: Dict[str, int],
    model_name: ModelName,
    # cluster
    complete_cluster_file_name: str,
    machine_profile_file_name: str,
    solution_file_name: str,
    simulator_cluster_file_name: str,
    # output
    output_dir: str,
    output_file_name: str,
) -> None:
    # ----------------------------------- Init ----------------------------------- #
    # load the layout
    layout_synthesizer = LayoutSynthesizer(complete_cluster_file_name=complete_cluster_file_name,
                                           machine_profile_name=machine_profile_file_name,
                                           model_name=model_name,
                                           workspace_path=output_dir,
                                           layout_method=LayoutMethod.LoadExisting,
                                           machine_num_dict=machine_num_dict)
    layout_args = {
        "solution_file_name": solution_file_name,
        "simulator_cluster_file_name": simulator_cluster_file_name
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # load the simulator, here we only use it to initialize the maxflow scheduler
    simulator = ClusterSimulator(model_name=model_name, machine_num_dict=machine_num_dict)
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        "kv_param": KVParameters(expected_kv_hwm=1, expected_output_length_ratio=0.8),
        "scheduling_mode": SchedulingMode.Online,
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()
    layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()
    # ----------------------------------- Init ----------------------------------- #
    # generate system config

    def write_machine(file, machine_id, ip, in_nodes, out_nodes, start_layer, end_layer):
        file.write(f"machine_id: {machine_id}\n")
        file.write(f"ip_address: {ip}\n")
        file.write(f"in_nodes:")
        for cur_in in in_nodes:
            file.write(f" {cur_in}")
        file.write("\n")
        file.write(f"out_nodes:")
        for cur_out in out_nodes:
            file.write(f" {cur_out}")
        file.write("\n")
        file.write(f"start_layer: {start_layer}\n")
        file.write(f"end_layer: {end_layer}\n")
        file.write(f"\n")

    output_file_full_name = os.path.join(output_dir, output_file_name)
    with open(output_file_full_name, "w", newline='\n') as output_file:
        # first we write the host, which must be the first one
        host_in_nodes, host_out_nodes = [], []
        for _, network_link in simulator.sink_node.inbound_links.items():
            host_in_nodes.append(to_real(network_link.node_in.node_uid))
        for _, network_link in simulator.source_node.outbound_links.items():
            host_out_nodes.append(to_real(network_link.node_out.node_uid))
        write_machine(file=output_file, machine_id=0, ip=host_ip, in_nodes=host_in_nodes,
                      out_nodes=host_out_nodes, start_layer=-1, end_layer=-1)

        # here we assert compute nodes are continuous
        num_compute_nodes = len(simulator.compute_nodes)
        for compute_node_uid in range(2, 2 + num_compute_nodes):
            compute_node = simulator.compute_nodes[compute_node_uid]
            compute_in_nodes, compute_out_nodes = [], []
            for _, network_link in compute_node.inbound_links.items():
                compute_in_nodes.append(to_real(network_link.node_in.node_uid))
            for _, network_link in compute_node.outbound_links.items():
                compute_out_nodes.append(to_real(network_link.node_out.node_uid))
            write_machine(file=output_file, machine_id=to_real(compute_node_uid),
                          ip=type2ips[compute_node.machine_type].pop(0),
                          in_nodes=compute_in_nodes, out_nodes=compute_out_nodes,
                          start_layer=min(compute_node.in_vram_model_layers),
                          end_layer=max(compute_node.in_vram_model_layers) + 1)
        for _, ip_list in type2ips.items():
            assert len(ip_list) == 0, "Found redundant ip!"
