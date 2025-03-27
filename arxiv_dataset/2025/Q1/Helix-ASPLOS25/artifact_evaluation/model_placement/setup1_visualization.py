# 2024.11.06 Yixuan Mei
import os
import sys
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer
from simulator.event_simulator.cluster_simulator import ClusterSimulator, ModelName, SchedulingMethod, RequestPhase
from simulator.trace_generator.simulator_query_feeder import OfflineRequestFeeder
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import KVParameters, SchedulingMode

def simulate_maxflow_offline(
        title: str,
        solution_file_name: str,
        simulator_cluster_file_name: str,
):
    # load cluster
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config_single/cluster24.ini",
        machine_profile_name="./config_single/machine_profiles.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./tmp",
        layout_method=LayoutMethod.LoadExisting,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )
    layout_args = {
        "solution_file_name": solution_file_name,
        "simulator_cluster_file_name": simulator_cluster_file_name,
    }
    cluster_file_path = layout_synthesizer.synthesize(args=layout_args)

    # initialize the simulator
    simulator = ClusterSimulator(model_name=ModelName.LLaMa70B, machine_num_dict={"A100": 4, "L4": 8, "T4": 12})
    simulator.from_ini_file(config_file_name=cluster_file_path)
    scheduler_args = {
        # offline
        "kv_param": KVParameters(expected_kv_hwm=0.85, expected_output_length_ratio=1),
        "scheduling_mode": SchedulingMode.Offline,
    }
    simulator.init_scheduler(scheduling_method=SchedulingMethod.MaxFlow, args=scheduler_args)
    simulator.init_query_manager()
    simulator.mark_as_ready()

    # load the models into the simulator and update scheduler
    layout_synthesizer.set_layout(simulator=simulator)
    simulator.update_scheduler()

    # visualize
    simulator.visualize_cluster(title=title, save_path="./visualization", show_fig=False)
    print(f"Model placement visualization is saved at ./visualization/{title}.jpg.")


def main():
    # parse arguments
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <helix/petals/swarm>"
    method = sys.argv[1]
    assert method in ["helix", "petals", "swarm"], f"Invalid method: {method}"
    os.makedirs("./visualization", exist_ok=True)

    # run simulation
    if method == "helix":
        simulate_maxflow_offline(
            title="helix",
            solution_file_name="./layout_single/ilp/ilp_sol.ini",
            simulator_cluster_file_name="./layout_single/ilp/simulator_cluster.ini"
        )

    elif method == "petals":
        simulate_maxflow_offline(
            title="petals",
            solution_file_name="./layout_single/petals/petals_sol.ini",
            simulator_cluster_file_name="./layout_single/petals/simulator_cluster.ini"
        )

    elif method == "swarm":
        simulate_maxflow_offline(
            title="swarm",
            solution_file_name="./layout_single/swarm/swarm_sol.ini",
            simulator_cluster_file_name="./layout_single/swarm/simulator_cluster.ini"
        )


if __name__ == '__main__':
    main()
