# 2024.10.29 Yixuan Mei
import os
import sys
from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec


def verify_and_generate():
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name="./config/single24.ini",
        machine_profile_name="./config/machine_profile.ini",
        model_name=ModelName.LLaMa70B,
        workspace_path="./layouts/verify",
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 4, "L4": 8, "T4": 12}
    )

    # setting arguments for ILP layout synthesis
    ilp_args = {
        # pruning
        # keep those parameters the same as the ones you use for solving
        "enable_pruning": False,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1 * mbps,
        # ILP
        # set "use_existing_sol" to True, and locate your ilp_solution.sol in "existing_sol_path"
        # keep other parameters unchanged
        "use_existing_sol": True,
        "allow_partial_inference": False,
        "remove_redundant": True,
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "existing_sol_path": "./layouts/ilp/ilp_solution.sol",
        # heuristic
        # keep those parameters the same as the ones you use for solving
        "start_from_heuristic": False,
        "heuristic_sol_path": "./layouts/petals/petals_sol.ini",
    }

    # run the ILP layout synthesis -> this time, ILPLayout will use the existing solution to
    # generate the simulator_cluster.ini and ilp_sol.ini.
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    """
    What if you press ctrl+c twice, and the program only gets the time to save the ilp_solution.sol?
        ---- Don't worry, you don't need to run the solver from scratch again. We have the following
             function to generate the simulator_cluster.ini and ilp_sol.ini.
    """
    verify_and_generate()
    print("ILP layout synthesis (verification only) finished!")


if __name__ == '__main__':
    main()
