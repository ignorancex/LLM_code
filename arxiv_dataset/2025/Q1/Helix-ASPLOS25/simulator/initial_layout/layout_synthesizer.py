# 2023.03.11 Yixuan Mei

import os

from enum import Enum
from datetime import datetime
from typing import List, Dict, Tuple, Any

from simulator.initial_layout.fake_cluster_generator import prune_cluster
from simulator.initial_layout.ilp_layout.ilp_layout import ILPLayout
from simulator.initial_layout.homogeneous_layout.homogeneous_layout import HomogeneousLayout
from simulator.initial_layout.heterogeneous_layout.swarm_layout import SwarmLayout
from simulator.initial_layout.heterogeneous_layout.petals_layout import PetalsLayout
from simulator.initial_layout.load_existing_layout import LoadExistingLayout
from simulator.model_manager.model_manager import ModelManager, ModelName
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.query_manager import QueryManagerParameters
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import FlowParameters


class LayoutMethod(Enum):
    ILP = "LayoutMethod.ILP"
    Homogeneous = "LayoutMethod.Homogeneous"
    Swarm = "LayoutMethod.Heterogeneous.Swarm"
    Petals = "LayoutMethod.Heterogeneous.Petals"
    LoadExisting = "LayoutMethod.LoadExisting"


class LayoutSynthesizer:
    def __init__(self, complete_cluster_file_name: str, machine_profile_name: str, model_name: ModelName,
                 workspace_path: str, layout_method: LayoutMethod, machine_num_dict: Dict[str, int]) -> None:
        """
        Synthesize initial model layout for the cluster.

        :param complete_cluster_file_name: name of the complete cluster file
        :param machine_profile_name: name of the machine profile file
        :param model_name: name of the model to work with
        :param workspace_path: path to the workspace (all the files will be saved in this directory)
        :param layout_method: layout method
        :param machine_num_dict: {machine_name -> num of machine}
        :return: None
        """
        # paths
        self.complete_cluster_file_name: str = complete_cluster_file_name
        self.machine_profile_name: str = machine_profile_name
        self.workspace_path: str = workspace_path

        # model name and statistics
        self.model_name: ModelName = model_name
        self.model_manager: ModelManager = ModelManager(model_name=model_name, machine_num_dict=machine_num_dict)

        # layout method
        self.layout_method: LayoutMethod = layout_method
        if self.layout_method == LayoutMethod.ILP:
            self.layout_synthesizer = ILPLayout(model_manager=self.model_manager)
        elif self.layout_method == LayoutMethod.Homogeneous:
            self.layout_synthesizer = HomogeneousLayout(model_manager=self.model_manager)
        elif self.layout_method == LayoutMethod.Swarm:
            self.layout_synthesizer = SwarmLayout(model_manager=self.model_manager)
        elif self.layout_method == LayoutMethod.Petals:
            self.layout_synthesizer = PetalsLayout(model_manager=self.model_manager)
        elif self.layout_method == LayoutMethod.LoadExisting:
            self.layout_synthesizer = LoadExistingLayout(model_manager=self.model_manager)
        else:
            assert False, "Unknown layout method!"

        # make sure workspace path exists and is empty
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)

    def synthesize(self, args: Dict[str, Any]) -> str:
        """
        Synthesize the initial layout. Below is contents of args for each layout method.
        When layout_method == LayoutMethod.ILP:
            seed: [Optional] int, seed for the ILP solver (0 if not provided)

            # pruning related
            "enable_pruning": bool, whether we should prune the cluster
            "min_keep": int, keep at least first min_keep links for each node during pruning
            "max_keep": int, keep at most first max_keep links for each node during pruning
            "keep_bandwidth_threshold": float, for links between [min_keep, max_keep), only those faster than
                                        the threshold will be kept.

            # ILP related
            "use_existing_sol": bool, whether to use existing solution
            "allow_partial_inference": bool, whether partial inference is allowed
            "remove_redundant": bool, whether redundant constraints are removed (mostly affects partial inference)
            "max_run_time": float, max ILP search time (only useful when use_existing_sol = False)
            "early_stop_time": float, early stop time (only useful when use_existing_sol = False)
            "early_stop_threshold": float, a value between 0 and 1 (only useful when use_existing_sol = False)
            "existing_sol_path": str, path to existing ILP solution (only useful when use_existing_sol = True)

            # heuristic solution to start from
            "start_from_heuristic": bool, whether to start from a heuristic solution (only useful when
                                    use_existing_sol = False)
            "heuristic_sol_path": str, path to heuristic solution (only useful when start_from_heuristic = True)

        When layout_method == LayoutMethod.Homogeneous:
            "seed": int, seed for random number generator

        When layout_method == LayoutMethod.Heterogeneous.Swarm:
            "seed": int, seed for random number generator
            "num_stages": int, should be able to divide total number of layers
            "max_out_links_per_node": int, how many nodes a node can connect to in the next stage

        When layout_method == LayoutMethod.Heterogeneous.Petals:
            "seed": int, random seed
            "max_out_links_per_node": int, how many nodes a node can connect to in the next stage

        When layout_method == LayoutMethod.LoadExisting:
            "solution_file_name": str, name of solution file
            "simulator_cluster_file_name": str, name of simulator cluster file

        :param args: a dict of arguments, see above for more info
        :return: simulator_cluster_file_path
        """
        if self.layout_method == LayoutMethod.ILP:
            self.layout_synthesizer: ILPLayout

            # if not using existing solution, make sure workspace path is empty
            if not args["use_existing_sol"]:
                assert os.listdir(self.workspace_path) == [], "Workspace path is not empty!"

            # save args as a file
            trail_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            trail_type: str = "load" if args["use_existing_sol"] else "new"
            with open(os.path.join(self.workspace_path, f"{trail_name + trail_type}.ini"), "w") as f:
                for k, v in args.items():
                    f.write(f"{k} = {v}\n")

            # get random seed
            seed: int = args["seed"] if "seed" in args else 0

            # prune the cluster
            enable_pruning: bool = args["enable_pruning"]
            if enable_pruning:
                min_keep: int = args["min_keep"]
                max_keep: int = args["max_keep"]
                keep_bandwidth_threshold: float = args["keep_bandwidth_threshold"]
                prune_cluster(
                    complete_cluster_file_name=self.complete_cluster_file_name,
                    pruned_cluster_file_name=os.path.join(self.workspace_path, "pruned_cluster.ini"),
                    min_keep=min_keep, max_keep=max_keep, keep_bandwidth_threshold=keep_bandwidth_threshold
                )
                processed_cluster_file_name = os.path.join(self.workspace_path, "pruned_cluster.ini")
            else:
                processed_cluster_file_name = self.complete_cluster_file_name

            # load the cluster and machine profile into layout synthesizer
            self.layout_synthesizer.from_ini(
                cluster_file_name=processed_cluster_file_name,
                machine_profile_name=self.machine_profile_name
            )

            # get heuristic solution
            start_from_heuristic: bool = args["start_from_heuristic"]
            heuristic_sol_path: str = args["heuristic_sol_path"]

            # find a solution with ILP (or load saved solution)
            use_existing_sol: bool = args["use_existing_sol"]
            allow_partial_inference: bool = args["allow_partial_inference"]
            remove_redundant: bool = args["remove_redundant"]
            if use_existing_sol:
                self.layout_synthesizer.load_and_verify_solution(
                    save_sol_path=args["existing_sol_path"], allow_partial_inference=allow_partial_inference
                )
            else:
                max_run_time: float = args["max_run_time"]
                early_stop_time: float = args["early_stop_time"]
                early_stop_threshold: float = args["early_stop_threshold"]
                self.layout_synthesizer.build_model(
                    seed=seed,
                    model_name=trail_name,
                    enable_partial_inference=allow_partial_inference,
                    remove_redundant=remove_redundant,
                    start_from_heuristic=start_from_heuristic,
                    heuristic_sol_path=heuristic_sol_path
                )
                self.layout_synthesizer.search_layout(
                    max_run_time=max_run_time,
                    early_stop_time=early_stop_time,
                    early_stop_threshold=early_stop_threshold,
                    save_sol_path=os.path.join(self.workspace_path, "ilp_solution.sol"),
                    save_model_path=os.path.join(self.workspace_path, "ilp_model.lp")
                )
                self.layout_synthesizer.load_and_verify_solution(
                    save_sol_path=os.path.join(self.workspace_path, "ilp_solution.sol"),
                    allow_partial_inference=allow_partial_inference
                )

            # generate simulator cluster input
            self.layout_synthesizer.generate_simulator_cluster(
                cluster_file_path=os.path.join(self.workspace_path, "simulator_cluster.ini"),
                allow_partial_inference=allow_partial_inference
            )
            self.layout_synthesizer.save_layout_solution(
                save_path=os.path.join(self.workspace_path, "ilp_sol.ini")
            )

        elif self.layout_method == LayoutMethod.Homogeneous:
            self.layout_synthesizer: HomogeneousLayout
            self.layout_synthesizer.from_ini(
                cluster_file_name=self.complete_cluster_file_name,
                machine_profile_name=self.machine_profile_name
            )
            self.layout_synthesizer.synthesize(seed=args["seed"])
            self.layout_synthesizer.generate_simulator_cluster(
                cluster_file_path=os.path.join(self.workspace_path, "simulator_cluster.ini")
            )
            self.layout_synthesizer.save_layout_solution(
                save_path=os.path.join(self.workspace_path, "homogeneous_sol.ini")
            )

        elif self.layout_method == LayoutMethod.Swarm:
            self.layout_synthesizer: SwarmLayout
            self.layout_synthesizer.from_ini(
                cluster_file_name=self.complete_cluster_file_name,
                machine_profile_name=self.machine_profile_name
            )
            self.layout_synthesizer.synthesize(num_stages=args["num_stages"])
            self.layout_synthesizer.generate_simulator_cluster(
                cluster_file_path=os.path.join(self.workspace_path, "simulator_cluster.ini"),
                max_out_links_per_node=args["max_out_links_per_node"],
                seed=args["seed"]
            )
            self.layout_synthesizer.save_layout_solution(
                save_path=os.path.join(self.workspace_path, "swarm_sol.ini")
            )

        elif self.layout_method == LayoutMethod.Petals:
            self.layout_synthesizer: PetalsLayout
            self.layout_synthesizer.from_ini(
                cluster_file_name=self.complete_cluster_file_name,
                machine_profile_name=self.machine_profile_name
            )
            self.layout_synthesizer.synthesize(seed=args["seed"])
            self.layout_synthesizer.generate_simulator_cluster(
                cluster_file_path=os.path.join(self.workspace_path, "simulator_cluster.ini"),
                max_out_links_per_node=args["max_out_links_per_node"]
            )
            self.layout_synthesizer.save_layout_solution(
                save_path=os.path.join(self.workspace_path, "petals_sol.ini")
            )

        elif self.layout_method == LayoutMethod.LoadExisting:
            self.layout_synthesizer: LoadExistingLayout
            self.layout_synthesizer.from_ini(
                cluster_file_name=self.complete_cluster_file_name,
                machine_profile_name=self.machine_profile_name
            )
            self.layout_synthesizer.load_solution(
                solution_file_name=args["solution_file_name"]
            )
            return args["simulator_cluster_file_name"]

        else:
            assert False, f"Found unknown layout method: {self.layout_method}!"

        # return the paths to the simulator cluster file and statistics file
        return os.path.join(self.workspace_path, "simulator_cluster.ini")

    def set_layout(self, simulator: ClusterSimulator) -> float:
        """
        Set the initial layout for the simulator.

        :param simulator: ClusterSimulator
        :return: float, time used to set the layout
        """
        if self.layout_method == LayoutMethod.ILP:
            self.layout_synthesizer: ILPLayout
            return self.layout_synthesizer.set_initial_layout(simulator=simulator)

        elif self.layout_method == LayoutMethod.Homogeneous:
            self.layout_synthesizer: HomogeneousLayout
            return self.layout_synthesizer.set_initial_layout(simulator=simulator)

        elif self.layout_method == LayoutMethod.Swarm:
            self.layout_synthesizer: SwarmLayout
            return self.layout_synthesizer.set_initial_layout(simulator=simulator)

        elif self.layout_method == LayoutMethod.Petals:
            self.layout_synthesizer: PetalsLayout
            return self.layout_synthesizer.set_initial_layout(simulator=simulator)

        elif self.layout_method == LayoutMethod.LoadExisting:
            self.layout_synthesizer: LoadExistingLayout
            return self.layout_synthesizer.set_initial_layout(simulator=simulator)

        else:
            assert False, f"Found unknown layout method: {self.layout_method}!"

    def get_flow_parameters(self) -> FlowParameters:
        """
        Get flow parameters based on the loaded cluster file.

        :return: FlowParameters
        """
        return self.layout_synthesizer.get_flow_parameters()

    def get_query_manager_parameters(self) -> QueryManagerParameters:
        """
        Get query manager parameters based on the loaded cluster file.

        :return: QueryManagerParameters
        """
        return self.layout_synthesizer.get_query_manager_parameters()
