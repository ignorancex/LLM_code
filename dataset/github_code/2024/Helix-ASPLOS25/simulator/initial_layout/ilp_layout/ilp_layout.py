# 2023.02.24 Yixuan Mei

import math
import time

import gurobipy as gp

from typing import List, Dict, Tuple, Set
from configparser import ConfigParser
from gurobipy import GRB

from simulator.event_simulator.utils import kbps, mbps, gbps, Byte, KB, MB, GB, Sec, MilliSec, ATOL, is_close
from simulator.event_simulator.cluster_simulator import ClusterSimulator
from simulator.event_simulator.query_manager import QueryManagerParameters
from simulator.model_manager.model_manager import ModelManager
from simulator.scheduler.global_maxflow.global_maxflow_scheduler import FlowParameters


class MachineProfile:
    def __init__(self, machine_name: str, config: ConfigParser) -> None:
        """
        Profile for a type of machine

        :param machine_name: name of the machine type
        :param config: config loaded from file
        """
        assert machine_name in config.sections(), "Machine type not found!"
        self.type_name: str = machine_name

        # network
        self.inbound_nic_speed: float = eval(config[machine_name]["inbound_nic_speed"])
        self.outbound_nic_speed: float = eval(config[machine_name]["outbound_nic_speed"])

        # storage
        self.disk_speed: float = eval(config[machine_name]["disk_speed"])
        self.vram_size: float = eval(config[machine_name]["vram_size"])


class ModelCard:
    def __init__(self, model_manager: ModelManager) -> None:
        """
        Description of the model we are serving.

        :param model_manager: model manager
        :return: None
        """
        self.num_layers: int = model_manager.get_num_layers()
        self.token_size: float = model_manager.get_model_token_size()
        self.activation_size: float = model_manager.get_model_activation_size()


class ILPLink:
    def __init__(self, from_index: int or str, to_index: int or str, throughput: float,
                 bandwidth: float, latency: float) -> None:
        """
        Represent a link in ILP. Contains all information needed.
        Note: The link is bidirectional!

        :param from_index: index of the start node
        :param to_index: index of the destination node
        :param throughput: token throughput over this link
        :param bandwidth: bandwidth of this link
        :param latency: latency of this link
        :return: None
        """
        self.from_index: int or str = from_index
        self.to_index: int or str = to_index
        self.throughput: float = throughput
        self.bandwidth: float = bandwidth
        self.latency: float = latency

        # from ilp solution
        # edge_switch: 0 = edge not enabled, 1 = edge enabled
        # forward edge (from -> to)
        self.forward_flow: float = -1.0
        self.forward_edge_switch: int = -1
        self.forward_edge_cond1: int = -1
        self.forward_edge_cond2: int = -1
        # backward edge (to -> from)
        self.backward_flow: float = -1.0
        self.backward_edge_switch: int = -1
        self.backward_edge_cond1: int = -1
        self.backward_edge_cond2: int = -1


class ILPNode:
    def __init__(self, node_index: int, machine_type: MachineProfile, max_num_layers: int,
                 connected_node_indices: List[int], layer_count_2_throughput: Dict[int, float]) -> None:
        """
        Represent a compute node in ILP. Contains all information needed.

        :param node_index: index of this node
        :param machine_type: machine type
        :param max_num_layers: max number of layers this machine can hold
        :param connected_node_indices: which nodes are connected to this node
        :param layer_count_2_throughput: throughput when there are k layers on node (bounded by nic speed)
        :return: None
        """
        self.node_index: int = node_index
        self.machine_type: MachineProfile = machine_type
        self.max_num_layers: int = max_num_layers
        self.connected_node_indices: List[int] = connected_node_indices
        self.layer_count_2_throughput: Dict[int, float] = layer_count_2_throughput

        # from ilp solution
        # model on node is [start_idx, end_idx)
        self.start_layer_idx: int = -1
        self.end_layer_idx: int = -1


class ILPLayout:
    # Usage:
    # 1. call "from_ini" to load a complete cluster topology and machine profile
    # 2. [omit if already have solution] call "build_model" to build the ILP model
    # 3. [omit if already have solution] call "search_layout" to let Gurobi find the optimal solution
    # 4. call "load_and_verify_solution" to load the ILP solution
    # 5. call "generate_simulator_cluster" to generate the cluster file and machine statistics file that
    #    will be used by the simulator
    # 6. after cluster is initialized (see line 134 of /simulator/event_simulator/cluster_simulator.py), call
    #    set_initial_layout to load the initial layout into cluster simulator

    def __init__(self, model_manager: ModelManager) -> None:
        """
        MILP-based initial layout Synthesizer.

        :return: None
        """
        # loaded problem information
        self.machine_profiles: Dict[str, MachineProfile] = {}
        self.model_card: ModelCard or None = None
        self.model_manager: ModelManager = model_manager

        # cluster information
        self.ilp_source: ILPNode or None = None
        self.ilp_sink: ILPNode or None = None
        self.ilp_nodes: Dict[int, ILPNode] = {}
        self.ilp_links: Dict[Tuple[int or str, int or str], ILPLink] = {}
        self.cluster_loaded: bool = False
        self.solution_loaded: bool = False

        # ilp model
        self.model_initialized: bool = False
        self.ilp_model: gp.Model or None = None

        # variables
        self.var_node_start: Dict[str, gp.Var] = {}
        self.var_node_hold_layer: Dict[int, Dict[str, gp.Var]] = {}
        self.var_flow: Dict[str, gp.Var] = {}
        self.var_edge_switch: Dict[str, gp.Var] = {}
        # tmp variables that are only used when allow partial inference
        self.tmp_var_compute_edge_cond1: Dict[str, gp.Var] = {}
        self.tmp_var_compute_edge_cond2: Dict[str, gp.Var] = {}

        # constraints
        self.constr_hold: Dict[str, gp.Constr] = {}
        self.constr_end: Dict[str, gp.Constr] = {}
        self.constr_node_flow: Dict[str, gp.Constr] = {}
        self.constr_node_throughput: Dict[str, gp.Constr] = {}
        self.constr_edge_enabled: Dict[str, gp.Constr] = {}
        self.constr_edge_disabled: Dict[str, gp.Constr] = {}
        self.constr_edge_flow: Dict[str, gp.Constr] = {}
        # tmp constraints that are only used when allow partial inference
        self.tmp_constr_cond1_enabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond1_disabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond2_enabled: Dict[str, gp.Constr] = {}
        self.tmp_constr_cond2_disabled: Dict[str, gp.Constr] = {}

        # optimization status
        # stopping criteria
        self.max_run_time: float = -1
        self.early_stop_threshold: float = -1
        self.early_stop_time: float = -1
        # run time
        self.opt_start_time: float = -1
        self.opt_best_obj: float = -1
        self.opt_best_obj_found_time: float = -1
        self.opt_upper_bound: float = -1

        # parameters for simulator cluster file and initial layout generation
        self.node_idx_offset = 2  # since 0 and 1 are reserved for source and sink

    def from_ini(self, cluster_file_name: str, machine_profile_name: str) -> None:
        """
        Initialize the ILP using a given cluster topology and machine profiles.

        :param cluster_file_name: name of the file that stores cluster topology
        :param machine_profile_name: name of the file that stores machine profiling results
        :return: None
        """
        # clear the dicts
        self.machine_profiles.clear()
        self.ilp_nodes.clear()
        self.ilp_links.clear()

        # load machine statistics
        machine_profile_parser = ConfigParser()
        machine_profile_parser.read(machine_profile_name)
        for machine_name in machine_profile_parser.sections():
            self.machine_profiles[machine_name] = MachineProfile(machine_name=machine_name,
                                                                 config=machine_profile_parser)

        # load cluster topology
        cluster_file_parser = ConfigParser()
        cluster_file_parser.read(cluster_file_name)

        # model
        self.model_card = ModelCard(model_manager=self.model_manager)

        # source and sink
        self.ilp_source = ILPNode(node_index=-1, machine_type=self.machine_profiles["SourceNode"], max_num_layers=-1,
                                  connected_node_indices=eval(cluster_file_parser["SourceNode"]["connected_nodes"]),
                                  layer_count_2_throughput={})
        self.ilp_sink = ILPNode(node_index=-1, machine_type=self.machine_profiles["SinkNode"], max_num_layers=-1,
                                connected_node_indices=eval(cluster_file_parser["SinkNode"]["connected_nodes"]),
                                layer_count_2_throughput={})

        # compute nodes
        total_compute_nodes: int = eval(cluster_file_parser["NodeNames"]["total_compute_nodes"])
        for node_idx in range(total_compute_nodes):
            # extract machine name, type and connected nodes from file
            machine_name: str = f"ComputeNode-{node_idx}"
            machine_type: MachineProfile = self.machine_profiles[cluster_file_parser[machine_name]["type"]]
            connected_nodes: List[int] = eval(cluster_file_parser[machine_name]["connected_nodes"])

            # compute max number of layers that can be stored on this node
            # Note: max # layers = (VRAM size / 2) / layer size
            max_num_layers: int = self.model_manager.get_max_num_layers(machine_type=machine_type.type_name)
            assert 2 * max_num_layers * max(self.model_manager.get_model_params()) <= machine_type.vram_size + 1, \
                "Trying to use more than half the vram to load model parameters!"

            # compute layer count to throughput
            # Note: 1. inference throughput is computed under typical batch size
            #       2. total throughput is the min of inference throughput and nic throughput
            bottleneck_nic_speed: float = min(machine_type.inbound_nic_speed, machine_type.outbound_nic_speed)
            bottleneck_nic_throughput: float = bottleneck_nic_speed / self.model_card.activation_size
            layer_count_2_throughput: Dict[int, float] = {}
            for layer_count in range(1, max_num_layers + 1):
                inference_throughput: float = self.model_manager.get_typical_token_throughput(
                    machine_type=machine_type.type_name, num_on_node_layers=layer_count
                )
                layer_count_2_throughput[layer_count] = min(inference_throughput, bottleneck_nic_throughput)

            # add node
            self.ilp_nodes[node_idx] = ILPNode(node_index=node_idx,
                                               machine_type=machine_type,
                                               max_num_layers=max_num_layers,
                                               connected_node_indices=connected_nodes,
                                               layer_count_2_throughput=layer_count_2_throughput)

        # links
        # Note: links here are bidirectional
        for entity_name in cluster_file_parser.sections():
            if "Link-" in entity_name:
                # end points
                from_idx: int or str = entity_name.split("-")[1]
                if not from_idx == "source":
                    from_idx = int(from_idx)
                to_idx: int or str = entity_name.split("-")[2]
                if not to_idx == "sink":
                    to_idx = int(to_idx)

                # bandwidth and latency
                bandwidth: float = eval(cluster_file_parser[entity_name]["bandwidth"])
                latency: float = eval(cluster_file_parser[entity_name]["latency"])
                if from_idx == "source" or to_idx == "sink":
                    throughput: float = bandwidth / self.model_card.token_size
                else:
                    assert isinstance(from_idx, int) and isinstance(to_idx, int), "Bad index!"
                    throughput: float = bandwidth / self.model_card.activation_size
                self.ilp_links[(from_idx, to_idx)] = ILPLink(from_index=from_idx,
                                                             to_index=to_idx,
                                                             throughput=throughput,
                                                             bandwidth=bandwidth,
                                                             latency=latency)

        # mark cluster as loaded
        self.cluster_loaded = True

    def get_end_layer_index(self, compute_node_idx: int) -> gp.LinExpr:
        """
        Get the end layer index of compute node i.

        :param compute_node_idx: index of the compute node
        :return: a LinExpr that represent the end layer index of compute node i
        """
        start_var: gp.Var = self.var_node_start[f"start_{compute_node_idx}"]
        k_hold_var: List[gp.LinExpr] = []
        for layer_count in range(1, self.ilp_nodes[compute_node_idx].max_num_layers + 1):
            hold_var_name = f"hold_{compute_node_idx}_{layer_count}"
            hold_var: gp.Var = self.var_node_hold_layer[compute_node_idx][hold_var_name]
            k_hold_var.append(layer_count * hold_var)
        return start_var + gp.quicksum(k_hold_var)

    def step1_initialize_ilp(self, seed: int, model_name: str) -> None:
        """
        Initialize the ILP program.

        :param seed: random seed
        :param model_name: name of the model
        :return: None
        """
        self.ilp_model = gp.Model(model_name)
        self.ilp_model.Params.Seed = seed

        # variables
        self.var_node_start.clear()
        self.var_node_hold_layer.clear()
        self.var_flow.clear()
        self.var_edge_switch.clear()
        self.tmp_var_compute_edge_cond1.clear()
        self.tmp_var_compute_edge_cond2.clear()

        # constraints
        self.constr_hold.clear()
        self.constr_end.clear()
        self.constr_node_flow.clear()
        self.constr_node_throughput.clear()
        self.constr_edge_enabled.clear()
        self.constr_edge_disabled.clear()
        self.constr_edge_flow.clear()
        self.tmp_constr_cond1_enabled.clear()
        self.tmp_constr_cond1_disabled.clear()
        self.tmp_constr_cond2_enabled.clear()
        self.tmp_constr_cond2_disabled.clear()

    def step2_add_variables(self, allow_partial_inference: bool, remove_redundant: bool,
                            start_from_heuristic: bool, heuristic_sol_path: str) -> Tuple[int, int, int]:
        """
        Add decision variables into this ILP program.

        :param allow_partial_inference: whether we allow partial inference or not
        :param remove_redundant: remove redundant constraints in the model
        :param start_from_heuristic: whether we start from a heuristic solution
        :param heuristic_sol_path: path to the heuristic solution
        :return: num_int, num_real, num_binary
        """
        num_int, num_real, num_binary = 0, 0, 0

        # Step 2.0: If we start from a heuristic solution, load the solution
        # we only set start layer id and end layer id for each node
        config = ConfigParser()
        if start_from_heuristic:
            config.read(heuristic_sol_path)
            heuristic_offset = eval(config["Settings"]["offset"])
        else:
            heuristic_offset = None

        # Step 2.1: add starting layer index for each node as variable
        # var_name: start_i (s_i)
        # var_type: int
        # var_range: {0, 1, ..., # layers  - 1}
        # number of variables: n
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            start_var_name = f"start_{compute_node_idx}"
            if remove_redundant:
                start_var = self.ilp_model.addVar(vtype=GRB.INTEGER,
                                                  lb=0,
                                                  name=start_var_name)
            else:
                start_var = self.ilp_model.addVar(vtype=GRB.INTEGER,
                                                  lb=0,
                                                  ub=self.model_card.num_layers - 1,
                                                  name=start_var_name)
            if start_from_heuristic:
                heuristic_node_idx = compute_node_idx + heuristic_offset
                layers_on_node: List[int] = eval(config["Solution"][f"compute_node_{heuristic_node_idx}"])
                start_var.Start = min(layers_on_node)
            self.var_node_start[start_var_name] = start_var
            num_int += 1

        # Step 2.2: add whether each node holds k layers as variable
        # var_name: hold_i_k (b_ik)
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: kn
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            self.var_node_hold_layer[compute_node_idx] = {}
            for layer_count in range(1, compute_node.max_num_layers + 1):
                hold_var_name = f"hold_{compute_node_idx}_{layer_count}"
                hold_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                 name=hold_var_name)
                if start_from_heuristic:
                    heuristic_node_idx = compute_node_idx + heuristic_offset
                    layers_on_node: List[int] = eval(config["Solution"][f"compute_node_{heuristic_node_idx}"])
                    if layer_count == len(layers_on_node):
                        hold_var.Start = 1
                    else:
                        hold_var.Start = 0
                self.var_node_hold_layer[compute_node_idx][hold_var_name] = hold_var
                num_binary += 1

        # Step 2.3: add flow over each edge as variable
        # var_name: flow_i_j (f_ij)
        # var_type: continuous
        # var_range: [0, +inf)
        # number of variables: 2e
        for link_name_tuple, link in self.ilp_links.items():
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                flow_var_name = f"flow_{link_name_tuple[0]}_{link_name_tuple[1]}"
                flow_var = self.ilp_model.addVar(vtype=GRB.CONTINUOUS,
                                                 lb=0,
                                                 ub=GRB.INFINITY,
                                                 name=flow_var_name)
                self.var_flow[flow_var_name] = flow_var
                num_real += 1
            if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                reversed_flow_var_name = f"flow_{link_name_tuple[1]}_{link_name_tuple[0]}"
                reversed_flow_var = self.ilp_model.addVar(vtype=GRB.CONTINUOUS,
                                                          lb=0,
                                                          ub=GRB.INFINITY,
                                                          name=reversed_flow_var_name)
                self.var_flow[reversed_flow_var_name] = reversed_flow_var
                num_real += 1

        # Step 2.4 add whether each edge is enabled (edge switch) as variable
        # var_name: switch_i_j (d_ij)
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: 2e
        for link_name_tuple, link in self.ilp_links.items():
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                forward_switch_name = f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"
                forward_switch_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                           name=forward_switch_name)
                self.var_edge_switch[forward_switch_name] = forward_switch_var
                num_binary += 1
            if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                backward_switch_name = f"switch_{link_name_tuple[1]}_{link_name_tuple[0]}"
                backward_switch_var = self.ilp_model.addVar(vtype=GRB.BINARY,
                                                            name=backward_switch_name)
                self.var_edge_switch[backward_switch_name] = backward_switch_var
                num_binary += 1

        # Step 2.5 add tmp condition variables for each edge between compute nodes if partial inference is allowed
        # var_name: edge_cond1_i_j and edge_cond2_i_j
        # var_type: bool
        # var_range: {0, 1}
        # number of variables: 4e
        if allow_partial_inference:
            for link_name_tuple, link in self.ilp_links.items():
                # edge between source and sink does not need tmp variables
                if "source" in link_name_tuple or "sink" in link_name_tuple:
                    continue

                # variable names
                forward_cond1_name = f"edge_cond1_{link_name_tuple[0]}_{link_name_tuple[1]}"
                forward_cond2_name = f"edge_cond2_{link_name_tuple[0]}_{link_name_tuple[1]}"
                backward_cond1_name = f"edge_cond1_{link_name_tuple[1]}_{link_name_tuple[0]}"
                backward_cond2_name = f"edge_cond2_{link_name_tuple[1]}_{link_name_tuple[0]}"

                # variables
                forward_cond1_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=forward_cond1_name)
                forward_cond2_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=forward_cond2_name)
                backward_cond1_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=backward_cond1_name)
                backward_cond2_var = self.ilp_model.addVar(vtype=GRB.BINARY, name=backward_cond2_name)

                # save variables
                self.tmp_var_compute_edge_cond1[forward_cond1_name] = forward_cond1_var
                self.tmp_var_compute_edge_cond2[forward_cond2_name] = forward_cond2_var
                self.tmp_var_compute_edge_cond1[backward_cond1_name] = backward_cond1_var
                self.tmp_var_compute_edge_cond2[backward_cond2_name] = backward_cond2_var
                num_binary += 4

        return num_int, num_real, num_binary

    def step3_model_placement_constraint(self) -> int:
        """
        Add model placement constraints.

        :return: number of constraints added
        """
        num_constraint = 0

        for compute_node_idx, compute_node in self.ilp_nodes.items():
            # Step 3.1: add constraint: only one model placement is valid
            # constraint_name: hold_constraint_i
            # constraint: \sum_k hold_i_k = 1 (\sum_k b_ik = 1)
            # number of constraints: n
            sum_of_hold_var: gp.LinExpr = gp.quicksum(list(self.var_node_hold_layer[compute_node_idx].values()))
            hold_constraint_name = f"hold_constraint_{compute_node_idx}"
            # noinspection PyTypeChecker
            hold_constraint: gp.Constr = self.ilp_model.addConstr(sum_of_hold_var == 1,
                                                                  name=hold_constraint_name)
            self.constr_hold[hold_constraint_name] = hold_constraint
            num_constraint += 1

            # Step 3.2: add constraint: end layer idx on each node should <= # layers
            # constraint_name: end_constraint_i
            # constraint: start_i + \sum_k k * hold_i_k <= m (s_{i} + \sum_k k * b_{ik} <= m)
            # number of constraints: n
            end_layer_idx_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=compute_node_idx)
            end_constraint_name = f"end_constraint_{compute_node_idx}"
            end_constraint: gp.Constr = self.ilp_model.addConstr(end_layer_idx_expr <= self.model_card.num_layers,
                                                                 name=end_constraint_name)
            self.constr_end[end_constraint_name] = end_constraint
            num_constraint += 1

        return num_constraint

    def step4_flow_in_out_constraint(self) -> int:
        """
        Add constraints for flow in = flow out.

        :return: number of constraints added
        """
        num_constraint = 0

        # Step 4.1: add constraint: flow in = flow out for each compute node
        # constraint_name: node_flow_constraint_i
        # constraint: \sum_{u} flow_u_i = \sum_{j} flow_i_j (\sum_u f_{ui} = \sum_j f_{ij})
        # number of constraints: n
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            # compute flow
            flow_in_list, flow_out_list = [], []
            for other_idx in compute_node.connected_node_indices:
                if not other_idx == "sink":
                    flow_in_list.append(self.var_flow[f"flow_{other_idx}_{compute_node_idx}"])
                if not other_idx == "source":
                    flow_out_list.append(self.var_flow[f"flow_{compute_node_idx}_{other_idx}"])

            # add constraint
            node_flow_constraint_name = f"node_flow_constraint_{compute_node_idx}"
            node_flow_constraint = self.ilp_model.addConstr(gp.quicksum(flow_in_list) == gp.quicksum(flow_out_list),
                                                            name=node_flow_constraint_name)
            self.constr_node_flow[node_flow_constraint_name] = node_flow_constraint
            num_constraint += 1

        return num_constraint

    def step5_node_throughput_constraint(self) -> int:
        """
        Add constraints for flow over compute nodes.

        :return: number of constraints added
        """
        num_constraint = 0

        # Step 5.1: add constraint: flow over each compute node is smaller than its inference throughput
        # constraint_name: node_throughput_constraint_i
        # constraint: \sum_{u} flow_u_i <= \sum_k hold_i_k * throughput at k layers
        #             (\sum_u f_{ui} <= \sum_k b_{ik} * throughput at k layers)
        # number of constraints: n
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            # get flow through the node
            flow_in_list: List[gp.Var] = []
            for other_idx in compute_node.connected_node_indices:
                if not other_idx == "sink":
                    flow_in_list.append(self.var_flow[f"flow_{other_idx}_{compute_node_idx}"])

            # get one hot throughput
            one_hot_throughput_list: List[gp.LinExpr] = []
            for layer_count in range(1, compute_node.max_num_layers + 1):
                throughput_at_k: float = compute_node.layer_count_2_throughput[layer_count]
                hold_k: gp.Var = self.var_node_hold_layer[compute_node_idx][f"hold_{compute_node_idx}_{layer_count}"]
                one_hot_throughput_list.append(throughput_at_k * hold_k)

            # add constraint
            node_tp_constr_name = f"node_throughput_constraint_{compute_node_idx}"
            node_tp_constr = self.ilp_model.addConstr(gp.quicksum(flow_in_list) <= gp.quicksum(one_hot_throughput_list),
                                                      name=node_tp_constr_name)
            self.constr_node_throughput[node_tp_constr_name] = node_tp_constr
            num_constraint += 1

        return num_constraint

    def step6_edge_switch_constraint(self, allow_partial_inference: bool, remove_redundant: bool) -> int:
        """
        Add constraint for edge switch variables.

        :param allow_partial_inference: whether partial inference is allowed
        :param remove_redundant: remove redundant constraints in the model
        :return: number of constraints added
        """
        num_constraints = 0

        # Step 6.1: build a list of edges that we need to process
        list_name_list: List[Tuple[str or int, str or int]] = []
        for link_name_tuple in self.ilp_links.keys():
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                list_name_list.append((link_name_tuple[0], link_name_tuple[1]))
            if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                list_name_list.append((link_name_tuple[1], link_name_tuple[0]))

        # Step 6.2: add constraint for edge switch
        for link_name_tuple in list_name_list:
            assert not (link_name_tuple[0] == "source" and link_name_tuple[1] == "sink"), \
                "Found direct link between source and sink!"

            if link_name_tuple[0] == "source":
                # Case 1: link from source to i
                # switch = cond(start_i == 0)

                # ------------ Prop 1: Linearize b = 1 iff a = 0 ------------ #
                # int a \in [0, m - 1]
                # bool b = 0 or 1
                # express b = 1 iff a = 0
                # if a = 0 then b = 1:	b >= 1 - a
                # if a > 0 then b = 0:	a <= m(1-b)
                # ----------------------------------------------------------- #

                # get the variables
                edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                node_i_start_var: gp.Var = self.var_node_start[f"start_{link_name_tuple[1]}"]

                # add two constraints
                # d = edge_switch_var, s_i = node_i_start_var
                # (1) if s_i = 0 then d = 1:	d >= 1 - s_i
                if not remove_redundant:
                    enable_condition_name = f"edge_enable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    enable_condition_constr = self.ilp_model.addConstr(edge_switch_var >= 1 - node_i_start_var,
                                                                       name=enable_condition_name)
                    self.constr_edge_enabled[enable_condition_name] = enable_condition_constr
                    num_constraints += 1
                # (2) if s_i > 0 then d = 0:	s_i <= m(1-d)
                disable_condition_name = f"edge_disable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                disable_condition_constr = self.ilp_model.addConstr(
                    node_i_start_var <= self.model_card.num_layers * (1 - edge_switch_var),
                    name=disable_condition_name)
                self.constr_edge_disabled[disable_condition_name] = disable_condition_constr
                num_constraints += 1

            elif link_name_tuple[1] == "sink":
                # Case 2: link from i to sink
                # switch = cond(end_i == m)

                # ------------ Prop 2: Linearize b = 1 iff a = m ------------ #
                # int a \in [0, m]
                # bool b = 0 or 1
                # express b = 1 iff a = m
                # if a = m then b = 1:		(m - 1)(b + 1) >= a
                # if 0 <= a < m then b = 0:	mb <= a
                # ----------------------------------------------------------- #

                # get the variables
                edge_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                node_i_start_var: gp.Var = self.var_node_start[f"start_{link_name_tuple[0]}"]

                # compute node i's end position
                k_hold_var: List[gp.LinExpr] = []
                for layer_count in range(1, self.ilp_nodes[link_name_tuple[0]].max_num_layers + 1):
                    hold_var_name = f"hold_{link_name_tuple[0]}_{layer_count}"
                    hold_var: gp.Var = self.var_node_hold_layer[link_name_tuple[0]][hold_var_name]
                    k_hold_var.append(layer_count * hold_var)
                node_i_end_expr: gp.LinExpr = node_i_start_var + gp.quicksum(k_hold_var)

                # add two constraints
                # d = edge_switch_var, e_i = node_i_end_expr
                # (1) if e_i = m then d = 1:		(m - 1)(d + 1) >= e_i
                if not remove_redundant:
                    enable_condition_name = f"edge_enable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    enable_condition_constr = self.ilp_model.addConstr(
                        (self.model_card.num_layers - 1) * (edge_switch_var + 1) >= node_i_end_expr,
                        name=enable_condition_name)
                    self.constr_edge_enabled[enable_condition_name] = enable_condition_constr
                    num_constraints += 1
                # (2) if 0 <= e_i < m then d = 0:	md <= e_i
                disable_condition_name = f"edge_disable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                disable_condition_constr = self.ilp_model.addConstr(
                    self.model_card.num_layers * edge_switch_var <= node_i_end_expr,
                    name=disable_condition_name)
                self.constr_edge_disabled[disable_condition_name] = disable_condition_constr
                num_constraints += 1

            else:
                if allow_partial_inference:
                    # Case 3.1: link between compute node i and j & allow partial inference
                    # switch = cond(start_j <= end_i < end_j)

                    # get the layer index variables
                    start_j_var: gp.Var = self.var_node_start[f"start_{link_name_tuple[1]}"]
                    end_i_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=link_name_tuple[0])
                    end_j_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=link_name_tuple[1])

                    # get the condition variables
                    cond1_var_name = f"edge_cond1_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    cond2_var_name = f"edge_cond2_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    edge_switch_name = f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    cond1_var: gp.Var = self.tmp_var_compute_edge_cond1[cond1_var_name]
                    cond2_var: gp.Var = self.tmp_var_compute_edge_cond2[cond2_var_name]
                    switch_var: gp.Var = self.var_edge_switch[edge_switch_name]

                    # Condition 1: end_i - start_j >= 0
                    # ------------ Prop 3: Linearize b = 1 iff a >= 0 ------------ #
                    # int a \in [-m, m]
                    # bool b = 0 or 1
                    # express b = 1 iff a >= 0
                    # if a >= 0 then b = 1:  	(m+1) b >= a + 1
                    # if a < 0 then b = 0: 	    (m+1)(1 - b) >= -a
                    # ------------------------------------------------------------ #
                    # add two constraints
                    # tmp_1 = end_i_expr - start_j_var, d_1 = cond1_var
                    # (1) if tmp_1 >= 0 then d_1 = 1:	(m + 1) d_1 >= tmp_1 + 1
                    if not remove_redundant:
                        cond1_enabled_constr_name = f"edge_cond1_enabled_{link_name_tuple[0]}_{link_name_tuple[1]}"
                        cond1_enabled_constr = self.ilp_model.addConstr(
                            (self.model_card.num_layers + 1) * cond1_var >= end_i_expr - start_j_var + 1,
                            name=cond1_enabled_constr_name
                        )
                        self.tmp_constr_cond1_enabled[cond1_enabled_constr_name] = cond1_enabled_constr
                        num_constraints += 1
                    # (2) if tmp_1 < 0 then d_1 = 0: 	(m + 1)(1 - d_1) >= -tmp_1
                    cond1_disabled_constr_name = f"edge_cond1_disabled_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    cond1_disabled_constr = self.ilp_model.addConstr(
                        (self.model_card.num_layers + 1) * (1 - cond1_var) >= start_j_var - end_i_expr,
                        name=cond1_disabled_constr_name
                    )
                    self.tmp_constr_cond1_disabled[cond1_disabled_constr_name] = cond1_disabled_constr
                    num_constraints += 1

                    # Condition 2: end_j - end_i > 0
                    # ------------ Prop 4: Linearize b = 1 iff a > 0 ------------ #
                    # int a \in [-m, m]
                    # bool b = 0 or 1
                    # express b = 1 iff a > 0
                    # if a > 0 then b = 1:  	(m+1) b >= a
                    # if a <= 0 then b = 0: 	a >= 1 - (m+1) (1 - b)
                    # ----------------------------------------------------------- #
                    # add two constraints
                    # tmp_2 = end_j_expr - end_i_expr, d_2 = cond2_var
                    # (1) if tmp_2 > 0 then d_2 = 1:  	(m+1) d_2 >= tmp_2
                    if not remove_redundant:
                        cond2_enabled_constr_name = f"edge_cond2_enabled_{link_name_tuple[0]}_{link_name_tuple[1]}"
                        cond2_enabled_constr = self.ilp_model.addConstr(
                            (self.model_card.num_layers + 1) * cond2_var >= end_j_expr - end_i_expr,
                            name=cond2_enabled_constr_name
                        )
                        self.tmp_constr_cond2_enabled[cond2_enabled_constr_name] = cond2_enabled_constr
                        num_constraints += 1
                    # (2) if tmp_2 <= 0 then d_2 = 0: 	tmp_2 >= 1 - (m+1) (1 - d_2)
                    cond2_disabled_constr_name = f"edge_cond2_disabled_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    cond2_disabled_constr = self.ilp_model.addConstr(
                        end_j_expr - end_i_expr >= 1 - (self.model_card.num_layers + 1) * (1 - cond2_var),
                        name=cond2_disabled_constr_name
                    )
                    self.tmp_constr_cond2_disabled[cond2_disabled_constr_name] = cond2_disabled_constr
                    num_constraints += 1

                    # Switch constraint
                    # ------------ Prop 5: Linearize z = 1 iff x and y are both 1 ------------ #
                    # bool x, y, z
                    # express z = 1 iff x and y are both 1
                    # if x = 1, y = 1 then z = 1:	    x + y - z <= 1
                    # if x = 0 or y = 0 then z = 0:	    z <= 0.5 * x + 0.5 * y
                    # ------------------------------------------------------------------------ #
                    # add two constraints
                    # d_1 = cond1_var, d_2 = cond2_var, d = switch_var
                    # (1) if d_1 = 1, d_2 = 1 then d = 1:		d_1 + d_2 - d <= 1
                    if not remove_redundant:
                        switch_enabled_constr_name = f"edge_enable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                        switch_enabled_constr = self.ilp_model.addConstr(
                            cond1_var + cond2_var - switch_var <= 1,
                            name=switch_enabled_constr_name
                        )
                        self.constr_edge_enabled[switch_enabled_constr_name] = switch_enabled_constr
                        num_constraints += 1
                    # (2) if d_1 = 0 or d_2 = 0 then d = 0:	    d <= 0.5 * d_1 + 0.5 * d_2
                    switch_disabled_constr_name = f"edge_disable_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    switch_disabled_constr = self.ilp_model.addConstr(
                        2 * switch_var <= cond1_var + cond2_var,
                        name=switch_disabled_constr_name
                    )
                    self.constr_edge_disabled[switch_disabled_constr_name] = switch_disabled_constr
                    num_constraints += 1

                else:
                    # Case 3.2: link between compute node i and j & does not allow partial inference
                    # switch = cond(end_i == start_j)

                    # get the layer index variables
                    start_j_var: gp.Var = self.var_node_start[f"start_{link_name_tuple[1]}"]
                    end_i_expr: gp.LinExpr = self.get_end_layer_index(compute_node_idx=link_name_tuple[0])

                    # get the condition variables
                    edge_switch_name = f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    switch_var: gp.Var = self.var_edge_switch[edge_switch_name]

                    # Switch constraint
                    # ----------------- Prop 6: Linearize b = 0 if a \neq 0 ------------------ #
                    # int a \in [-m, m]
                    # bool b
                    # express b = 0 if a \neq 0
                    # b <= 1 + a / m
                    # b <= 1 - a / m
                    # ------------------------------------------------------------------------ #
                    # add two constraints (both to disable constraints)
                    # a = start_j_var - end_i_expr, b = switch_var (m = self.model_card.num_layers)
                    _m = self.model_card.num_layers
                    # (1) m * b <= m + a
                    switch_disabled_constr1_name = f"edge_disable_constr1_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    switch_disabled_constr1 = self.ilp_model.addConstr(
                        _m * switch_var <= _m + (start_j_var - end_i_expr),
                        name=switch_disabled_constr1_name
                    )
                    self.constr_edge_disabled[switch_disabled_constr1_name] = switch_disabled_constr1
                    num_constraints += 1
                    # (2) m * b <= m - a
                    switch_disabled_constr2_name = f"edge_disable_constr2_{link_name_tuple[0]}_{link_name_tuple[1]}"
                    switch_disabled_constr2 = self.ilp_model.addConstr(
                        _m * switch_var <= _m - (start_j_var - end_i_expr),
                        name=switch_disabled_constr2_name
                    )
                    self.constr_edge_disabled[switch_disabled_constr2_name] = switch_disabled_constr2
                    num_constraints += 1

        return num_constraints

    def step7_edge_flow_constraint(self) -> int:
        """
        Add constraint for flow over each edge.

        :return: number of constraints added
        """
        num_constraints = 0

        # constraint_name: edge_flow_constr_i_j
        # constraint: flow_i_j <= link_throughput * switch_i_j
        # number of constraints: 2e
        for link_name_tuple, link in self.ilp_links.items():
            link_throughput = link.throughput

            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                forward_flow_var: gp.Var = self.var_flow[f"flow_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                forward_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[0]}_{link_name_tuple[1]}"]
                forward_edge_flow_constraint_name = f"edge_flow_constr_{link_name_tuple[0]}_{link_name_tuple[1]}"
                forward_edge_flow_constraint = self.ilp_model.addConstr(
                    forward_flow_var <= link_throughput * forward_switch_var,
                    name=forward_edge_flow_constraint_name
                )
                self.constr_edge_flow[forward_edge_flow_constraint_name] = forward_edge_flow_constraint
                num_constraints += 1

            if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                backward_flow_var: gp.Var = self.var_flow[f"flow_{link_name_tuple[1]}_{link_name_tuple[0]}"]
                backward_switch_var: gp.Var = self.var_edge_switch[f"switch_{link_name_tuple[1]}_{link_name_tuple[0]}"]
                backward_edge_flow_constraint_name = f"edge_flow_constr_{link_name_tuple[1]}_{link_name_tuple[0]}"
                backward_edge_flow_constraint = self.ilp_model.addConstr(
                    backward_flow_var <= link_throughput * backward_switch_var,
                    name=backward_edge_flow_constraint_name
                )
                self.constr_edge_flow[backward_edge_flow_constraint_name] = backward_edge_flow_constraint
                num_constraints += 1

        return num_constraints

    def build_model(self, seed: int, model_name: str, enable_partial_inference: bool, remove_redundant: bool,
                    start_from_heuristic: bool, heuristic_sol_path: str) -> Tuple[int, int, int, int]:
        """
        Build the ILP model.
        Note: 1. Here we build the ILP model exactly as is based on the cluster we just loaded. Optimizations
                 like prune edge / layer fusion / limit on the lower bound of layers on node can be done by
                 changing the cluster description file.

        :param seed: random seed
        :param model_name: name of the ILP model
        :param enable_partial_inference: whether partial inference is enabled or not
        :param remove_redundant: remove redundant constraints in the model
        :param start_from_heuristic: whether to start from a heuristic solution
        :param heuristic_sol_path: path to the heuristic solution
        :return: (int variables, real variables, binary variables, num_constraints)
        """
        # prepare the ILP program
        assert self.cluster_loaded, "Must load a cluster before building the ilp model!"
        num_int, num_real, num_binary, num_constraint = 0, 0, 0, 0

        # Step 1: initial the ILP program
        self.step1_initialize_ilp(seed=seed, model_name=model_name)

        # Step 2: add variables
        cur_num_int, cur_num_real, cur_num_binary = self.step2_add_variables(
            allow_partial_inference=enable_partial_inference,
            remove_redundant=remove_redundant,
            start_from_heuristic=start_from_heuristic,
            heuristic_sol_path=heuristic_sol_path
        )
        num_int += cur_num_int
        num_real += cur_num_real
        num_binary += cur_num_binary

        # Step 3: add constraint for model placement
        cur_num_constraint = self.step3_model_placement_constraint()
        num_constraint += cur_num_constraint

        # Step 4: add constraint for flow in = flow out
        cur_num_constraint = self.step4_flow_in_out_constraint()
        num_constraint += cur_num_constraint

        # Step 5: add constraint for node throughput
        cur_num_constraint = self.step5_node_throughput_constraint()
        num_constraint += cur_num_constraint

        # Step 6: add constraint for edge switch
        cur_num_constraint = self.step6_edge_switch_constraint(
            allow_partial_inference=enable_partial_inference,
            remove_redundant=remove_redundant
        )
        num_constraint += cur_num_constraint

        # Step 7: add constraint for flow over edge
        cur_num_constraint = self.step7_edge_flow_constraint()
        num_constraint += cur_num_constraint

        # Step 8: set optimization target
        source_flow_out_list: List[gp.Var] = []
        for other_idx in self.ilp_source.connected_node_indices:
            flow_var: gp.Var = self.var_flow[f"flow_source_{other_idx}"]
            source_flow_out_list.append(flow_var)
        self.ilp_model.setObjective(gp.quicksum(source_flow_out_list), GRB.MAXIMIZE)

        # return the size of the ILP problem
        self.model_initialized = True
        return num_int, num_real, num_binary, num_constraint

    def search_layout(self, max_run_time: float, early_stop_threshold: float, early_stop_time: float,
                      save_sol_path: str, save_model_path: str or None = None) -> None:
        """
        Search a layout that maximizes the max flow based on the ILP program.

        :param max_run_time: max running time allowed
        :param early_stop_threshold: a value between 0 and 1 (usually 0.98), if the solution is at least this close
                                     to upper bound, then we may early stop the optimization (see below)
        :param early_stop_time: if the solution reaches early_stop_threshold and no improvement is made in this
                                amount of time, we will early stop the optimization
        :param save_sol_path: save solution into this path
        :param save_model_path: save model into this path
        :return: None
        """
        # check input
        assert self.model_initialized, "Model should be initialized before searching for a layout!"
        assert save_sol_path.endswith(".sol"), "Solution file must end with .sol!"
        assert save_model_path is None or save_model_path.endswith(".lp"), "Model file should end with .lp!"

        # initialize optimization settings
        assert 0 <= early_stop_threshold < 1, "Early stop threshold must be in [0, 1)!"
        self.max_run_time = max_run_time
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_time = early_stop_time
        self.opt_start_time = time.time()
        self.opt_best_obj = -1
        self.opt_best_obj_found_time = self.opt_start_time
        self.opt_upper_bound = self.get_flow_upper_bound()

        # define early stopping callback function
        def early_stopping_callback(model, where):
            if where == GRB.Callback.MIP:
                best_objective = model.cbGet(GRB.Callback.MIP_OBJBST)

                # update best objective found
                current_time = time.time()
                if best_objective > self.opt_best_obj:
                    self.opt_best_obj = best_objective
                    self.opt_best_obj_found_time = current_time

                # criteria 1: if max time is reached, then we terminate the search
                if current_time - self.opt_start_time >= self.max_run_time:
                    print(f"[ILP Layout - Info] Early stop because max search time ({self.max_run_time}) is reached")
                    print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                    model.terminate()
                    return

                # criteria 2: if early stop criteria is satisfied
                if best_objective >= self.early_stop_threshold * self.opt_upper_bound:
                    # if no improvement for a long time
                    if current_time - self.opt_best_obj_found_time > self.early_stop_time:
                        print(f"[ILP Layout - Info] Early stop because the best solution found is at least "
                              f"{round(self.early_stop_threshold * 100, 1)}% optimal and no improvement is "
                              f"made in {self.early_stop_time} seconds!")
                        print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                        model.terminate()

                # criteria 3: force early stop if really very optimal
                if best_objective >= 0.995 * self.opt_upper_bound:
                    print(f"[ILP Layout - Info] Early stop because the best solution found is at least 99.5% optimal!")
                    print(f"[ILP Layout - Info] Found: {best_objective}, Upper Bound: {self.opt_upper_bound}.")
                    model.terminate()

        # solve
        print("# ----------------------------------------- Gurobi ----------------------------------------- #")
        self.ilp_model.optimize(early_stopping_callback)
        print("# ------------------------------------------------------------------------------------------ #")

        # save solution and model
        self.ilp_model.write(save_sol_path)
        if save_model_path is not None:
            self.ilp_model.write(save_model_path)

    def check_link_validity(self, from_idx: int or str, to_idx: int or str, allow_partial_inference: bool) -> bool:
        """
        Check whether the link between the two nodes is valid.

        :param from_idx: index of the input node
        :param to_idx: index of the output node
        :param allow_partial_inference: whether partial inference is allowed
        :return: whether the link is valid
        """
        assert not from_idx == "sink" and not to_idx == "source", "Invalid end points!"
        assert not (from_idx == "source" and to_idx == "sink"), "Found edge between source and sink!"
        if from_idx == "source":
            return self.ilp_nodes[to_idx].start_layer_idx == 0
        elif to_idx == "sink":
            return self.ilp_nodes[from_idx].end_layer_idx == self.model_card.num_layers
        else:
            s_j = self.ilp_nodes[to_idx].start_layer_idx
            e_i = self.ilp_nodes[from_idx].end_layer_idx
            e_j = self.ilp_nodes[to_idx].end_layer_idx
            if allow_partial_inference:
                return s_j <= e_i < e_j
            else:
                return e_i == s_j

    def load_and_verify_solution(self, save_sol_path: str, allow_partial_inference: bool) -> None:
        """
        Load a solution from the file and verify it.

        :param save_sol_path: the file that saves the solution
        :param allow_partial_inference: whether partial inference is allowed
        :return: None
        """
        assert self.cluster_loaded, "Cluster must be loaded before we can load and verify solution!"

        # load the variables into a dict
        name_2_val: Dict[str, int or float] = {}
        with open(save_sol_path, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                name, val = line.split(" ")
                name_2_val[name] = eval(val)

        # load into ilp nodes
        for node_idx, compute_node in self.ilp_nodes.items():
            compute_node: ILPNode

            # start layer index
            compute_node.start_layer_idx = round(name_2_val[f"start_{node_idx}"])
            assert 0 <= compute_node.start_layer_idx, "Bad start layer index!"
            assert is_close(compute_node.start_layer_idx, name_2_val[f"start_{node_idx}"]), \
                "Start layer index should be an int!"

            # check that only one configuration is selected
            hold_sum, k_hold_sum = 0, 0
            for layer_count in range(1, compute_node.max_num_layers + 1):
                hold_var_val = round(name_2_val[f"hold_{node_idx}_{layer_count}"])
                assert is_close(hold_var_val, name_2_val[f"hold_{node_idx}_{layer_count}"]), \
                    "Hold var should be an int!"
                assert hold_var_val == 0 or hold_var_val == 1, "Hold var must be binary!"
                hold_sum += hold_var_val
                k_hold_sum += layer_count * hold_var_val
            assert hold_sum == 1, f"Only one configuration can be selected (now {hold_sum})!"

            # end layer index
            compute_node.end_layer_idx = compute_node.start_layer_idx + k_hold_sum
            assert compute_node.end_layer_idx <= self.model_card.num_layers, "Bad end layer index!"

        # load into ilp links
        for link_name_tuple, link in self.ilp_links.items():
            link: ILPLink

            # forward edge
            if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                from_idx = link_name_tuple[0]
                to_idx = link_name_tuple[1]

                # load variables
                link.forward_flow = name_2_val[f"flow_{from_idx}_{to_idx}"]
                assert link.forward_flow >= 0 - ATOL, "Flow should be larger or equal to 0!"
                link.forward_edge_switch = round(name_2_val[f"switch_{from_idx}_{to_idx}"])
                assert is_close(link.forward_edge_switch, name_2_val[f"switch_{from_idx}_{to_idx}"]), \
                    "Switch variable should be an int!"
                assert link.forward_edge_switch == 0 or link.forward_edge_switch == 1, "Switch is binary!"
                if not from_idx == "source" and not to_idx == "sink" and allow_partial_inference:
                    link.forward_edge_cond1 = round(name_2_val[f"edge_cond1_{from_idx}_{to_idx}"])
                    link.forward_edge_cond2 = round(name_2_val[f"edge_cond2_{from_idx}_{to_idx}"])
                    assert is_close(link.forward_edge_cond1, name_2_val[f"edge_cond1_{from_idx}_{to_idx}"]), \
                        "Condition 1 should be an int!"
                    assert is_close(link.forward_edge_cond2, name_2_val[f"edge_cond2_{from_idx}_{to_idx}"]), \
                        "Condition 2 should be an int!"
                    assert link.forward_edge_cond1 == 0 or link.forward_edge_cond1 == 1, "Condition 1 is binary!"
                    assert link.forward_edge_cond2 == 0 or link.forward_edge_cond2 == 1, "Condition 2 is binary!"
                    if link.forward_edge_switch:
                        assert link.forward_edge_cond1 == 1, "Condition 1 should be 1 for switch to be True!"
                        assert link.forward_edge_cond2 == 1, "Condition 2 should be 1 for switch to be True!"

                # check that the switch can be enabled
                if link.forward_edge_switch == 1:
                    assert self.check_link_validity(from_idx=from_idx, to_idx=to_idx,
                                                    allow_partial_inference=allow_partial_inference), \
                        "Found edge that could not be set to enabled!"

                # check flow consistency
                assert link.forward_flow <= link.throughput * link.forward_edge_switch + ATOL, \
                    f"Bad flow over link from {from_idx} to {to_idx}!"

            # backward edge
            if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                from_idx = link_name_tuple[1]
                to_idx = link_name_tuple[0]

                # load variables
                link.backward_flow = name_2_val[f"flow_{from_idx}_{to_idx}"]
                assert link.backward_flow >= 0 - ATOL, "Flow should be larger or equal to 0!"
                link.backward_edge_switch = round(name_2_val[f"switch_{from_idx}_{to_idx}"])
                assert is_close(link.backward_edge_switch, name_2_val[f"switch_{from_idx}_{to_idx}"]), \
                    "Switch variable should be an int!"
                assert link.backward_edge_switch == 0 or link.backward_edge_switch == 1, "Switch is binary!"
                if not from_idx == "source" and not to_idx == "sink" and allow_partial_inference:
                    link.backward_edge_cond1 = round(name_2_val[f"edge_cond1_{from_idx}_{to_idx}"])
                    link.backward_edge_cond2 = round(name_2_val[f"edge_cond2_{from_idx}_{to_idx}"])
                    assert is_close(link.backward_edge_cond1, name_2_val[f"edge_cond1_{from_idx}_{to_idx}"]), \
                        "Condition 1 should be an int!"
                    assert is_close(link.backward_edge_cond2, name_2_val[f"edge_cond2_{from_idx}_{to_idx}"]), \
                        "Condition 2 should be an int!"
                    assert link.backward_edge_cond1 == 0 or link.backward_edge_cond1 == 1, "Condition 1 is binary!"
                    assert link.backward_edge_cond2 == 0 or link.backward_edge_cond2 == 1, "Condition 2 is binary!"
                    if link.backward_edge_switch:
                        assert link.backward_edge_cond1 == 1, "Condition 1 should be 1 for switch to be True!"
                        assert link.backward_edge_cond2 == 1, "Condition 2 should be 1 for switch to be True!"

                # check that the switch can be enabled
                if link.backward_edge_switch == 1:
                    assert self.check_link_validity(from_idx=from_idx, to_idx=to_idx,
                                                    allow_partial_inference=allow_partial_inference), \
                        "Found edge that could not be set to enabled!"

                # check flow consistency
                assert link.backward_flow <= link.throughput * link.backward_edge_switch + ATOL, \
                    f"Bad flow over link from {from_idx} to {to_idx}!"

            # forward and backward edge can not have flow at the same time
            if "source" not in link_name_tuple and "sink" not in link_name_tuple:
                assert (link.forward_flow == 0) or (link.backward_flow == 0), "Only one direction can have flow!"

        # check that flow in = flow out and flow < inference throughput for each node
        for compute_node_idx, compute_node in self.ilp_nodes.items():
            # flow in = flow out
            flow_in, flow_out = 0, 0
            for other_idx in compute_node.connected_node_indices:
                if not other_idx == "sink":
                    flow_in += name_2_val[f"flow_{other_idx}_{compute_node_idx}"]
                if not other_idx == "source":
                    flow_out += name_2_val[f"flow_{compute_node_idx}_{other_idx}"]
            assert math.isclose(flow_in, flow_out, abs_tol=ATOL), f"Flow in = {flow_in} != {flow_out} = Flow out!"

            # flow < inference throughput
            num_layers_on_node = compute_node.end_layer_idx - compute_node.start_layer_idx
            assert flow_in <= compute_node.layer_count_2_throughput[num_layers_on_node] + ATOL, \
                "Flow in should be smaller than inference throughput!"

        self.solution_loaded = True

    def generate_simulator_cluster(self, cluster_file_path: str, allow_partial_inference: bool) -> None:
        """
        Generate the cluster file and statistics file that will be used by the simulator.

        :param cluster_file_path: path to save the cluster file
        :param allow_partial_inference: whether partial inference is allowed
        :return: None
        """
        assert self.solution_loaded, "Solution must be loaded before generating simulator cluster!"

        # generate cluster file
        with open(cluster_file_path, "w") as file:
            # header notes
            file.write("# Simulator cluster file generated by ILP layout synthesizer.\n")
            file.write("\n")

            # write coordinator
            file.write(f"[Coordinator]\n")
            inbound_nic_speed: float = self.ilp_sink.machine_type.inbound_nic_speed / mbps
            outbound_nic_speed: float = self.ilp_source.machine_type.outbound_nic_speed / mbps
            file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
            file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
            file.write("\n")

            # write machine types
            file.write(f"[MachineTypes]\n")
            machine_types = list(self.machine_profiles.keys())
            machine_types.remove("SourceNode")
            machine_types.remove("SinkNode")
            file.write(f"types={machine_types}\n")
            file.write("\n")

            # write node names
            file.write("[ComputeNodes]\n")
            node_names = [f"compute_node_{self.node_idx_offset + i}" for i in range(len(self.ilp_nodes))]
            file.write(f"names={node_names}\n")
            file.write("\n")

            # write the nodes
            for node_idx, ilp_node in self.ilp_nodes.items():
                file.write(f"[compute_node_{self.node_idx_offset + node_idx}]\n")
                vram_size: float = ilp_node.machine_type.vram_size / MB
                file.write(f"vram_size={vram_size} * MB\n")
                inbound_nic_speed: float = ilp_node.machine_type.inbound_nic_speed / mbps
                file.write(f"inbound_nic_speed={inbound_nic_speed} * mbps\n")
                outbound_nic_speed: float = ilp_node.machine_type.outbound_nic_speed / mbps
                file.write(f"outbound_nic_speed={outbound_nic_speed} * mbps\n")
                disk_speed: float = ilp_node.machine_type.disk_speed / mbps
                file.write(f"disk_speed={disk_speed} * mbps\n")
                file.write(f"machine_type=\"{ilp_node.machine_type.type_name}\"\n")
                kv_cache_capacity: int = self.model_manager.get_kv_cache_capacity(
                    machine_type=ilp_node.machine_type.type_name,
                    num_on_node_layers=ilp_node.end_layer_idx - ilp_node.start_layer_idx
                )
                file.write(f"kv_cache_capacity={kv_cache_capacity}\n")
                activation_backup_capacity: int = self.model_manager.get_activation_backup_capacity(
                    machine_type=ilp_node.machine_type.type_name,
                    num_on_node_layers=ilp_node.end_layer_idx - ilp_node.start_layer_idx
                )
                file.write(f"activation_backup_capacity={activation_backup_capacity}\n")
                file.write("\n")

            # write the links
            # note that we write all links as long as the link is valid (by checking models at two endpoints)
            # find all valid links
            valid_links: Dict[Tuple[int or str, int or str], ILPLink] = {}
            for link_name_tuple, ilp_link in self.ilp_links.items():
                if not link_name_tuple[0] == "sink" and not link_name_tuple[1] == "source":
                    forward_link_valid = self.check_link_validity(from_idx=link_name_tuple[0],
                                                                  to_idx=link_name_tuple[1],
                                                                  allow_partial_inference=allow_partial_inference)
                    if forward_link_valid:
                        valid_links[link_name_tuple] = ilp_link
                if not link_name_tuple[1] == "sink" and not link_name_tuple[0] == "source":
                    backward_link_valid = self.check_link_validity(from_idx=link_name_tuple[1],
                                                                   to_idx=link_name_tuple[0],
                                                                   allow_partial_inference=allow_partial_inference)
                    if backward_link_valid:
                        valid_links[(link_name_tuple[1], link_name_tuple[0])] = ilp_link

            # remove links associated with nodes with no outbound link
            # if a node other than sink does not have any outbound link, then remove all inbound links
            # associated with node, as this node can not be used in inference
            nodes_with_outbound: Set[int or str] = set()
            for link_name_tuple in valid_links.keys():
                nodes_with_outbound.add(link_name_tuple[0])
            valid_links_with_outbound: Dict[Tuple[int or str, int or str], ILPLink] = {}
            for link_name_tuple, ilp_link in valid_links.items():
                if link_name_tuple[1] == "sink" or link_name_tuple[1] in nodes_with_outbound:
                    valid_links_with_outbound[link_name_tuple] = ilp_link
            valid_links = valid_links_with_outbound

            # remove links associated with nodes with no inbound link
            # if a node other than source does not have any inbound link, then remove all outbound links
            # associated with node, as this node can not be used in inference
            nodes_with_inbound: Set[int or str] = set()
            for link_name_tuple in valid_links.keys():
                nodes_with_inbound.add(link_name_tuple[1])
            valid_links_with_inbound: Dict[Tuple[int or str, int or str], ILPLink] = {}
            for link_name_tuple, ilp_link in valid_links.items():
                if link_name_tuple[0] == "source" or link_name_tuple[0] in nodes_with_inbound:
                    valid_links_with_inbound[link_name_tuple] = ilp_link
            valid_links = valid_links_with_inbound

            # write the valid link names
            file.write("[Links]\n")
            valid_link_names = []
            for valid_link_name_tuple in valid_links.keys():
                from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
                to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
                valid_link_names.append(f"link_{from_name}_{to_name}")
            file.write(f"names={valid_link_names}\n")
            file.write("\n")

            # write the links
            for valid_link_name_tuple, valid_link in valid_links.items():
                from_name = valid_link_name_tuple[0] if valid_link_name_tuple[0] == "source" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[0]}"
                to_name = valid_link_name_tuple[1] if valid_link_name_tuple[1] == "sink" else \
                    f"compute_node_{self.node_idx_offset + valid_link_name_tuple[1]}"
                file.write(f"[link_{from_name}_{to_name}]\n")
                file.write(f"in={from_name}\n")
                file.write(f"out={to_name}\n")
                file.write(f"latency={valid_link.latency * 1000} * MilliSec\n")
                file.write(f"bandwidth={valid_link.bandwidth / mbps} * mbps\n")
                file.write("\n")

    def save_layout_solution(self, save_path: str) -> None:
        """
        Save the layout solution found.
        Format:
        [Solution]
        name_in_cluster_file=[a list of layer ids]

        :param save_path: save path of solution file
        :return: None
        """
        assert self.solution_loaded, "Must find a solution before saving!"
        with open(save_path, "w") as file:
            file.write("[Settings]\n")
            file.write(f"offset={self.node_idx_offset}\n")
            file.write("\n")
            file.write("[Solution]\n")
            for ilp_node_idx, ilp_node in self.ilp_nodes.items():
                file.write(f"compute_node_{self.node_idx_offset + ilp_node_idx}=")
                file.write(f"{list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))}\n")

    def set_initial_layout(self, simulator: ClusterSimulator) -> float:
        """
        Load the initial model layout found by the ILP solver into the simulator.

        :param simulator: the cluster simulator to load model into
        :return: expected loading time in simulation
        """
        assert self.solution_loaded, "Must load a solution before setting initial layout for simulator!"
        assert simulator.current_time == 0, "Initial layout can only be set at the beginning!"

        max_load_time: float = 0
        for ilp_node_idx, ilp_node in self.ilp_nodes.items():
            # get the corresponding compute node in the simulator
            compute_node_name = f"compute_node_{self.node_idx_offset + ilp_node_idx}"
            compute_node = simulator.name_2_compute_node[compute_node_name]

            # get the model layers to load and corresponding loading time
            new_layers = list(range(ilp_node.start_layer_idx, ilp_node.end_layer_idx))
            new_layers_size = sum(self.model_manager.get_model_params()[ilp_node.start_layer_idx:
                                                                        ilp_node.end_layer_idx])
            loading_time = new_layers_size / compute_node.disk_speed
            max_load_time = max(max_load_time, loading_time)

            # issue load command
            simulator.issue_command_load_model(load_time=simulator.current_time,
                                               node_uid=compute_node.node_uid,
                                               new_layers=new_layers,
                                               request_uids_to_wait=[])

        # advance simulator
        max_load_time = math.ceil(max_load_time) + 1
        simulator.simulate(until=max_load_time)
        return max_load_time

    def get_flow_parameters(self) -> FlowParameters:
        """
        Get flow parameters based on the loaded cluster file.

        :return: FlowParameters
        """
        assert self.solution_loaded, "Solution must be loaded before FlowParameters can be returned!"
        return FlowParameters(token_size=self.model_card.token_size,
                              token_activation_size=self.model_card.activation_size)

    def get_query_manager_parameters(self) -> QueryManagerParameters:
        """
        Get query manager parameters based on the loaded cluster file.

        :return: QueryManagerParameters
        """
        assert self.solution_loaded, "Solution must be loaded before QueryManagerParameters can be returned!"
        return QueryManagerParameters(token_size=self.model_card.token_size,
                                      token_activation_size=self.model_card.activation_size,
                                      total_num_layers=self.model_card.num_layers)

    def get_ilp_max_flow(self) -> float:
        """
        Get the max flow found by the ILP solver.

        :return: max flow.
        """
        assert self.solution_loaded, "Solution must be loaded before ILP max flow can be returned!"
        sum_of_flow = 0
        for other_idx in self.ilp_source.connected_node_indices:
            sum_of_flow += self.ilp_links[("source", other_idx)].forward_flow
        return sum_of_flow

    def get_flow_upper_bound(self) -> float:
        """
        Get the upper bound of max flow over this cluster, which is defined as the max flow when all network
        transmissions are instant.

        :return: flow upper bound
        """
        assert self.cluster_loaded, "Cluster must be loaded before we can compute flow upper bound!"
        total_compute_throughput: float = 0
        for node_idx, compute_node in self.ilp_nodes.items():
            cur_node_max = -1
            for i in range(1, compute_node.max_num_layers + 1):
                cur_node_max = max(cur_node_max, compute_node.layer_count_2_throughput[i] * i)
            assert not cur_node_max == -1, "Bad max throughput!"
            total_compute_throughput += cur_node_max
        return total_compute_throughput / self.model_card.num_layers

    def detect_ilp_partial_inference(self) -> bool:
        """
        Detect whether the solution found by ILP solver uses partial inference in the max flow.

        :return: whether partial inference is used
        """
        assert self.solution_loaded, "Solution must be loaded before we can detect partial inference!"
        for link_name_tuple, link in self.ilp_links.items():
            if link.forward_flow > 0:
                valid_with_no_partial = self.check_link_validity(from_idx=link_name_tuple[0],
                                                                 to_idx=link_name_tuple[1],
                                                                 allow_partial_inference=False)
                if not valid_with_no_partial:
                    return True

            if link.backward_flow > 0:
                valid_with_no_partial = self.check_link_validity(from_idx=link_name_tuple[1],
                                                                 to_idx=link_name_tuple[0],
                                                                 allow_partial_inference=False)
                if not valid_with_no_partial:
                    return True
        return False
