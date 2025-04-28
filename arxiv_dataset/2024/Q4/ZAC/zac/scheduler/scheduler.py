import time
import math
import rustworkx as rx

class Scheduler_mixin:
    def scheduling(self):
        """
        solve gate scheduling problem for all-commutable gate cases by graph coloring algorithm
        """
        t_s = time.time()
        self.gate_scheduling = []
        if self.has_dependency:
            if self.scheduling_strategy == "asap":
                result_scheduling = self.asap()
            else:
                result_scheduling = [[i] for i in range(len(self.g_q))]
        else:
            result_scheduling = self.graph_coloring()
        
        # TODO: handle the case that the number of gates in one layer exceed the capacity of rydberg zone
        max_gate_num = 0
        for zone in self.architecture.entanglement_zone:
            slm = self.architecture.dict_SLM[zone[0]]
            max_gate_num += (slm.n_r * slm.n_c)
            # break
        
        self.gate_scheduling_idx = []
        for gates in result_scheduling:
            if len(gates) < max_gate_num:
                self.gate_scheduling_idx.append(gates)
            else:
                num_layer = math.ceil(len(gates) / max_gate_num)
                gates_per_layer = math.ceil(len(gates) / num_layer)
                for i in range(0, len(gates), gates_per_layer):
                    self.gate_scheduling_idx.append(gates[i: i + gates_per_layer])

        self.gate_scheduling = []
        self.gate_1q_scheduling = []

        for gates in self.gate_scheduling_idx:
            tmp = [self.g_q[i] for i in gates]
            self.gate_scheduling.append(tmp)
            self.gate_1q_scheduling.append([])
            for gate_idx in gates:
                if gate_idx in self.dict_g_1q_parent:
                    for gate_1q in self.dict_g_1q_parent[gate_idx]:
                        self.gate_1q_scheduling[-1].append(gate_1q)
        
        self.runtime_analysis["scheduling"] = time.time()- t_s
        
        print("[INFO]               Time for scheduling: {}s".format(self.runtime_analysis["scheduling"]))

    def asap(self):
        # as soon as possible algorithm for self.g_q
        gate_scheduling = []
        list_qubit_time = [0 for i in range(self.n_q)]
        for i, gate in enumerate(self.g_q):
            tq0 = list_qubit_time[gate[0]]
            tq1 = list_qubit_time[gate[1]]
            tg = max(tq0, tq1)
            if tg >= len(gate_scheduling):
                gate_scheduling.append([])
            gate_scheduling[tg].append(i)

            tg += 1
            list_qubit_time[gate[0]] = tg
            list_qubit_time[gate[1]] = tg
        return gate_scheduling
    
    def graph_coloring(self):
        graph = rx.PyGraph()
        graph.add_nodes_from(list(range(self.n_q)))
        for edge in self.g_q:
            graph.add_edge(edge[0], edge[1], edge)
        edge_colors = rx.graph_misra_gries_edge_color(graph)
        max_color = 0
        for i in edge_colors:
            max_color = max(max_color, edge_colors[i])
        gate_scheduling = [[] for i in range(max_color + 1)]
        for i in range(len(self.g_q)):
            gate_scheduling[edge_colors[i]].append(i)
        return gate_scheduling