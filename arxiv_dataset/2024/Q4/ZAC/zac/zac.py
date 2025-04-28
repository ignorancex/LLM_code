from zac.scheduler.scheduler import Scheduler_mixin
from zac.placer.placer import Placer_mixin
from zac.animator.animator import Animator
from zac.router.router import Router_mixin
from zac.verifier.verifier import Verifier_mixin
from zac.ds.architecture import Architecture
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import time
import json
from qiskit import transpile, QuantumCircuit, qpy

class ZAC(Scheduler_mixin, Placer_mixin, Router_mixin, Verifier_mixin, Animator):
    """class to solve QLS problem."""

    def __init__(self):
        self.dir = "./result/"
        self.n_q = 0 # number of qubits
        self.n_g = 0 # number of gates
        self.architecture = None
        self.result_json = {'name': "", 'architecture_spec_path': None, 'instructions': [], "runtime": 0}
        self.runtime_analysis = {}
        self.to_verify = True
        self.trivial_placement = False
        self.routing_strategy = "maximalis_sort"
        self.scheduling_strategy = "asap"
        self.dynamic_placement = True
        self.given_initial_mapping = None
        self.has_dependency = True
        self.l2 = False
        self.use_window = True
        self.reuse = True
        self.resyn = True
        self.common_1q = 0

    def parse_setting(self, setting: dict):
        if "name" in setting:
            self.result_json['name'] = setting["name"] 
        if "dir" in setting:
            self.dir = setting["dir"]
        if "dependency" in setting:
            self.has_dependency = setting["dependency"]
        if "routing_strategy" in setting:
            self.routing_strategy = setting["routing_strategy"]
        if "trivial_placement" in setting:
            self.trivial_placement = setting["trivial_placement"]
        if "dynamic_placement" in setting:
            self.dynamic_placement = setting["dynamic_placement"]
        if "use_window" in setting:
            self.use_window = setting["use_window"]
        if "use_verifier" in setting:
            self.to_verify = setting["use_verifier"]
        if "window_size" in setting:
            self.window_size = setting["window_size"]
        if "l2" in setting:
            self.l2 = setting["l2"]
        if "reuse" in setting:
            self.reuse = setting["reuse"]
        if "scheduling" in setting:
            self.scheduling_strategy = setting["scheduling"]
        if "resyn" in setting:
            self.resyn = setting["resyn"]
    
    def set_architecture_spec_path(self, path: str):
        self.result_json['architecture_spec_path'] = path


    def set_initial_mapping(self, mapping):
        # todo: check if the given mapping is valid
        self.given_initial_mapping = mapping

    def print_setting(self):
        print("[INFO] ZAC: Setting")
        print(f"[INFO]           Result directory: {self.dir}")
        
        if self.has_dependency:
            print(f"[INFO]           Scheduling strategy: {self.scheduling_strategy}")
        else:
            print("[INFO]           Scheduling strategy: edge coloring")
        if self.trivial_placement:
            print("[INFO]           Placement strategy: trivial placement")
        elif self.given_initial_mapping != None:
            print("[INFO]           Initial placement strategy: user-defined placement")
        elif self.l2:
            print("[INFO]           Initial placement strategy: SA-based placement with L2 distance model")
        else:
            print("[INFO]           Initial placement strategy: SA-based placement with Euclidean distance model")
        if self.dynamic_placement:
            print("[INFO]           Intermediate placement strategy: minimal weighted matching")
        else:
            print("[INFO]           Intermediate placement strategy: return to intial mapping")
        if self.reuse:
            print(f"[INFO]                                         : reuse aware")
        else:
            print(f"[INFO]                                         : no reuse")

        routing_info = self.routing_strategy
        if self.use_window:
            routing_info += f" with window size {self.window_size}"
        else:
            routing_info += f" without window"
        print(f"[INFO]           Routing strategy: {routing_info}")
        if self.to_verify:
            print("[INFO]           Verifier: enable")
        else:
            print("[INFO]           Verifier: disable")


    def set_architecture(self, arch: Architecture):
        self.architecture = arch

    def set_program(self, benchmark: str):
        # assume program is a iterable of pairs of qubits in 2Q gate
        # assume that the qubit indices used are consecutively 0, 1, ...
        self.g_q = []
        self.dict_g_1q_parent = {-1: []}
        print("[INFO] ZAC: Parse Circuit")
        file_type = benchmark.split('.')[-1]
        if file_type == "qasm" or file_type == 'qpy':
            n_single_qubit_gate = 0
            if file_type == "qasm":
                with open(benchmark, 'r') as f:
                    qasm_str = f.read()
                    circuit = QuantumCircuit.from_qasm_str(qasm_str)
            else:
                # for qcrank circuit
                with open(benchmark, 'rb') as f:
                    circuit = qpy.load(f)[0]
                self.common_1q = 19
            cz_circuit = circuit
            if self.resyn:
                cz_circuit = transpile(circuit, basis_gates=["cz", "id", "u2", "u1", "u3"],
                                        optimization_level=3,
                                        seed_transpiler=0)
            # print(cz_circuit)
            self.n_q = cz_circuit.num_qubits
            list_qubit_last_2q_gate = [-1 for i in range(0, self.n_q)]
            register_idx = dict()
            idx_begin = 0
            for i, qubit in enumerate(cz_circuit.qubits):
                if qubit._register == None:
                    qubit._index = i
                elif qubit._register not in register_idx:
                    register_idx[qubit._register] = idx_begin
                    idx_begin += qubit._register.size
            instruction = cz_circuit.data
            for ins in instruction:
                if ins.operation.num_qubits == 2:
                    offset = 0
                    if ins.qubits[0]._register != None:
                        offset = register_idx[ins.qubits[0]._register]
                    q0 = offset + ins.qubits[0]._index
                    offset = 0
                    if ins.qubits[1]._register != None:
                        offset = register_idx[ins.qubits[1]._register]
                    q1 = offset + ins.qubits[1]._index
                    list_qubit_last_2q_gate[q0] = len(self.g_q)
                    list_qubit_last_2q_gate[q1] = len(self.g_q)
                    if q0 < q1:
                        self.g_q.append([q0, q1])
                    else:
                        self.g_q.append([q1, q0])
                elif ins.operation.name != "measure" and ins.operation.name != "barrier":
                    offset = 0
                    if ins.qubits[0]._register != None:
                        offset = register_idx[ins.qubits[0]._register]
                    q0 = offset + ins.qubits[0]._index
                    if list_qubit_last_2q_gate[q0] not in self.dict_g_1q_parent:
                        self.dict_g_1q_parent[list_qubit_last_2q_gate[q0]] = []    
                    self.dict_g_1q_parent[list_qubit_last_2q_gate[q0]].append((ins.operation.name, q0))
                    n_single_qubit_gate += 1
            # print(self.g_q)
        elif  file_type == "txt":
            # graph instance 
            with open(benchmark, 'r') as f:
                self.g_q = eval(f.read())
            for gate in self.g_s:
                self.n_q = max(gate[0], self.n_q)
                self.n_q = max(gate[1], self.n_q)
            self.n_q += 1
            self.dict_g_1q_parent[-1] = [("h", i) for i in range(self.n_q)]
                
        elif  file_type == "json":
            # special case for parsing olsq-dpqa
            with open(benchmark, 'r') as f:
                smt_program = json.load(f)
                layers = smt_program["layers"]
                for l in layers:
                    for gate in l["gates"]:
                        if gate["q0"] < gate["q1"]:
                            self.g_q.append([gate["q0"], gate["q1"]])
                        else:
                            self.g_q.append([gate["q1"], gate["q0"]])
            for gate in self.g_s:
                self.n_q = max(gate[0], self.n_q)
                self.n_q = max(gate[1], self.n_q)
            self.n_q += 1
        else:
            raise ValueError
    
        self.n_g = len(self.g_q)
        self.g_s = tuple(['CRZ' for _ in range(self.n_g)])
        print("[INFO]           number of qubits: {}".format(self.n_q))
        print("[INFO]           number of two-qubit gates: {}".format(len(self.g_q)))
        print("[INFO]           number of single-qubit gates: {}".format(n_single_qubit_gate))
        

    def solve(self, save_file: bool = True):
        self.code_filename = self.dir + f"code/{self.result_json['name']}_code.json"
        # member to hold intermedite results
        self.gate_scheduling = None
        self.gate_scheduling_idx = None
        self.gate_1q_scheduling = None
        self.reuse_qubit = None
        self.qubit_mapping = []

        print("[INFO] ZAC: A compiler for neutral atom-based compute-store architecture")
        self.print_setting()
        # todo: check if the program input is valid, i.e., #q < #p
        t_s = time.time()
        # gate scheduling with graph coloring
        print("[INFO] ZAC: Run scheduling")
        self.scheduling()

        if self.reuse:
            self.collect_reuse_qubit()
        else:
            self.reuse_qubit = [set() for i in range(len(self.gate_scheduling))]
        # print("result_scheduling")
        # print(result_scheduling)
        
        self.place_qubit_initial()
        print("[INFO]               Time for initial placement: {}s".format(self.runtime_analysis["initial placement"]))
        self.place_qubit_intermedeiate()
        print("[INFO]               Time for intermediate placement: {}s".format(self.runtime_analysis["intermediate placement"]))
        self.route_qubit()
        self.runtime_analysis["total"] = time.time()- t_s
        print("[INFO]               Time for routing: {}s".format(self.runtime_analysis["routing"]))
        print("[INFO] ZAC: Toal Time: {}s".format(self.runtime_analysis["total"]))
        if save_file:
            if not self.dir:
                self.dir = "./result/"
            self.code_filename = self.dir + f"code/{self.result_json['name']}_code.json"
            with open(self.code_filename, 'w') as f:
                json.dump(self.result_json, f)
            with open(self.dir + f"time/{self.result_json['name']}_time.json", 'w') as f:
                json.dump(self.runtime_analysis, f)
    
        if self.to_verify:
            print("[INFO] ZAC: Start Verification")
            self.verify_scheduling(self.gate_scheduling_idx)
            self.verify_qubit_mapping(0)
    
        return self.result_json

    def collect_reuse_qubit(self):
        """
        collect qubits that will remain in Rydberg zone between two Rydberg stages
        """
        self.reuse_qubit = []
        qubit_is_used = [[-1 for i in range(self.n_q)] for j in range(len(self.gate_scheduling))]
        for gate_idx, gate in enumerate(self.gate_scheduling[0]):
            for q in gate:
                qubit_is_used[0][q] = gate_idx
        
        extra_reuse_qubit = 0
        for i in range(1, len(self.gate_scheduling)):
            # print("previous gate")
            # print(self.gate_scheduling[i - 1])
            # print("current gate")
            # print(self.gate_scheduling[i])
            # m_j_k = gate j can use qubit of gate k
            self.reuse_qubit.append(set())
            matrix = [[0 for k in range(len(self.gate_scheduling[i - 1]))] for j in range(len(self.gate_scheduling[i]))] 
            for gate_idx, gate in enumerate(self.gate_scheduling[i]):
                if qubit_is_used[i - 1][gate[0]] != -1 and qubit_is_used[i - 1][gate[0]] == qubit_is_used[i - 1][gate[1]]:
                    self.reuse_qubit[-1].add(gate[0])
                    self.reuse_qubit[-1].add(gate[1])
                    # print("YYYYYY")
                else:
                    for q in gate:
                        if qubit_is_used[i - 1][q] > -1:
                            matrix[gate_idx][qubit_is_used[i - 1][q]] = 1
                            extra_reuse_qubit += 1
                for q in gate:
                    qubit_is_used[i][q] = gate_idx
            # print(matrix)
            sparse_matrix = csr_matrix(matrix)
            matching = maximum_bipartite_matching(sparse_matrix, perm_type='column')
            for gate_idx, reuse_gate in enumerate(matching):
                if reuse_gate == -1:
                    continue
                extra_reuse_qubit -= 1
                gate = self.gate_scheduling[i][gate_idx]
                for q in gate:
                    if qubit_is_used[i - 1][q] == reuse_gate:
                        self.reuse_qubit[-1].add(q)
        #     print("cur_reuse_qubit:", extra_reuse_qubit)
        # print("extra_reuse_qubit: ", extra_reuse_qubit)
        assert(extra_reuse_qubit >= 0)
        self.extra_reuse_qubit = extra_reuse_qubit
            # print("reuse qubit")
            # print(self.reuse_qubit[-1])
            # input()
        self.reuse_qubit.append(set())
    




            
    
    
