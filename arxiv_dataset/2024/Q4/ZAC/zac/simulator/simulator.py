import json

class Simulator():
    """calculate circuit fidelity based on code_full files."""

    def __init__(self):
        """
        Args:
        """
        # default fidelity parameter
        self.fidelity_2q_gate = 0.995
        self.fidelity_2q_gate_for_idle = 1 - (1-self.fidelity_2q_gate)/2
        self.fidelity_1q_gate = 0.9997
        self.fidelity_atom_transfer = 0.999
        self.time_coherence = 1.5e6 # us
        self.time_atom_transfer = 15 # us
        self.time_rydberg = 0.36 # us
        self.time_1q_gate = 52 # us

        self.entanglement_zone = set()
        self.n_aod = 0
        
        self.n_qubit = 0
        self.list_instrcution = []

        # data members for fidelity computation
        self.cir_fidelity = 1
        self.cir_fidelity_2q_gate = 1
        self.cir_fidelity_2q_gate_for_idle = 1
        self.cir_fidelity_1q_gate = 1
        self.cir_fidelity_atom_transfer = 1
        self.cir_fidelity_coherence = 1
        self.cir_qubit_busy_time = []
    
    def set_arch_spec(self, spec: dict):
        if "operation_fidelity" in spec:
            if "two_qubit_gate" in spec["operation_fidelity"]:
                self.fidelity_2q_gate = spec["operation_fidelity"]["two_qubit_gate"]
                self.fidelity_2q_gate_for_idle = 1 - (1-self.fidelity_2q_gate)/2
            if "single_qubit_gate" in spec["operation_fidelity"]:
                self.fidelity_1q_gate = spec["operation_fidelity"]["single_qubit_gate"]
            if "atom_transfer" in spec["operation_fidelity"]:
                self.fidelity_atom_transfer = spec["operation_fidelity"]["atom_transfer"]
            if "T" in spec["qubit_spec"]:
                self.time_coherence = spec["qubit_spec"]["T"]
            if "transfer_time" in spec["operation_duration"]:
                self.time_atom_transfer = spec["operation_duration"]["atom_transfer"] 
            if "1qGate" in spec["operation_duration"]:
                self.time_1q_gate = spec["operation_duration"]["1qGate"] 
            if "rydberg" in spec["operation_duration"]:
                self.time_rydberg = spec["operation_duration"]["rydberg"] 
        if "entanglement_zones" in spec:
            for zone in spec["entanglement_zones"]:
                for slm_spec in zone["slms"]:
                    self.entanglement_zone.add(slm_spec["id"])
        if "aods" in spec:
            self.n_aod = len(spec["aods"])

    def parse(self, code_file: str):
        with open(code_file, 'r') as f:
            result = json.load(f)
        self.list_instrcution = result["instructions"]
        if self.list_instrcution[0]["type"] == "init":
            self.n_qubit = len(self.list_instrcution[0]['init_locs'])
        else:
            assert(0)

    def process_rydberg(self, instruction, set_qubit_in_rydberg):
        list_gates = instruction["gates"]
        num_idle_qubits = len(set_qubit_in_rydberg) - 2 * len(list_gates)
        assert(num_idle_qubits >= 0)
        if len(list_gates) == 0:
            return
        # calculate the fidelity of two-qubit gates
        self.cir_fidelity_2q_gate *= pow(self.fidelity_2q_gate, len(list_gates))
        # calculate the fidelity of idle qubits affected by Rydberg laser
        # self.cir_fidelity_2q_gate_for_idle *= pow(self.fidelity_2q_gate_for_idle, num_idle_qubits)
        for i in set_qubit_in_rydberg:
            self.cir_qubit_busy_time[i] += self.time_rydberg
    
    def process_1q_gate(self, instruction):
        list_gates = instruction["gates"]
        if len(list_gates) == 0:
            return
        # calculate the fidelity of two-qubit gates
        self.cir_fidelity_1q_gate *= pow(self.fidelity_1q_gate, len(list_gates))
        # self.single_qubit_extra_time += ((len(list_gates) - 1) * self.time_1q_gate)
        # calculate the fidelity of idle qubits affected by Rydberg laser
        for gate_info in list_gates:
            self.cir_qubit_busy_time[gate_info["q"]] += self.time_1q_gate
            # self.cir_qubit_busy_time[gate_info["q"]] += self.time_1q_gate

    def process_arrangement(self, instruction, set_qubit_in_rydberg):
        # get instruction duration
        time_atom_transfer = self.time_atom_transfer * 2
        num_transfer = 0
        # collect the atoms involving in atom transfer
        list_aod_qubits = instruction["aod_qubits"]
        num_transfer += len(list_aod_qubits)
        for q in list_aod_qubits:
            self.cir_qubit_busy_time[q] += time_atom_transfer
        self.cir_fidelity_atom_transfer *= pow(self.fidelity_atom_transfer, 2*num_transfer)
        # collect qubit location
        list_qubit_end_location = instruction["end_locs"]
        for qubit_info in list_qubit_end_location:
            q_idx = qubit_info[0]
            new_array = qubit_info[1]
            if new_array in self.entanglement_zone:
                set_qubit_in_rydberg.add(q_idx)
            else:
                set_qubit_in_rydberg.remove(q_idx)
        
        for inst in instruction["insts"]:
            if inst["end_time"] - inst["begin_time"] > 200:
                print("long duration: ", inst["end_time"] - inst["begin_time"])

    def simulate(self):
        self.cir_fidelity = 1
        self.cir_fidelity_2q_gate = 1
        self.cir_fidelity_2q_gate_for_idle = 1
        self.cir_fidelity_1q_gate = 1
        self.cir_fidelity_atom_transfer = 1
        self.cir_fidelity_coherence = 1
        self.cir_qubit_busy_time = [0 for i in range(self.n_qubit)]

        # self.time_1q_gate = 25
        circuit_end_time = 0
        self.single_qubit_extra_time = 0
        set_qubit_in_rydberg = set()
        for i, instruction in enumerate(self.list_instrcution):
            if instruction["type"] == "init":
                continue
            elif instruction["type"] == "rydberg":
                self.process_rydberg(instruction, set_qubit_in_rydberg)
            elif instruction["type"] == "rearrangeJob":
                self.process_arrangement(instruction, set_qubit_in_rydberg)
            elif instruction["type"] == "1qGate":
                self.process_1q_gate(instruction)
            else:
                raise ValueError("Wrong instruction type")
            instruction_end_time = instruction["end_time"]
            if circuit_end_time < instruction_end_time:
                circuit_end_time = instruction_end_time
            # print("after {}, the fidelity is:".format(instruction["type"]))
            # print("                          cir_fidelity_2q_gate = {:10.04f},".format(self.cir_fidelity_2q_gate))
            # print("                          cir_fidelity_2q_gate_for_idle = {:10.04f},".format(self.cir_fidelity_2q_gate_for_idle))
            # print("                          cir_fidelity_atom_transfer = {:10.04f}".format(self.cir_fidelity_atom_transfer))
            # print("                          cir_qubit_busy_time = {}".format(self.cir_qubit_busy_time))
            # print("                          instruction_end_time[idx] = {}".format(self.instruction_end_time[i]))
            # input()
        # print(self.single_qubit_extra_time)
        # ! simulation directly add single qubit execution time
        circuit_end_time += self.single_qubit_extra_time
        for t in self.cir_qubit_busy_time:
            idle_t = circuit_end_time - t
            self.cir_fidelity_coherence *= (1 - idle_t/self.time_coherence)
        self.cir_fidelity = self.cir_fidelity_1q_gate * self.cir_fidelity_2q_gate * self.cir_fidelity_2q_gate_for_idle \
                            * self.cir_fidelity_atom_transfer * self.cir_fidelity_coherence
        results = { "cir_fidelity" : self.cir_fidelity,
                    "cir_fidelity_1q_gate": self.cir_fidelity_1q_gate,
                    "cir_fidelity_2q_gate": self.cir_fidelity_2q_gate,
                    "cir_fidelity_2q_gate_for_idle": self.cir_fidelity_2q_gate_for_idle,
                    "cir_fidelity_atom_transfer": self.cir_fidelity_atom_transfer,
                    "cir_fidelity_coherence": self.cir_fidelity_coherence,
                    "cir_duration": circuit_end_time}
        return results
        

