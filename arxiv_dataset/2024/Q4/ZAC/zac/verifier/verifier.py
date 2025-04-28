
class Verifier_mixin:
    def verify_scheduling(self, result_scheduling: list):
        print(f"[INFO] ZAC: Gate Scheduling Verification: Start.")
        time_gate_scheduled = [-1 for i in range(len(self.g_q))]
        success = True
        for i, stage in enumerate(result_scheduling):
            qubit_gate = [-1 for i in range(self.n_q)]
            for gate in stage:
                if time_gate_scheduled[gate] > -1:
                    success = False
                    print("[Error] Gate {} is already scheduled in stage {}, but is assigned to stage {} again.".format(gate, time_gate_scheduled[gate], i))
                time_gate_scheduled[gate] = i
                q0 = self.g_q[gate][0]
                q1 = self.g_q[gate][1]
                if qubit_gate[q0] > -1:
                    success = False
                    print("[Error] Qubit {} is already used in gate {}, but is involved in gate {} at the same stage ({}) again.".format(q0, qubit_gate[q0], gate, i))
                if qubit_gate[q1] > -1:
                    success = False
                    print("[Error] Qubit {} is already used in gate {}, but is involved in gate {} at the same stage ({}) again.".format(q1, qubit_gate[q1], gate, i))
        
        for i, t in enumerate(time_gate_scheduled):
            if t == -1:
                success = False
                print("[Error] Gate {} is not scheduled.".format(i))
        if success:
            print(f"[INFO]               Gate Scheduling Verification: Pass.")

    def verify_qubit_mapping(self, layer_id):
        print("[INFO] ZAC: Qubit Placement Verification: Start.")
        success = True
        layer = self.result_json['instructions'][layer_id]["init_locs"]
        mapped_area = [set() for i in range(len(self.architecture.dict_SLM))]
        for qubits in layer:
            if not self.architecture.is_valid_SLM(qubits[1]):
                print(f'[Error] Qubit {qubits["id"]} is not mapped on a valid SLM ({qubits[1]}).')
                success = False
            if not self.architecture.is_valid_SLM_position(qubits[1], qubits[2], qubits[3]):
                print(f'[Error] Qubit {qubits["id"]} is mapped outside the SLM  (r={qubits[2]}, c={qubits[3]}).')
                success = False
            
            coord = (qubits[3], qubits[2])
            if coord in mapped_area[qubits[1]]:
                print(f'[Error] Qubit {qubits["id"]} is overlapped with the other qubit at (r={qubits[2]}, c={qubits[3]}).')
                success = False
            mapped_area[qubits[1]].add(coord)
        if len(layer) != self.n_q:
            print(f'[Error] Not all qubits are mapped: #mapped qubit= {len(layer["init_locs"])}, #qubit={self.n_q}.')
            success = False
        if success:
            print("[INFO]               Qubit Placement Verification: Pass.")
