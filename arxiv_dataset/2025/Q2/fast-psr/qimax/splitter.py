import qiskit, re
import numpy as np
from . import utilities


from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

def qc_to_qcs(qc):
	def split_layer_into_subcircuits(layer_ops, num_qubits):
		"""
		Given a list of DAG operations (from one layer), split them into 
		depth-1 QuantumCircuits such that each circuit satisfies:
		- It contains at most one one-qubit rotation gate (rx, ry, or rz)
			possibly combined with other one-qubit (non-parameterized) gates,
			OR
		- It contains a single two-qubit gate (e.g. a cx gate).
		
		Parameters:
		layer_ops: list of DAGOpNode objects from one layer.
		num_qubits: number of qubits in the original circuit.
		
		Returns:
		A list of QuantumCircuit objects (each depth-1 and meeting the constraints).
		"""
		subcircuits = []
		
		# ----- 1. Handle multi-qubit gates (e.g. cx) -----
		# Each two-qubit gate (or in general, any gate acting on >1 qubit)
		# must be isolated.
		for op in layer_ops:
			if len(op.qargs) > 1:
				qc_temp = QuantumCircuit(num_qubits)
				qc_temp.append(op.op, op.qargs, op.cargs)
				subcircuits.append(qc_temp)
		
		# ----- 2. Handle one-qubit gates -----
		one_qubit_ops = [op for op in layer_ops if len(op.qargs) == 1]
		rotation_names = ['rx', 'ry', 'rz']
		
		# Partition one-qubit ops into rotation and non-rotation gates.
		rotation_ops = []
		non_rotation_ops = []
		for op in one_qubit_ops:
			if op.op.name in rotation_names:
				rotation_ops.append(op)
			else:
				non_rotation_ops.append(op)
		
		# We want to group one-qubit ops so that each resulting circuit 
		# has at most one rotation.
		# Here’s one simple strategy:
		#   (a) For each rotation gate, create a circuit that includes that rotation.
		#       (Optionally, add any non-rotation gates that act on different qubits.)
		#   (b) Then, combine any leftover non-rotation gates into a single circuit.
		
		# (a) Process each rotation op:
		used_non_rotation = set()  # to mark non-rotation ops we assign along with a rotation
		
		for rot in rotation_ops:
			qc_temp = QuantumCircuit(num_qubits)
			qc_temp.append(rot.op, rot.qargs, rot.cargs)
			# Add any non-rotation op that is on a different qubit than the rotation.
			# (In a DAG layer, these should be disjoint—but we check just in case.)
			for i, op in enumerate(non_rotation_ops):
				if i not in used_non_rotation:
					if op.qargs[0] != rot.qargs[0]:
						qc_temp.append(op.op, op.qargs, op.cargs)
						used_non_rotation.add(i)
			subcircuits.append(qc_temp)
		
		# (b) For any remaining non-rotation ops not merged above, 
		# group them together in one circuit.
		remaining_non_rotation = [op for i, op in enumerate(non_rotation_ops) if i not in used_non_rotation]
		if remaining_non_rotation:
			qc_temp = QuantumCircuit(num_qubits)
			for op in remaining_non_rotation:
				qc_temp.append(op.op, op.qargs, op.cargs)
			subcircuits.append(qc_temp)
		
		return subcircuits


	# Convert the circuit to a DAG and extract its layers.
	dag = circuit_to_dag(qc)
	layers = list(dag.layers())

	qcs = []
	for layer in layers:
		# layer['graph'].op_nodes() returns the operations in that layer.
		layer_ops = list(layer['graph'].op_nodes())
		subcirs = split_layer_into_subcircuits(layer_ops, qc.num_qubits)
		qcs.extend(subcirs)
	return qcs




def qasm_to_qasmgates(qc_qasm):
    gates = qc_qasm.split('\n')[3:-1]
    qasm_gates = []
    for gate in gates:
        gate = gate[:-1].split(' ')
        # print(gate)
        indices = re.findall(r'\d+', gate[1])
        indices = [int(index) for index in indices]
        matches = re.match(r'([a-z]+)(\(\d+\.\d+\))?', gate[0])
        # Extract the function name and value from the matches
        name = matches.group(1).upper()
        if matches.group(2) is None:
            param = -999
        else:
            param = float(matches.group(2)[1:-1])
        qasm_gates.append((name, param, indices))
    return qasm_gates

def qasmgates_to_qcs2(gates: list) -> list[qiskit.QuantumCircuit]: 
    """Add only 1 param and at much one control in one layer

    Args:
        gates (list): _description_

    Returns:
        list[qiskit.QuantumCircuit]: _description_
    """
    qcs = []
    sub_qc = []
    counter = 0
    active_2qubit = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    active_1param = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    num_qubits = max(max(gates, key=lambda x: max(x[2]))[2]) + 1
    slots = np.zeros(num_qubits)
    for name, param, indices in gates: 
        if len(indices) == 2:
            active_2qubit += 1
        if param != -999:
            active_1param += 1
        slots = utilities.update_slot(slots, indices)
        if any(slot > 1 for slot in slots) or active_2qubit == 2 or active_1param == 2:
            qcs.append(sub_qc)
            #print(f"Append sub_qc {len(qcs)}")
            active_2qubit = 0
            active_1param = 0
            sub_qc = []
            sub_qc.append((name, param, indices))
            #print(f"Append {name}")
            counter += 1
            if len(indices) == 2:
                active_2qubit += 1
                #print(f"Active 2qubit {active_2qubit}")
            if param != -999:
                active_1param += 1
                #print(f"Active 1param {active_1param}")
            slots = np.zeros(num_qubits)
            slots = utilities.update_slot(slots, indices)
            gates = gates[counter:]
            counter = 0
        else:
            sub_qc.append((name, param, indices))
            #print(f"Append {name}")
            counter += 1
        if counter >= len(gates):
            qcs.append(sub_qc)
            return qcs
    return qcs


def qasmgates_to_qcs3(gates: list) -> list[qiskit.QuantumCircuit]: 
    """Add only 1 param in one layer

    Args:
        gates (list): _description_

    Returns:
        list[qiskit.QuantumCircuit]: _description_
    """
    qcs = []
    sub_qc = []
    counter = 0
    active_1param = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    num_qubits = max(max(gates, key=lambda x: max(x[2]))[2]) + 1
    slots = np.zeros(num_qubits)
    for name, param, indices in gates: 
        if param != -999:
            active_1param += 1
        slots = utilities.update_slot(slots, indices)
        if any(slot > 1 for slot in slots) or active_1param == 2:
            qcs.append(sub_qc)
            #print(f"Append sub_qc {len(qcs)}")
            active_2qubit = 0
            active_1param = 0
            sub_qc = []
            sub_qc.append((name, param, indices))
            #print(f"Append {name}")
            counter += 1
            if param != -999:
                active_1param += 1
                #print(f"Active 1param {active_1param}")
            slots = np.zeros(num_qubits)
            slots = utilities.update_slot(slots, indices)
            gates = gates[counter:]
            counter = 0
        else:
            sub_qc.append((name, param, indices))
            #print(f"Append {name}")
            counter += 1
        if counter >= len(gates):
            qcs.append(sub_qc)
            return qcs
    return qcs


def qasmgates_to_qcs4(gates: list) -> list[qiskit.QuantumCircuit]: 
    """Add only 1 param or one control in one layer

    Args:
        gates (list): _description_

    Returns:
        list[qiskit.QuantumCircuit]: _description_
    """
    qcs = []
    sub_qc = []
    temp_noncx = []
    active_2qubit = 0 # 0 mean dactive, 1 mean break instruction
    active_1param = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    num_qubits = max(max(gates, key=lambda x: max(x[2]))[2]) + 1
    slots = np.zeros(num_qubits)
    for name, param, indices in gates: 
        if name == 'CX':
            sub_qc.append((name, param, indices))
            active_2qubit = True
    return qcs



def operatoring(instructors):
    """Construct operators from the list of operators and list of xoperators
    """
    is_cx_first = False
    operators = []
    operator = []
    operator_temp = []
    xoperator = []
    xoperator_temp = []
    xoperators = []
    
    if instructors[0][0] == "CX":
        is_cx_first = True
    num_qubits = max(max(instructors, key=lambda x: max(x[2]))[2]) + 1
    barriers_cx = [0] * num_qubits
    barriers_noncx = [0] * num_qubits
    index_u_cx = 0
    index_u_noncx = 0
    index_u_next_cx = 0
    index_u_next_noncx = 0
    break_noncx = False
    j = 0
    gatess = [[0 for _ in range(num_qubits)] for _ in range(len(instructors))]
    while len(instructors) > 0:
        gate, param, index = instructors[0]
        if gate == "CX":
            barriers_cx[index[0]] += 1
            barriers_cx[index[1]] += 1
            gatess[index_u_cx][0] = instructors.pop(0)
            index_u_next_noncx = index_u_cx + 1
        else:
            if barriers_noncx[index[0]] == 0 and barriers_cx[index[0]] == 0:
                barriers_noncx[index[0]] += 1
                gatess[index_u_noncx][index[0]] = instructors.pop(0)
                if sum(barriers_noncx) >= num_qubits and np.all(barriers_noncx):
                    index_u_noncx = index_u_next_noncx
                    barriers_cx = [0] * num_qubits
                    barriers_noncx = [0] * num_qubits
            else:
                operator_temp.append(instructors.pop(0))
    

    return gatess








def qasmgates_to_qcs(gates: list) -> list[qiskit.QuantumCircuit]: 
    qcs = []
    sub_qc = []
    counter = 0
    active_2qubit = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    num_qubits = max(max(gates, key=lambda x: max(x[2]))[2]) + 1
    print(num_qubits)
    slots = np.zeros(num_qubits)
    for name, param, indices in gates: 
        if len(indices) == 2:
            active_2qubit += 1
        
        slots = utilities.update_slot(slots, indices)
        if any(slot > 1 for slot in slots) or active_2qubit == 2:
            qcs.append(sub_qc)
            active_2qubit = 0
            sub_qc = []
            sub_qc.append((name, param, indices))
            if len(indices) == 2:
                active_2qubit += 1
            slots = np.zeros(num_qubits)
            slots = utilities.update_slot(slots, indices)
            gates = gates[counter + 1:]
            counter = 0
        else:
            sub_qc.append((name, param, indices))
            counter += 1
            if counter >= len(gates):
                qcs.append(sub_qc)
                return qcs
    return qcs

def qc_to_qcs_noncontrol2(qc: qiskit.QuantumCircuit) -> list[qiskit.QuantumCircuit]: 
    qcs = []
    sub_qc = qiskit.QuantumCircuit(qc.num_qubits)
    counter = 0
    active_2qubit = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    slots = np.zeros(qc.num_qubits)
    for _, qargs, _ in qc.data: 
        indices = utilities.get_qubit_indices(qargs)
        if len(indices) == 2:
            active_2qubit += 1
        
        slots = utilities.update_slot(slots, indices)
        if any(slot > 1 for slot in slots) or active_2qubit == 2:
            qcs.append(sub_qc)
            active_2qubit = 0
            sub_qc = qiskit.QuantumCircuit(qc.num_qubits)
            sub_qc.append(qc[counter][0], qc[counter][1])
            if len(indices) == 2:
                active_2qubit += 1
            slots = np.zeros(qc.num_qubits)
            slots = utilities.update_slot(slots, utilities.get_qubit_indices(qargs))
            qc.data = qc.data[counter + 1:]
            counter = 0
        else:
            sub_qc.append(qc[counter][0], qc[counter][1])
            counter += 1
            if counter >= len(qc.data):
                qcs.append(sub_qc)
                return qcs
    return qcs

def qc_to_qcs_noncontrol(qc: qiskit.QuantumCircuit) -> list[qiskit.QuantumCircuit]:    
    """Split n-depth circuit into n sub-circuits, each sub-circuit does not have > 2 2-qubit gates

    Args:
        qc (qiskit.QuantumCircuit): Original circuit

    Returns:
        list[qiskit.QuantumCircuit]: Splitted circuit depth by depth
    """
    def valid_forward(qc: qiskit.QuantumCircuit, x) -> bool:
        count = 0
        qc.append(x[0], x[1])
        for _, qargs, _ in qc.data:
            if len(qargs) == 2:
                count += 1
        if qc.depth() == 1 and count < 2:
            return True
        return False
    depth = qc.depth()
    qcs = []
    if depth < 0:
        raise "The depth must be >= 0"
    sub_qc = qiskit.QuantumCircuit(qc.num_qubits)
    counter = 0 
    while(True):
        valid = valid_forward(sub_qc.copy(), qc[counter])
        if valid:
            sub_qc.append(qc[counter][0], qc[counter][1])
            counter += 1
        if valid is False or counter == len(qc.data):
            qc.data = qc.data[counter:]
            qcs.append(sub_qc)
            counter = 0
            sub_qc = qiskit.QuantumCircuit(qc.num_qubits)
        if len(qc.data) == 0:
            break
    return qcs



# def qc_to_qcs(qc: qiskit.QuantumCircuit) -> list[qiskit.QuantumCircuit]:    
#     """Split n-depth circuit into n sub-circuits

#     Args:
#         qc (qiskit.QuantumCircuit): Original circuit

#     Returns:
#         list[qiskit.QuantumCircuit]: Splitted circuit depth by depth
#     """
#     def look_forward(qc: qiskit.QuantumCircuit, x):
#         qc.append(x[0], x[1])
#         return qc
#     depth = qc.depth()
#     qcs = []
#     if depth < 0:
#         raise "The depth must be >= 0"
#     for _ in range(depth):
#         qc1 = qiskit.QuantumCircuit(qc.num_qubits)
#         counter = 0
#         if qc.depth() == 1:
#             qcs.append(qc)
#             return qcs
#         for i in range(len(qc)):
#             qc1.append(qc[i][0], qc[i][1])
#             counter += 1
#             if qc1.depth() == 1 and look_forward(qc1.copy(), qc[i+1]).depth() > 1:
#                 qc.data = qc.data[counter:]
#                 qcs.append(qc1)
#                 counter = 0
#                 break
#     return qcs