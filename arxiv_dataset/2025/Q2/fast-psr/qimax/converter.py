
import numpy as np
import re, qiskit
from . import utilities, gate, tensor
from .constant import constant_gate
def qc_to_qasm(qc):
    from qiskit.qasm2 import dumps 
    return dumps(qc)

def qasm_to_qasmgates(qasm):
    """_summary_

    Args:
        qasm (_type_): _description_

    Returns:
        _type_: _description_
    """
    gates = qasm.split('\n')[3:-1]
    qasm_gates = []
    for g in gates:
        g = g[:-1].split(' ')
        # print(gate)
        indices = re.findall(r'\d+', g[1])
        indices = [int(index) for index in indices]
        matches = re.match(r'([a-z]+)(\(\d+\.\d+\))?', g[0])
        # Extract the function name and value from the matches
        name = matches.group(1).upper()
        name = name[1:] if name[0] == 'C' else name
        if matches.group(2) is None:
            param = -999
        else:
            param = float(matches.group(2)[1:-1])
        qasm_gates.append((name, param, indices))
    return qasm_gates

def parse_qasm(qasm_code):
    instructions = []
    lines = qasm_code.splitlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("//") or line == "":
            continue  # Ignore comments and empty lines
        
        match = re.match(r'([a-zA-Z]+)\s*(\(.*?\))?\s+([q0-9,\[\]]+);?', line)
        if match:
            name = match.group(1)
            param = match.group(2)
            indices = match.group(3)
            
            param = float(param.strip('()').split(',')[0]) if param else -999
            indices = [int(i) for i in re.findall(r'\d+', indices)]
            
            instructions.append((name.upper(), param, indices))
    
    return instructions[2:]


def get_wires_of_gate(gate: qiskit.circuit.Gate):
    """Get index bit that gate act on

    Args:
        - gate (qiskit.circuit.Gate): Quantum gate

    Returns:
        - List[int]: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register._index)
    return list_wire

def qcs_to_gatess(qcs):
    gatess = []
    for qc in qcs:
        gates = []
        for gate in qc.data:
            if gate.params == []:
                params = -999
            else:
                params = gate.params[0]
            gates.append((gate.name.upper(), params, get_wires_of_gate(gate)))
        gatess.append(gates)
    return gatess

def gatess_to_str(gatess):
    strs = []
    strs.append(str(len(gatess)))
    for gates in gatess:
        # Consider a U include n gates
        for gate1 in gates:
            if gate1 == 0:
                continue
            gate_matrix = constant_gate[gate1[0]]
            if gate1[0] in ['I', 'RZ']:
                # Sparse
                if gate1[0] in ['I']:
                    gate_data = [0, gate1[2][0], (gate_matrix[0][0]), (gate_matrix[1][1])]
                if gate1[0] in ['RZ']:
                    param = gate1[1]
                    gate_data = [0, gate1[2][0], gate_matrix(param)[0][0], gate_matrix(param)[1][1]]
            if gate1[0] in ['RX', 'RY', 'H']:
                # Dense
                if gate1[0] in ['RX', 'RY']:
                    param = gate1[1]
                    gate_data = [1, gate1[2][0], gate_matrix(param)[0][0] + gate_matrix(param)[0][1],
                            gate_matrix(param)[1][0] + gate_matrix(param)[1][1]]
                if gate1[0] in ['H']:
                    gate_data = [1, gate1[2][0], gate_matrix[0][0] + gate_matrix[0][1],
                            gate_matrix[1][0] + gate_matrix[1][1]]
            if gate1[0] in ['CX']:
                gate_data = [2, str(gate1[2][0]), (gate1[2][1])]
            for i in range(len(gate_data)):
                strs.append(gate_data[i])
            strs.append('xxxxx\n')
    return strs


def gatess_to_gatess_with_I(gatess1, num_qubits):
    gatess = [[0 for _ in range(num_qubits)] for _ in range(len(gatess1))]
    for index, gates in enumerate(gatess1):
        slots = np.ones(num_qubits)
        # print('---')
        # print(gates)
        for gate in gates:
            if gate[0] == 'CX':
                gatess[index][0] = ('CX', gate[1], gate[2])
                slots = np.zeros(num_qubits)
            else:
                slots[gate[2][0]] = 0
                gatess[index][gate[2][0]] = (gate[0], gate[1], gate[2])

        for j, is_active in enumerate(slots):
            if is_active:
                gatess[index][j] = (['I', -999, [j]])
    return gatess



def qasmgates_to_qcs(gates: list) -> list[qiskit.QuantumCircuit]:
    """_summary_

    Args:
        gates (list): _description_

    Returns:
        list[qiskit.QuantumCircuit]: _description_
    """
    qcs = []
    sub_qc = []
    counter = 0
    active_2qubit = 0 # 0 mean dactive, 1 mean active, 2 mean break instruction
    num_qubits = max(max(gates, key=lambda x: max(x[2]))[2]) + 1
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


def qasmgates_to_gates(qasmgates: list[str]):
    """_summary_

    Args:
        qasmgates (list[str]): _description_

    Returns:
        _type_: _description_
    """
    gates = []
    for name, param, indices in qasmgates:
        gates.append(gate.Gate(name, indices = indices, param = param))
    return gates
  
def gates_to_string(gates: list[gate.Gate], num_qubits: int) -> np.ndarray:
    """Return the matrix representation of a circuit 
    composed of a list of gates acting on `num_qubits` qubits.

    Args:
        gates (list[Gate]): List of gates in the circuit
        num_qubits (int): # of qubits in the system

    Returns:
        np.ndarray: Matrix representation of the circuit
    """
    # Example: {CNOT_1,3, RY_2(\pi)} act on 5 qubits circuit
    # Init tensor form as : I \otimes I \otimes ... \otimes I (num_qubits otimes)
    # and params form as [0, 0, ..., 0] (num_qubits times)
    tensor_form = [['I'] * num_qubits]
    params_form = [-999] * num_qubits
    # Update tensor form and params form based on gates
    # => tensor_form = [['I', 'M', 'RY', 'I', 'I'], ['I', 'P', 'RY', 'X', 'I']]
    # => params_form = [0, 0, np.pi, 0, 0]
    for g in gates:
        if g.param is not None:
            if len(g.indices) == 1:
                params_form[g.indices[0]] = g.param
            else:
                params_form[g.indices[1]] = g.param
        if len(g.indices) == 1:
            tensor_form[0][g.indices[0]] = g.name
        else:
            tensor_form = utilities.duplicate_xss(tensor_form)
            tensor_form[0][g.indices[0]] = 'M' # P0
            tensor_form[1][g.indices[0]] = 'P' # P3
            tensor_form[1][g.indices[1]] = g.name
    return params_form, tensor_form

def string_to_matrix(params_form: np.ndarray, tensor_form: np.ndarray) -> np.ndarray:
    """Convert the tensor form and params form to the matrix representation of the circuit.

    Args:
        params_form (np.ndarray): lisr of parameters
        tensor_form (np.ndarray): list of gates (only gate names, no parameters)

    Returns:
        np.ndarray: matrix representation of the circuit
    """
    # Convert tensor form and params form to matrix
    # => [['I', 'M', 'RY', 'I', 'I'], ['I', 'P', 'RY', 'X', 'I']]
    # -> I \otimes M \otimes RY(np.pi) \otimes I \otimes I
    #  + I \otimes P \otimes RY(np.pi) \otimes X \otimes I
    matrices = [np.array([1])]*len(tensor_form)
    for i in range(len(tensor_form[0])):
        for j in range(len(tensor_form)):
            #print(f'A{tensor_form[j][i]}')
            if len(tensor_form[j][i]) >= 2:
                matrices[j] = getattr(tensor, f'A{tensor_form[j][i]}')(matrices[j], params_form[i])
            else:
                matrices[j] = getattr(tensor, f'A{tensor_form[j][i]}')(matrices[j])
    return sum(matrices)
