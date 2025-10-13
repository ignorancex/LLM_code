from . import constant, converter, splitter, circuit
import numpy as np
import qiskit.quantum_info as qi
def psr(qc):
    qc = qc.assign_parameters([1] * qc.num_parameters)
    num_qubits = qc.num_qubits
    matrices = []
    Us = []
    Usm = [] # [U_{0:m-1}, U_{1:m-1}, ... U_{m-1:m-1}]
    index = 0
    qasm_gates = converter.qasm_to_qasmgates(converter.qc_to_qasm(qc))
    qcs = splitter.qasmgates_to_qcs2(qasm_gates)
    #qcs.reverse() 
    for qasmgates in qcs:
        gates = converter.qasmgates_to_gates(qasmgates)
        params_form, tensor_form = converter.gates_to_string(gates, num_qubits)
        u = circuit.U(params_form, tensor_form, index)
        if u.compare(Us) is False:
            index += 1
            u.to_matrix()
        Us.append(u)
    
    Uresult = []
    Ufront = np.eye(2**num_qubits)
    for i in range(len(Us)):
        Ufront = Us[i].to_matrix() @ Ufront
        Uleft = np.eye(2**num_qubits)
        Uright = np.eye(2**num_qubits)
        for j, u in enumerate(Us):
            if j == i:
                u.plus_params_form(constant.epsilon)
            Uleft = u.to_matrix() @ Uleft
            if j == i:
                u.plus_params_form(-constant.epsilon)
            Uright = u.to_matrix() @ Uright
        Uresult.append(Uleft @ constant.state0(num_qubits))
        Uresult.append(Uright @ constant.state0(num_qubits))
    Ufront = Ufront @ constant.state0(num_qubits)
    return Uresult

def qiskit(qc):
    Uresult = []
    num_qubits = qc.num_qubits
    for i in range(len(qc.parameters)):
        Uleft = qi.Statevector.from_instruction(qc.assign_parameters([1]*qc.num_parameters)).data
        Uright = qi.Statevector.from_instruction(qc.assign_parameters([1]*qc.num_parameters)).data
        Uresult.append(Uleft)
        Uresult.append(Uright)
def proposed_psr(qc):
    qc = qc.assign_parameters([1] * qc.num_parameters)
    num_qubits = qc.num_qubits
    matrices = []
    Us = []
    
    index = 0
    qasm_gates = converter.qasm_to_qasmgates(converter.qc_to_qasm(qc))
    qcs = splitter.qasmgates_to_qcs2(qasm_gates)
    #qcs.reverse() 
    # print(qcs)
    for qasmgates in qcs:
        gates = converter.qasmgates_to_gates(qasmgates)
        params_form, tensor_form = converter.gates_to_string(gates, num_qubits)
        u = circuit.U(params_form, tensor_form, index)
        if u.compare(Us) is False:
            index += 1
            u.to_matrix()
        Us.append(u)
    Us.reverse() # Now is [..., U_2, U_1, U_0]
    Usm = [np.eye(2**num_qubits)] 
    Uk = np.eye(2**num_qubits)
    for u in Us[1:]:
        Uk = u.matrix_form @ Uk
        Usm.append(Uk)
    # [I, U_{m-1:m-1}, ..., U_{1:m-1}]
    Usm.reverse() # Now is [U_{1:m-1}, ... U_{m-1:m-1}, I] # ignore U_{0:m-1}, m = 5, [U_{1:4}, U_{2:4}, U_{3:4}, U_{4:4}, I]

    Uresult = []
    Ufront = np.eye(2**num_qubits)
    i = 0
    while i < len(Usm):
        if Us[i].len_params() == 0:
            Uleft = Ufront @ Us[i].to_matrix() @ Usm[i]
            Uright = Ufront
        else:
            Uleft = Ufront @ Us[i].plus_params_form(constant.epsilon).to_matrix() @ Usm[i]
            Uright = Ufront @ Us[i].plus_params_form(-constant.epsilon).to_matrix() @ Usm[i]
        Uresult.append(Uleft @ constant.state0(num_qubits))
        Uresult.append(Uright @ constant.state0(num_qubits))
        Ufront = Us[i].to_matrix() @ Ufront 
        i += 1
    Ufront = Ufront @ constant.state0(num_qubits)
    return Uresult
