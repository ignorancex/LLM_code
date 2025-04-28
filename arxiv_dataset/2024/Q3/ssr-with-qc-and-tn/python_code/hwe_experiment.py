from type_aliases import *
from laminate import Laminate
from encoding import stack_to_state, state_to_stack
from hamiltonians import generate_loss_hamiltonian, generate_penalty_hamiltonian

import numpy as np
import h5py
from os import listdir
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit, ParameterVector


def build_hwe_circuit(num_qubits: int, num_reps: int, reverse_qubits: bool = False) -> QuantumCircuit:
    """Create hardware efficient variational circuit

    Args:
        num_qubits (int): The number of qubits
        num_reps (int): The number of repetitions in the circuit
        reverse_qubits (:obj:`bool`, optional): Reverse the qubits after creating the circuit.

    Returns:
        QuantumCircuit: The variational circuit

    """
    qc = QuantumCircuit(num_qubits)
    num_pars_rep = 2 * num_qubits - 2
    num_parameters = num_qubits + num_reps * num_pars_rep
    pars = ParameterVector('p', num_parameters)
    # print(len(pars))
    for q in range(num_qubits):
        qc.ry(pars[q], q)
    par_counter = num_qubits
    for r in range(num_reps):
        for q in range(0, num_qubits - 1, 2):  # minus 1 to exclude last qubits
            qc.cx(q, q + 1)
            qc.ry(pars[par_counter], q)
            qc.ry(pars[par_counter + 1], q + 1)
            par_counter += 2
        for q in range(1, num_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.ry(pars[par_counter], q)
            qc.ry(pars[par_counter + 1], q + 1)
            par_counter += 2
    # assert par_counter == num_parameters
    # print(len(qc.parameters),qc.num_parameters)
    return qc.reverse_bits() if reverse_qubits else qc


def hwe_pars_on_qubits(
        num_qubits: int, num_reps: int, qubits: Iterable[int]
) -> list[int]:
    """Find the parameters in the HWE circuit that are on specific qubits

    Args:
        num_qubits (int): Total number of qubits
        num_reps (int): Number of repetitions in the HWE circuit
        qubits (Iterable[int]): Qubit indices for which the parameters are found

    Returns:
        list[int]: Sorted list of the indices of the parameters on the qubits `qubits`

    """
    pars_per_rep = (2 * num_qubits - 2)

    parameter_inds = set(qubits)
    for qb in qubits:
        if num_qubits % 2 == 0 or qb != num_qubits - 1:
            parameter_inds.update(
                num_qubits + np.arange(num_reps) * pars_per_rep + qb
            )
        if num_qubits % 2 == 0 and qb not in (0, num_qubits - 1):
            parameter_inds.update(
                2 * num_qubits + np.arange(num_reps) * pars_per_rep + qb - 1
            )
        if num_qubits % 2 != 0 and qb != 0:
            parameter_inds.update(
                2 * num_qubits + np.arange(num_reps) * pars_per_rep + qb - 2
            )

    return sorted(parameter_inds)


def run_hwe(
        folder: str, identifyer: str, laminate: Laminate, target_stack: Stack,
        encoding: Encoding, dyad_ops: DyadOps, num_reps: int,
        penalty: bool = False, penalty_strength: float = 1.,
        constraint_matrix: Optional[NDArray[bool | int]] = None,
        optimizer: str = 'BFGS', x0: Optional[Sequence[float]] = None,
        optimizer_options: Optional[dict[str,Any]] = None,
        sweep_inward: bool = False, num_sweeps: int = 2,
        target_idx: Optional[int] = None, trial_num: Optional[int] = None
) -> str:
    """Run hardware-efficient VQA in order to find a stacking sequence
    that (approximately) produces the target lamination parameters

    The results are saved in an HDF5-file in the folder `folder`. The filename followed the format
    `f"{identifyer}_targ_{target_idx}_pen_{penalty_strength:.3f}_reps_{num_reps}_trial_{trial_num:04}.hdf5"`
    `where `_targ_{target_idx}` is omitted if no target index is provided and `_pen_{penalty_strength:.3f} is
    omitted if `penalty==False`. If `trial_num` is `None`, then it is set to one higher than the maximum trial number
    of the files in the folder with otherwise identical filenames.

    Note:
        This function intends to investigate how well the VQA can find the optimal solution. Because of this,
        the function does not have target lamination parameters as an argument, but a target stacking sequence
        from which the target lamination parameters are calculated. Under real-world conditions, this target
        stacking sequence is unknown, as it is the objective of the optimization. For these cases, this function
        needs to be redefined accordingly.

    Args:
        folder (str): The folder for saving the HDF5 file
        identifyer (str): A string for the start of the filename
        laminate (Laminate): The specified laminate
        target_stack (Stack): The stacking sequence from which the target lamination parameters
            are calculated.
        encoding (Encoding): A list specifying the qubit encoding of the angle indices where
            `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)
        dyad_ops (DyadOps): A list[SparsePauliOp] where `dyad_ops[angle_idx]` is the outer
            product of the state corresponding to the encoding `encoding[angle_idx]
        num_reps (int): Number of repetitions in the variational QAOA-circuit
        penalty (:obj:`bool`,optional): Whether to impose a penalty for the disorientation constraint.
            Defaults to False
        penalty_strength (:obj:float, optional): Factor to scale the weight of the penalty. Defaults to `1.`
        constraint_matrix (:obj:`NDArray[bool | int]`, optional): A numpy array of shape `(num_angles,num_angles)`
            where `constraint_matrix[s1,s2]` is `True` or `1` if the constraint violated for ply-angle indices
            `s1` and `s2` on neighboring plies, and `False` or `0` if the constraint is satisfied.
        optimizer (:obj:`str`, optional): Choice of optimizer in `scipy.optimize.minimize. Defaults to `'BFGS'`
        x0 (:obj:`NDArray[float]`, optional): Initial parameters in the variational circuit.
            If `None` (default), random values are generated.
        optimizer_options (:obj:`dict`, optional): Passed to the keyword `options` in `scipy.optimize.minimize
        sweep_inward (:obj:`bool`, optional): Whether to perform the local optimization proceeding from high to
            low (inward) or from low to high (outward) qubit indices. Defaults to `False` (outward).
        num_sweeps (:obj:,`int`): Number of sweeps of local optimizations over all qubits.
        target_idx (:obj:`int`): An integer to identify target stacking sequences by their given label. This
            label is added to the filename such that the files for different target stacking sequences can be
            distinguished.
        trial_num (:obj:`int`, optional): Label to distinguish the files of different trials with otherwise
            identical specifications. If `None` (default), then this is set to one higher than the highest
            trial number in otherwise identical filenames.

    Returns:
        str: The path to the created HDF5-files with the results

    """
    optimizer_options = {} if optimizer_options is None else optimizer_options
    num_qubits = len(encoding[0]) * laminate.num_plies
    qc = build_hwe_circuit(num_qubits, num_reps, False)
    if x0 is None:
        x0 = np.random.rand(qc.num_parameters) * 2 * np.pi

    target = laminate.parameters(target_stack)
    ham = generate_loss_hamiltonian(laminate, target, encoding, dyad_ops)
    if penalty:
        ham += generate_penalty_hamiltonian(laminate, constraint_matrix, encoding, dyad_ops, strength=penalty_strength)
    ham = ham.simplify()

    pars = x0.copy()

    results = []
    par_inds_list = []
    qb_list = []
    for sweep in range(num_sweeps):
        results_sweep = []
        par_inds_list_sweep = []
        qb_list_sweep = []
        for n in (range(num_qubits - 1)[::-1] if sweep_inward else range(num_qubits - 1)):
            par_inds = hwe_pars_on_qubits(num_qubits, num_reps, (n, n + 1))
            x0_loc = pars[par_inds]
            qc_temp = qc.assign_parameters({
                p: v for idx, (p, v) in enumerate(zip(qc.parameters, pars))
                if idx not in par_inds
            })

            def func(x):
                return Statevector(qc_temp.assign_parameters(x)).expectation_value(ham).real

            res = minimize(func, x0_loc, method=optimizer, options=optimizer_options)

            results_sweep.append(res)
            par_inds_list_sweep.append(par_inds)
            qb_list_sweep.append((n, n + 1))
            pars[par_inds] = res.x

        results.append(results_sweep)
        par_inds_list.append(par_inds_list_sweep)
        qb_list.append(qb_list_sweep)

    # create hdf5 file and fill

    filename = identifyer
    if target_idx is not None:
        filename += f"_targ_{target_idx}"
    if penalty:
        filename += f"_pen_{penalty_strength:.3f}".replace('.', '_')
    filename += f"_reps_{num_reps}"

    folder = folder.replace('\\', '/').lstrip('/')

    if trial_num is None:
        number_list = [
            int(file.rsplit(".", 1)[0].rsplit("_", 1)[-1])
            for file in listdir(folder) if file.startswith(filename)
        ]
        trial_num = 0 if len(number_list) == 0 else max(number_list) + 1
    filename += f"_trial_{trial_num:04}.hdf5"

    filepath = f"{folder}/{filename}"

    final_state = Statevector(qc.assign_parameters(pars))

    max_pars = max(max(len(inds) for inds in ps) for ps in par_inds_list)
    par_inds_array = np.array(
        [[inds + [-1] * (max_pars - len(inds)) for inds in inds_sweep] for inds_sweep in par_inds_list])
    x_array = np.array([[np.append(r.x, [np.nan] * (max_pars - len(r.x))) for r in rs] for rs in results])

    max_state = str(max(prob_dict := final_state.probabilities_dict(), key=lambda k: prob_dict[k]))
    final_stack = state_to_stack(max_state, encoding)
    final_lamination_parameters = laminate.parameters(final_stack)
    final_rmse = np.sqrt(np.sum((final_lamination_parameters - target) ** 2))
    final_exact = np.all(final_stack == target_stack)
    if constraint_matrix is not None:
        final_valid = (sum(constraint_matrix[s1, s2] for s1, s2 in zip(target_stack[:-1], target_stack[1:])) == 0)

    with h5py.File(filepath, 'w') as file:
        # properties
        props = file.create_group("properties")
        props.attrs["num_plies"] = laminate.num_plies
        props.attrs["num_angles"] = laminate.num_angles
        props.create_dataset("target_stack", data=target_stack)
        props.create_dataset("target_parameters", data=target)
        props.create_dataset("x0", data=x0)
        props.create_dataset("laminate.funcs", data=laminate.funcs)
        props.create_dataset("laminate.weights", data=laminate.weights)
        props.attrs["penalty"] = penalty
        if penalty:
            props.attrs["penalty_strength"] = penalty_strength
            props.create_dataset("constraint_matrix", data=constraint_matrix)
        props.attrs["num_reps"] = num_reps
        props.create_dataset("encoding", data=np.array(encoding))
        props.attrs["optimizer"] = optimizer
        if len(optimizer_options) > 0:
            props.attrs["optimizer_options"] = str(optimizer_options)
        props.attrs["num_sweeps"] = num_sweeps
        props.attrs["sweep_inward"] = sweep_inward

        # optimize results
        res_group = file.create_group("optimize_results")
        res_group.create_dataset("fun", data=np.array([[r.fun for r in rs] for rs in results]))
        res_group.create_dataset("x", data=x_array)
        res_group.create_dataset("nit", data=np.array([[r.nit for r in rs] for rs in results]))
        res_group.create_dataset("success", data=np.array([[r.success for r in rs] for rs in results]))
        res_group.create_dataset("status", data=np.array([[r.status for r in rs] for rs in results]))
        res_group.create_dataset("par_inds", data=par_inds_array)
        res_group.create_dataset("qubits", data=np.array(qb_list))

        # final results
        res = file.create_group("results")
        res.create_dataset("parameters", data=pars)
        res.attrs["basis_state"] = max_state
        res.create_dataset("stack", data=final_stack)
        res.create_dataset("lamination_parameters", data=final_lamination_parameters)
        res.attrs["rmse"] = final_rmse
        res.attrs["is_exact"] = final_exact
        if constraint_matrix is not None:
            res.attrs["is_valid"] = final_valid

    return filepath
