from type_aliases import *
from laminate import Laminate
from encoding import stack_to_state
from hamiltonians import generate_loss_hamiltonian,generate_penalty_hamiltonian

import numpy as np
import h5py
from os import listdir
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QAOAAnsatz


def optimize_result_to_hdf5(file: h5py.File, result: dict) -> None:
    """Add the content of the results from `scipy.optimize.minimize` to an HDF5 file

    The function creates a new group `optimize_result` and add numpy arrays as datasets
    and other values as attributes to this group.

    Args:
        file (h5py.File): The HDF5-file
        result (dict): The `OptimizeResult`-instance resulting from `scipy.optimize.minimize`,
            converted to a `dict`.
    """
    file.create_group('optimize_result')
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            file['optimize_result'].create_dataset(key, data=val)
        else:
            file['optimize_result'].attrs[key] = val


def run_qaoa(
        folder: str, identifyer: str, laminate: Laminate, target_stack: Stack,
        encoding: Encoding, dyad_ops: DyadOps, num_reps: int,
        penalty: bool = False, penalty_strength: float = 1., constraint_matrix: Optional[NDArray[bool | int]] = None,
        optimizer: str = 'BFGS', x0: Optional[Sequence[float]] = None, optimizer_options: dict[str, Any] = None,
        target_idx: Optional[int] = None, trial_num: Optional[int] = None
) -> str:
    r"""Run QAOA in order to find a stacking sequence that (approximately) produces the target lamination parameters

    The results are saved in an HDF5-file in the folder `folder`. The filename followed the format
    `f"{identifyer}_targ_{target_idx}_pen_{penalty_strength:.3f}_reps_{num_reps}_trial_{trial_num:04}.hdf5"`
    `where `_targ_{target_idx}` is omitted if no target index is provided and `_pen_{penalty_strength:.3f} is
    omitted if `penalty==False`. If `trial_num` is `None`, then it is set to one higher than the maximum trial number
    of the files in the folder with otherwise identical filenames.

    Note:
        This function intends to investigate how well QAOA can find the optimal solution. Because of this,
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
    # qb_per_qd = len(encoding[0])
    # num_qubits = laminate.num_plies * qb_per_qd
    if x0 is None:
        x0 = np.random.rand(2 * num_reps)

    target = laminate.parameters(target_stack)
    ham = generate_loss_hamiltonian(laminate, target, encoding, dyad_ops)
    if penalty:
        ham += generate_penalty_hamiltonian(laminate, constraint_matrix, encoding, dyad_ops, strength=penalty_strength)
    ham = ham.simplify()
    qc = QAOAAnsatz(ham, reps=num_reps)

    def func(x):
        return Statevector(qc.assign_parameters(x)).expectation_value(ham).real

    result = minimize(func, x0, method=optimizer, options=optimizer_options)

    # return result

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

    final_state = Statevector(qc.assign_parameters(result.x))

    with h5py.File(filepath, 'w') as file:
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

        optimize_result_to_hdf5(file, dict(result))

        res = file.create_group("results")
        res.create_dataset("parameters", data=result.x)
        res.attrs["ham_expectation"] = result.fun
        res.attrs["prob_exact"] = abs(final_state[stack_to_state(target_stack, encoding)]) ** 2
        if constraint_matrix is not None:
            res.attrs["overlap_constraint"] = final_state.expectation_value(
                generate_penalty_hamiltonian(laminate, constraint_matrix, encoding, dyad_ops, strength=1.)
            )

    return filepath