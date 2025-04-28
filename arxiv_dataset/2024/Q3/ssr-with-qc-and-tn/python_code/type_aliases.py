from typing import TypeAlias, Optional, Callable, Sequence, Collection, Iterable, Any
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

StackElem: TypeAlias = int  # 0,1,...,num_angles-1
Stack: TypeAlias = NDArray[StackElem]  # length num_plies
FuncArray: TypeAlias = NDArray[float]  # size num_angles x num_funcs
WeightsArray: TypeAlias = NDArray[float]  # size num_weights x num_plies
AngleFunction: TypeAlias = Callable[[int | float], float]  # function on angles (in rad)
Parameters: TypeAlias = NDArray[float]  # size num_weighs x num_funcs
Constraint: TypeAlias = Callable[[Stack],bool]  # boolean function on stack, True if constraint is satisfied

Encoding: TypeAlias = Sequence[tuple[int,...]]  # length: num_angels, length of elements: qubits per ply
DyadOps: TypeAlias = Sequence[SparsePauliOp]  # Outer products corresponding to the above encoded states
