from type_aliases import *
import numpy as np


def stack_to_state(stack: Stack, encoding: Encoding) -> str:
    """Convert a stacking sequence to a qubit basis state

    Args:
        stack (Stack): The stacking sequence as a numpy array with
            integer entries 0,...,num_angles-1
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where
            `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        str: The according qubit basis state

    """
    return ''.join(''.join(str(t) for t in encoding[s]) for s in stack)


def state_to_stack(state: str, encoding: Encoding) -> Stack:
    """Convert a qubit basis state to a stacking sequence

    Args:
        state (str): The qubit basis state as a string of `'0'` and `'1'`
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where
            `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        Stack: The corresponding stacking sequence as a numpy array
            with the according angle indices from 0,...,`num_angles-1
            as elements

    """
    qb_per_qd = len(encoding[0])
    num_plies = len(state) // qb_per_qd
    encoding_dict = {''.join(str(t) for t in enc): s for s, enc in enumerate(encoding)}
    return np.array([encoding_dict[state[n * qb_per_qd:(n + 1) * qb_per_qd]] for n in range(num_plies)], dtype=int)
