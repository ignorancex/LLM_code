from type_aliases import *

import numpy as np

_DEFAULT_ANGLE_FUNCTIONS = (
    lambda x: np.cos(2 * x),
    lambda x: np.sin(2 * x),
    lambda x: np.cos(4 * x),
    lambda x: np.sin(4 * x),
)


def generate_weights_A(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the A parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.
            Here, this has no effect.

    Returns:
        NDArray[float]: the weights, of shape `(num_plies,)`
    """
    return np.full(num_plies, 1 / num_plies)


def generate_weights_B(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the B parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.
            Here `True` raises an error.

    Returns:
        NDArray[float]: the weights, of shape `(num_plies,)`
    """
    if symmetric:
        raise ValueError("B-parameters only available for non-symmetric laminates.")
    boundaries_B = 2 * (np.arange(0, num_plies + 1) / num_plies - 1 / 2) ** 2
    return boundaries_B[1:] - boundaries_B[:-1]


def generate_weights_D(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the D parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.

    Returns:
        NDArray[float]: the weights, of shape `(num_plies,)`
    """
    if symmetric:
        boundaries_D = (np.arange(0, num_plies + 1) / num_plies) ** 3
    else:
        boundaries_D = 4 * (np.arange(0, num_plies + 1) / num_plies - 1 / 2) ** 3
    return boundaries_D[1:] - boundaries_D[:-1]


_WEIGHT_GENERATORS = {
    'A': generate_weights_A,
    'B': generate_weights_B,
    'D': generate_weights_D
}


def generate_weights(num_plies: int, symmetric: bool = True, which: bool = None) -> WeightsArray:
    """Calculate the weight for a laminate.

    For a symmetric laminate, `num_plies` is the number of plies for only half the stack,
    and the plies are indexed from the middle of the stack to the outside.

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric
        which (str): A string containing letters form ('A','B','D'),
            specifying the weight types

    Returns:
        WeightsArray: A numpy array of shape `(len(which),num_plies)` containing the weights`
    """
    if which is None:
        which = 'AD' if symmetric else 'ABD'
    which = which.upper()
    if symmetric and 'B' in which:
        raise ValueError("B-parameters only available for non-symmetric laminates.")

    return np.stack([
        _WEIGHT_GENERATORS[w](num_plies, symmetric) for w in which
    ])


def generate_funcs(
        angles: Sequence[int | float], angle_functions: Optional[Sequence[AngleFunction]] = None,
        deg: bool = False, round_decimals: Optional[int] = None
) -> FuncArray:
    """Calculate the angle-functions array

    Args:
        angles (Sequence[int | float]): The allowed ply angles
        angle_functions (:obj:`Sequence[AngleFunction]]`, optional):
            The ply angle functions which take a real value and output a float.
            If None (default), the functions usual functions [cos(2*x), sin(2*x), cos(4*x), sin(4*x)]
            are used.
        deg (:obj:`bool`, optional): Whether `angles` is in degrees or radians. Defaults to False.
        round_decimals (:obj:`int`, optional): Round the resulting array to a given number of decimals.
            If None (default), the array is left unchanged.

    Returns:
        FuncsArray: The resulting array of shape `(len(angles),len(angle_functions))` containing the
            function evaluations for the allowed ply-angles

    """
    if angle_functions is None:
        angle_functions = _DEFAULT_ANGLE_FUNCTIONS

    funcs = np.array([
        [f(a * np.pi / 180 if deg else a) for f in angle_functions]
        for a in angles
    ])

    return funcs if round_decimals is None else funcs.round(decimals=round_decimals)


class Laminate:
    """Class to specify the properties of a laminate

    Note:
        The laminate is defined through the weights and functions arrays.
        The allowed ply angles are not stored in this class, but are implicitly
        included through the functions array.

    Args:
        weights (WeightsArray): A numpy array of shape`(num_weights,num_plies)`
            containing the ply-dependent weights for the lamination parameters
            where the rows correspond to the different types of parameters
            (usually A, B and D)
        funcs (FuncArray): A numpy array of shape `(num_angles,num_funcs)`
            containing the function evaluations for the allowed ply angles
            where the columns correspond to the different ply-angle functions
            (usually cos(2x), sin(2x), cos(4x) and sin(4x))
        constraints (:obj:`Collection[Constraint]):
            A collection of functions Stack -> bool that evaluate to `true`
            if the according constraint is satisfied for a given stacking sequence.

    Attributes:
        weights (WeightsArray): The weights array
        funcs (FuncArray): The functions array
        num_plies (int): The number of plies
        num_angles (int): The number of ply-angles
        num_weights (int): The number of distinct sets of weights
        num_funcs (int): The number of distinct ply-angle functions
        num_parameters (int): The number of according lamination parameters
            (num_weights * num_funcs)
        constraints (list[Constraint]): The constraint functions


    """
    def __init__(self, weights: WeightsArray, funcs: FuncArray,
                 constraints: Optional[Collection[Constraint]] = None):
        self.weights: WeightsArray = weights
        self.funcs: FuncArray = funcs

        self.num_plies: int
        self.num_angles: int
        self.num_weights: int
        self.num_funcs: int
        self.num_weights, self.num_plies = self.weights.shape
        self.num_angles, self.num_funcs = self.funcs.shape
        self.num_parameters: int = self.num_weights * self.num_funcs

        self.constraints: list[Constraint] = [] if constraints is None else list(constraints)

    def parameters(self, stack: Stack) -> Parameters:
        """Calculate the lamination parameters for a stacking sequence

        Args:
            stack (Stack): The stacking sequence, given as an numpy array of length `num_plies`
                with integer elements from 0,...,`num_angles-1`, corresponding to the allowed
                ply-angles

        Returns:
            Parameters: A numpy array of shape `(num_weights,num_funcs)` containing the
                lamination parameters of the stack

        """
        return self.weights @ self.funcs[stack]

    def is_valid(self, stack: Stack) -> bool:
        """Check if a stacking sequence is valid under the given constraints

        Args:
            stack (Stack): The stacking sequence, given as an numpy array of length `num_plies`
                with integer elements from 0,...,`num_angles-1`, corresponding to the allowed
                ply-angles

        Returns:
            bool: True if all constraints are satisfied, else False

        """
        return all(constr(stack) for constr in self.constraints)
