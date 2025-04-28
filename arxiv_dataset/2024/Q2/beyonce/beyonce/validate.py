"""
This module is used to perform basic validations on arguments to methods
and functions. Method/function specific validations are defined inside the
methods/functions themselves.
"""

import numpy as np
from typing import Any, Union
from beyonce.shallot.errors import InvalidBoundsError, InvalidDimensionsError
from beyonce.shallot.errors import InvalidShapeError, ValueOutsideRangeError


def boolean(parameter: Any, name: str) -> bool:
    """
    This function is used to raise an Exception if the object passed is not 
    a boolean.
    
    Parameters
    ----------
    parameter : Any
        Object that is validated as a boolean.
    name : str
        Name of the object to be passed in exception message.
    """
    return class_object(parameter, name, bool, "boolean")


def string(parameter: Any, name: str) -> str:
    """
    This function is used to raise an Exception if the object passed is not
    a string.
    
    Parameters
    ----------
    parameter : Any
        Object that is validated as a string.
    name : str
        Name of the object to be passed in exception message.
    """
    return class_object(parameter, name, str, "string")


def number(
        parameter: Any, 
        name: str, 
        check_integer: bool = False, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf, 
        exclusive: bool = False
    ) -> Union[float, int]:
    """
    Parameters
    ----------
    parameter : Any
        Object that is validated as a number.
    name : str
        Name of the object to be passed in exception message.
    check_integer : bool
        Check if object is an integer [default = False].
    lower_bound : float
        Lower bound of the number [default = -np.inf].
    upper_bound : float
        Upper bound of the number [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    """
    # validations
    parameter = _number_type(parameter, name, check_integer)
    parameter = _number_bounds(parameter, name, lower_bound, upper_bound, 
        exclusive)
    
    return parameter


def _number_type(
        parameter: Any, 
        name: str, 
        check_integer: bool = False
    ) -> Union[float, int]:
    """
    This function is used to raise an Exception if the object passed is not a
    number.
    
    Parameters
    ----------
    parameter : Any
        Object that is validated as a number.
    name : str
        Name of the object to be passed in exception message.
    check_integer : bool
        Check if object is an integer [default = False].
    """
    boolean(check_integer, 'check_integer')
    
    if check_integer:
        return _integer_type(parameter, name)
    
    invalid = True
    try:
        parameter = _integer_type(parameter, name)
        invalid = False
    except TypeError:
        pass

    try:
        parameter = _float_type(parameter, name)
        invalid = False
    except TypeError:
        pass

    if invalid:
        raise TypeError(f"{name} is not of type 'number'")
    
    return parameter


def _number_bounds(
        number: Union[float, int], 
        name: str, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf,
        exclusive: bool = False
    ) -> Union[float, int]:
    """
    This function validates whether the passed object agrees with the given 
    bounds.
    
    Parameters
    ----------
    number : float or int
        Number to be bounded.
    name : str
        Name of the object to be passined in exception message.
    lower_bound : float
        Lower bound of the number [default = -np.inf].
    upper_bound : float
        Upper bound of the number [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    """
    # validations
    _number_type(lower_bound, "lower_bound")
    _number_type(upper_bound, "upper_bound")
    boolean(exclusive, "exclusive")
    
    if lower_bound > upper_bound:
        raise InvalidBoundsError(
            "lower_bound", lower_bound, "upper_bound", upper_bound
        )

    # validate number bounds
    raise_error = False

    # inclusive
    lower = np.less
    upper = np.greater
    
    # exclusive
    if exclusive:
        lower = np.less_equal
        upper = np.greater_equal

    # try lower bound
    if lower(number, lower_bound):
        raise_error = True

    # try upper bound
    if upper(number, upper_bound):
        raise_error = True

    if raise_error:
        raise ValueOutsideRangeError(
            name, number, lower_bound, upper_bound, exclusive
        )

    return number


def array(
        parameter: Any, 
        name: str, 
        lower_bound: float = None, 
        upper_bound: float = None, 
        exclusive: bool = False, 
        dtype: str = None,
        num_dimensions: int = None
    ) -> np.ndarray:
    """
    This function is used to raise an Exception if the object passed is not 
    a numpy array with the given restrictions.

    Parameters
    ----------
    parameter : Any
        Object to be checked.
    name : str
        Name of the object to be passed in Exception message.
    lower_bound : float
        Lower bound of the array [default = None].
    upper_bound : float
        Upper bound of the array [default = None].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound].
    dtype : str
        Type the array should have.
    num_dimensions : int
        Number of dimensions the array should have.
    
    """
    # array?
    _array_like(parameter, name)

    # bounds?
    if (lower_bound is not None) or (upper_bound is not None):
        if lower_bound is None:
            lower_bound = -np.inf
        
        if upper_bound is None:
            upper_bound = np.inf

        _array_bounds(parameter, name, lower_bound, upper_bound, 
            exclusive)
    
    # dimensions?
    if num_dimensions is not None:
        _array_dimensions(parameter, name, num_dimensions)

    # array type?
    if dtype is not None:
        _array_type(parameter, name, dtype)

    return parameter


def _array_like(
        parameter: Any, 
        name: str
    ) -> np.ndarray:
    """
    This function is used to raise an Exception if the object passed is not 
    iterable.
    
    Parameters
    ----------
    parameter : Any
        Object to be checked for iterability.
    name : str
        Name of the object to be passed in Exception message.
    """
    return class_object(parameter, name, np.ndarray, "np.ndarray")


def _array_bounds(
        array: np.ndarray, 
        array_name: str, 
        lower_bound: float = -np.inf, 
        upper_bound: float = np.inf, 
        exclusive: bool = False
    ) -> np.ndarray:
    """
    This function validates whether the passed array agrees with the given 
    bounds.

    Parameters
    ----------
    array: np.ndarray
        Array to be validated with lower and upper bounds.
    array_name : string
        Name of the array to be passed to exception message.
    lower_bound : float
        Lower bound of the array [default = -np.inf].
    upper_bound : float
        Upper bound of the array [default = np.inf].
    exclusive : bool
        Determines whether to have exclusive bounds i.e. (lower_bound, 
        upper_bound), instead of [lower_bound, upper_bound]. 
    """
    # validate
    _number_type(lower_bound, "lower_bound")
    _number_type(upper_bound, "upper_bound")
    boolean(exclusive, "exclusive")

    if lower_bound > upper_bound:
        raise InvalidBoundsError(
            "lower_bound", lower_bound, "upper_bound", upper_bound
        )

    # validate array bounds
    raise_error = False

    # inclusive
    lower = np.less
    upper = np.greater
    
    # exclusive
    if exclusive:
        lower = np.less_equal
        upper = np.greater_equal

    # try lower bound
    if np.any(lower(array, lower_bound)):
        raise_error = True

    # try upper bound
    if np.any(upper(array, upper_bound)):
        raise_error = True

    if raise_error:
        raise ValueOutsideRangeError(
            array_name, array, lower_bound, upper_bound, exclusive
        )
    
    return array


def _array_dimensions(
        array: np.ndarray, 
        array_name: str, 
        num_dimensions: int
    ) -> np.ndarray:
    """
    This function validates the number of dimension an array has.
    
    Parameters
    ----------
    array : np.ndarray
        Array to be validate for dimensions.
    array_name : str
        Name of the array to be passed to the exception message.
    num_dimensions : int
        Number of dimensions the array should have.
    """
    # validations
    _number_type(num_dimensions, "num_dimensions", check_integer=True)

    if array.ndim != num_dimensions:
        raise InvalidDimensionsError(array_name, array, num_dimensions)

    return array


def _array_type(array: np.ndarray, array_name: str, dtype: str) -> np.ndarray:
    """
    This function validates the dtype of the input array.
    
    Parameters
    ----------
    array : np.ndarray
        Array to be validate for dimensions.
    array_name : str
        Name of the array to be passed to the exception message.
    dtype : str
        Type the array should have.
    """
    # validate
    string(dtype, "dtype")

    # validate array type
    array_type = str(array.dtype)

    if array_type != dtype:
        raise TypeError(f"{array_name} ({array_type}) is not of type {dtype}")

    return array


def same_shape_arrays(
        arrays_list: list[np.ndarray], 
        names_list: list[str]
    ) -> None:
    """
    This method ensures that all the arrays in the list have the same length.
    
    Parameters
    ----------
    arrays_list : list (np.ndarray)
        List of arrays that will have their lengths compared.
    names_list : list (str)
        List of the names of each array.
    """
    # validate
    if len(arrays_list) < 2:
        raise AttributeError("arrays_list argument should contain at least "
            "two arrays")

    if len(arrays_list) != len(names_list):
        raise AttributeError(f"arrays_list ({len(arrays_list)}) should have "
            f"the same length as names_list ({len(names_list)})")

    shape = arrays_list[0].shape
    error = False

    for array, name in zip(arrays_list, names_list):
        array = _array_like(array, name)
        if array.shape != shape:
            error = True
    
    if error:
        raise InvalidShapeError(names_list, arrays_list)


def class_object(
        parameter: Any, 
        name: str, 
        class_type: Any, 
        class_name: str
    ) -> Any:
    """
    This method is used to ensure that the object passed is of the class type
    
    Parameters
    ----------
    parameter : Any
        Object that is validate against the class.
    name : str
        Name of the object to be passed in exception message.
    class_type : Any
        Class that the object is validated against.
    class_name : str
        Name of the class to be passed in exception message.

    Returns
    -------
    parameter : Any
        Object passed in.    
    """
    if not isinstance(parameter, class_type):
        raise TypeError(f"{name} is not of type {class_name}")

    return parameter


def _integer_type(parameter: Any, name: str):
    """
    This method is used to ensure that the object passed is an integer type.

    Parameters
    ----------
    parameter : Any
        Object that is validate against the class.
    name : str
        Name of the object to be passed in exception message.
    """
    int_types = [int, np.int0, np.int8, np.int16, np.int32, np.int64]
    int_names = ["int", "np.int0", "np.int8", "np.int16", "np.int32", 
        "np.int64"]
    
    valid = False
    for int_type, int_name in zip(int_types, int_names):
        try:
            class_object(parameter, name, int_type, int_name)
            valid = True
        except TypeError:
            pass

        if valid:
            break

    if not valid:
        raise TypeError(f"{name} is not of type integer")

    return parameter


def _float_type(parameter: Any, name: str):
    """
    This method is used to ensure that the object passed is a float type.

    Parameters
    ----------
    parameter : Any
        Object that is validate against the class.
    name : str
        Name of the object to be passed in exception message.
    """
    float_types = [float, np.float16, np.float32, np.float64, np.float128]
    float_names = ["float", "np.float16", "np.float32", "np.float64", 
        "np.float128"]
    
    valid = False
    for float_type, float_name in zip(float_types, float_names):
        try:
            class_object(parameter, name, float_type, float_name)
            valid = True
        except TypeError:
            pass

        if valid:
            break

    if not valid:
        raise TypeError(f"{name} is not of type 'number'")

    return parameter