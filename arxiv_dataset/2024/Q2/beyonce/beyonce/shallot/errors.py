"""
This python module contains all the custom exceptions used for BeyonCE.
"""

import numpy as np
from typing import Union


class LoadError(Exception):
    """Used when loading a class instance from a directory fails."""

    def __init__(self, type_string: str, directory: str) -> None:
        """
        This method contains the initialisation parameters for this custom
        exception.
        
        Parameters
        ----------
        type_string : str
            Name of the class that should be instantiated.
        directory : str
            Directory from which the class instance should be loaded.
        """
        message = f"Failed to load {type_string} from {directory}."
        super().__init__(message)

class InvalidShapeError(Exception):
    """Used when multiple arrays do not have the same shape."""

    def __init__(self, 
            names_list: list[str],
            arrays_list: list[np.ndarray]
        ) -> None:
        """
        This method contains the initialisation parameters for this custom
        exception.

        Parameters
        ----------
        names_list : list[str]
            List of the names of all the input arrays.
        arrays_list : list[np.ndarray]
            List of the input arrays.
        """
        message = ""
        for name, array in zip(names_list, arrays_list):
            message = message + f"{name} {str(array.shape)}, "
        message = message[:-2] + " should all have the same shape."
        super().__init__(message)

class InvalidDimensionsError(Exception):
    """Used when an array doesn't have the right number of dimensions."""

    def __init__(self, 
            array_name: str, 
            array: np.ndarray, 
            num_dimensions: int
        ) -> None:
        """
        This method contains the initialisation parameters for this custom
        exception.
        
        Parmaters
        ---------
        array_name : str
            Name of the input array.
        array : np.ndarray
            Input array.
        num_dimensions : int
            Number of dimensions the input array should have.
        """
        message = (f"The {array_name} ({array.ndim} dimensions) should "
            f"have {num_dimensions} dimensions.")
        super().__init__(message)


class InvalidBoundsError(Exception):
    """Used when the upper bounds are lower than the lower bounds."""

    def __init__(self, 
            name_lower: str, 
            lower_bound: Union[float, int], 
            name_upper: str, 
            upper_bound: Union[float, int],
        ) -> None:
        """
        This method contains the initialisation parameters for this custom
        exception.
        
        Parameters
        ----------
        name_lower : str
            This is the name of the lower bound.
        lower_bound : float
            This is the value of the lower bound.
        name_upper : str
            This is the name of the upper bound.
        upper_bound : float
            This is the value of the upper bound.
        """
        message = (f"The {name_upper} argument ({upper_bound:.4f}) must be "
            f"greater than the {name_lower} argument ({lower_bound:.4f}).")
        super().__init__(message)


class ValueOutsideRangeError(Exception):
    """Used when the expected value is not within the proposed range"""

    def __init__(self, 
            property_name: str, 
            property_value: Union[float, int, np.ndarray], 
            lower_bound: Union[float, int] = -np.inf, 
            upper_bound: Union[float, int] = np.inf, 
            exclusive: bool = False
        ) -> None:
        """
        This method contains the initialisation parameters for the custom
        exception class
        
        Parameters
        ----------
        property_name : str
            Name of the property.
        property_value : float, int, np.ndarray
            Numerical value of the property.
        lower_bound : float, int
            Lower bound the property must adhere to.
        upper_bound : float, int
            Upper bound the property must adhere to.
        exclusive : bool
            Whether the bounds are acceptable values for the property or not.
        """
        if isinstance(property_value, np.ndarray):
            addition = f"must contain solely values"
        else:
            addition = f"({property_value:.4f}) must be"

        comparison = "between or equal to"
        if exclusive:
            comparison = "between"

        message = (f"The {property_name} argument {addition} {comparison} "
            f"{lower_bound:.4f} and {upper_bound:.4f}.")
        super().__init__(message)


class OriginMissingError(Exception):
    """This class is used when the origin is expected for some computation."""

    def __init__(self, object_name: str) -> None:
        """
        The initialiser method requires no parameters and returns no values.
        """
        message = (f"The {object_name} parameters can not be extended. That is"
            " only possible when the grid parameters include the origin.")
        super().__init__(message)