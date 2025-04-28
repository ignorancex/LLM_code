"""
This module contains the Diagnostics class which holds information about the
accuracy of the numerical determination of the rf array.
"""

from __future__ import annotations

import os
import numpy as np

from beyonce.shallot.grid_parameters import Parameters

import beyonce.validate as validate
from beyonce.shallot.errors import LoadError 



class Diagnostics:
    """
    This class contains all the diagnostic information for the grid. This is
    concerns how well the linear interpolation of the rf dimension works. It
    can be saved and loaded.
    """


    def __init__(self, parameters: Parameters) -> None:
        """
        This is the constructor for the disk grid parameter class.
        """
        validate.class_object(parameters, "parameters", 
            Parameters, "Parameters")
        self._max_y, self._max_x = parameters.slice_shape
        
        self._fy_dict: dict = {}
        self._disk_radius_dict: dict = {}
        self._extended: bool = False


    def __str__(self) -> str:
        """
        This method returns a representation string of the grid diagnostics
        class.

        Returns
        -------
        str_string : str
            Representation string of the grid diagnostics class.
        """
        return self.__repr__()


    def __repr__(self) -> str:
        """
        This method returns a representation string of the grid diagnostics
        class.

        Returns
        -------
        repr_string : str
            Representation string of the grid diagnostics class.
        """
        lines: list[str] = [""]
        lines.append("Grid Diagnostics")
        lines.append(28 * "-")
        lines.append(f"diagnostics saved: {len(self._fy_dict.keys())}")
        
        return "\n".join(lines)


    def _generate_key(self, y: float, x: float) -> None:
        """
        This method determines whether or not the (y, x) pair is allowed
        if allowed values are set.

        Parameters
        ----------
        y : float
            The value of y that should be associated with the key.
        x : float
            The value of x that should be associated with the key.

        Returns
        -------
        key : str
            Key for the diagnostic dictionaries of the form (y, x).
        """
        if self._extended:
            upper_bound_y = 2 * (self._max_y - 1)
            upper_bound_x = 2 * (self._max_x - 1)
        else:
            upper_bound_y = self._max_y - 1
            upper_bound_x = self._max_x - 1

        y = validate.number(y, "y", lower_bound=0, upper_bound=upper_bound_y, 
            check_integer=True)
        x = validate.number(x, "x", lower_bound=0, upper_bound=upper_bound_x, 
            check_integer=True)

        if self._extended:
            y = int(np.abs(y - self._max_y))
            x = int(np.abs(x - self._max_x))

        key = f"({y}, {x})"

        return key


    def get_diagnostic(self, 
            y: int, 
            x: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to extract the diagnostic values from each 
        dictionary according to the key provided
        
        Parameters
        ----------
        y : int
            The value of y that should be associated with the key.
        x : int
            The value of x that should be associated with the key.
        
        Returns
        -------
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        key = self._generate_key(y, x)
        fy = self._fy_dict[key]
        disk_radius = self._disk_radius_dict[key]

        return fy, disk_radius


    def save_diagnostic(self, 
            y: int,
            x: int, 
            fy: np.ndarray, 
            disk_radius: np.ndarray
        ) -> None:
        """
        This method is used to save a diagnostic value to each dictionary
        
        Parameters
        ----------
        y : int
            The value of y that should be associated with the key.
        x : int
            The value of x that should be associated with the key.
        fy : np.ndarray
            The fy values that correspond to the provided key (stored in 
            fy_dict property).
        disk_radius : np.ndarray
            The disk radius values that correspond to the provided key (stored 
            in disk_radius_dict property).
        """
        if self._extended:
            raise RuntimeError("diagnostics have been extended and can not "
                "be changed")
                
        fy = validate.array(fy, "fy", dtype="float64", num_dimensions=1)
        disk_radius = validate.array(disk_radius, "disk_radius", 
            dtype="float64", num_dimensions=1)
        validate.same_shape_arrays([fy, disk_radius], ["fy", "disk_radius"])

        key = self._generate_key(y, x)
        self._fy_dict[key] = fy
        self._disk_radius_dict[key] = disk_radius


    def extend(self) -> None:
        """
        This method sets the status to extended for the diagnostics object
        """
        self._extended = True


    def save(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        directory = validate.string(directory, "directory")
        
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.save(f"{directory}/fy_dict", self._fy_dict)
        np.save(f"{directory}/disk_radius_dict", self._disk_radius_dict)

        np.save(f"{directory}/max_yx", np.array([self._max_y, self._max_x]))


    @classmethod
    def load(cls, directory: str) -> Diagnostics:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        diagnostics : Diagnostics
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")
        parameters = Parameters(0, 1, 1, 0, 1, 1, 2, 1)
        
        try:    
            diagnostics = cls(parameters)
            
            diagnostics._fy_dict = np.load(f"{directory}/fy_dict.npy",
                allow_pickle=True).item()
            diagnostics._disk_radius_dict = np.load(
                f"{directory}/disk_radius_dict.npy", allow_pickle=True).item()
            
            max_y, max_x = np.load(f"{directory}/max_yx.npy")
            diagnostics._max_x = max_x
            diagnostics._max_y = max_y
        
        except Exception:
            raise LoadError("diagnostics", directory) 

        return diagnostics