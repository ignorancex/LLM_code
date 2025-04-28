"""
This module contains the parameter class used to hold grid information and 
instantiate new shallot grids.
"""

from __future__ import annotations

import os
import numpy as np

import beyonce.validate as validate
from beyonce.shallot.errors import LoadError, InvalidBoundsError, OriginMissingError


class Parameters:
    """
    This class contains all the information pertaining to the grid:
        dx (x-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        dy (y-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        rf (extent of rf_array in one-direction)
        rf_array (radius factor that compares with the smallest disk at a 
            given point)
        grid_shape (shape of the total grid)
        slice_shape (shape of dy, dx slice)
        extendable (whether the grid contains point (0, 0))
    It also contains methods to save and load the grid parameters.
    """

 
    def __init__(self, 
            min_x: float, 
            max_x: float, 
            num_x: int, 
            min_y: float, 
            max_y: float, 
            num_y: int, 
            max_rf: float, 
            num_rf: int
        ) -> None:
        """
        This is the constructor for the disk grid parameter class

        Parameters
        ----------
        num_xy : integer
            This is the resolution of the grid in the dx and dy directions.
        maximum_radius : float
            This is the maximum radius of the disk [t_ecl].
        num_rf : integer
            This is the resolution of the grid in the rf direction. Note that
            the size of this grid dimension is then (2 * num_rf - 1).
        maximum_rf : float
            This is the maximum rf value.
        """
        self.dx = self._determine_dx(min_x, max_x, num_x)
        self.dy = self._determine_dy(min_y, max_y, num_y)
        self.rf, self.rf_array = self._determine_rf(max_rf, num_rf)
        
        self.grid_shape, self.slice_shape = self._determine_grid_and_slice_shape()
        self.extendable = self._determine_extendable()

 
    def __str__(self) -> str:
        """
        This returns the string representation of the class.

        Returns
        -------
        str_string : str
            String representation of the Parameters class.
        """
        return self.__repr__()

 
    def __repr__(self) -> str:
        """
        This generates a string representation of the grid parameters object. 
        
        Returns
        -------
        repr_string : str
            Representation string of the grid parameters class. 
        """
        dy, dx, rf_array = self.get_vectors()
        lines: list[str] = [""]
        lines.append("Grid Parameters")
        lines.append(28 * "-")
        
        dx_min = f'{f"{dx[0]:.2f}":>6}'
        dx_max = f'{f"{dx[-1]:.2f}":>6}'
        lines.append(f"dx: {dx_min} -> {dx_max} ({len(dx)})")

        dy_min = f'{f"{dy[0]:.2f}":>6}'
        dy_max = f'{f"{dy[-1]:.2f}":>6}'
        lines.append(f"dy: {dy_min} -> {dy_max} ({len(dy)})")

        rf_min = f'{f"{1:.2f}":>6}'
        rf_max = f'{f"{rf_array[0]:.2f}":>6}'
        rf_num = len(rf_array)
        lines.append(f"rf: {rf_max} -> {rf_min} -> {rf_max} ({rf_num})")

        lines.append(f"grid_shape: {self.grid_shape}")

        repr_string = "\n".join(lines)
        return repr_string


    def __eq__(self, other: Parameters) -> bool:
        """
        This method is used to determine whether or not two instances are 
        equal to each other
`
        Returns
        -------
        equal : bool
            Whether two class instances contain the same information.
        """
        dx_equal = np.all(self.dx == other.dx)
        dy_equal = np.all(self.dy == other.dy)
        rf_equal = np.all(self.rf == other.rf)
        
        equal = dx_equal and dy_equal and rf_equal
        return equal

    
    def _generate_vector(self,
            min_value : float,
            max_value : float,
            num_points : int,
            name_vector : str
        ) -> np.ndarray:
        """
        This method generates a linspace array defined by the input parameters
        
        Parameters
        ----------
        min_value : float
            The minimum value of the vector.
        max_value : float
            The maximum value of the vector.
        num_points : int
            The length of the vector.
        name_vector : str
            The name of the vector attached to error messages.
        """
        name_vector = validate.string(name_vector, "name_vector")
        min_value = validate.number(min_value, "min_value")
        max_value = validate.number(max_value, "max_value")
        if min_value >= max_value:
            raise InvalidBoundsError(
                f"min_{name_vector}", min_value, f"max_{name_vector}", max_value
            )
        
        num_points = validate.number(num_points, "num_points", check_integer=True, 
            lower_bound=1)
        
        return np.linspace(min_value, max_value, num_points)

 
    def _determine_dx(self, 
            min_x: float, 
            max_x: float, 
            num_x: int
        ) -> np.ndarray:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        min_x : float
            The minimum value of x [t_ecl].
        max_x : float
            The maximum value of x [t_ecl].
        num_x : int
            The number of dx elements.

        Returns
        -------
        dx : np.ndarray
            Grid dx dimension vector.
        """
        return self._generate_vector(min_x, max_x, num_x, "x")[None, :, None]

 
    def _determine_dy(self, 
            min_y: float, 
            max_y: float, 
            num_y: int
        ) -> np.ndarray:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        min_y : float
            The minimum value of y [t_ecl].
        max_y : float
            The maximum value of y [t_ecl].
        num_y : int
            The number of dy elements.

        Returns
        -------
        dy : np.ndarray
            Grid dy dimension vector.
        """
        return self._generate_vector(min_y, max_y, num_y, "y")[:, None, None]

 
    def _determine_rf(self, 
            max_rf: float, 
            num_rf: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to determine the dx vector
        
        Parameters
        ----------
        max_rf : float
            The maximum value of rf [-].
        num_rf : int
            The number of rf elements (in one direction).

        Returns
        -------
        rf : np.ndarray
            Rf range from 1 to max_rf in num_rf.
        rf_array : np.ndarray
            Grid rf dimension vector.
        """
        rf = self._generate_vector(1, max_rf, num_rf, "rf")
        rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
        return rf, rf_array

 
    def _determine_grid_and_slice_shape(self) -> tuple[
            tuple[int, int, int], 
            tuple[int, int]
        ]:
        """
        This method sets useful grid parameters (grid shape and slice shape).

        Returns
        -------
        grid_shape : tuple
        """
        dy, dx, rf_array = self.get_vectors()
        grid_shape = (len(dy), len(dx), len(rf_array))
        slice_shape = (len(dy), len(dx))

        return grid_shape, slice_shape

 
    def _determine_extendable(self) -> bool:
        """
        This method is used to determine whether this particular set of grid
        parameters can be extended.

        Returns
        -------
        extendable : bool
            Whether the parameters object can be pe
        """
        dy, dx, _ = self.get_vectors()
        extendable = dy[0] == 0 and dx[0] == 0
        return extendable

 
    def extend_grid(self) -> None:
        """
        This method is used to reflect the grid parameters about the x and y
        axes.
        """
        if not self.extendable:
            raise OriginMissingError("grid")
        num_y, num_x = self.slice_shape
        max_y = self.dy[-1, 0, 0]
        max_x = self.dx[0, -1, 0]
        
        self.dx = self._determine_dx(-max_x, max_x, 2 * num_x - 1)
        self.dy = self._determine_dy(-max_y, max_y, 2 * num_y - 1)
        self.grid_shape, self.slice_shape = self._determine_grid_and_slice_shape()
        self.extendable = self._determine_extendable()

 
    def get_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method returns the flattened dy, dx, and rf grid vectors.
        
        Returns
        -------
        dy : np.ndarray
            The y coordinates of the centre of the ellipse [t_ecl]
        dx : np.ndarray
            The x coordinates of the centre of the ellipse [t_ecl]
        rf_array : np.ndarray
            The rf radius stretch factors of the ellipse [-]
        """
        return self.dy.flatten(), self.dx.flatten(), self.rf_array

 
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

        np.save(f"{directory}/dx", self.dx)
        np.save(f"{directory}/dy", self.dy)
        np.save(f"{directory}/rf", self.rf)
        np.save(f"{directory}/rf_array", self.rf_array)


    @classmethod
    def load(cls, directory: str) -> Parameters:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        parameters : Parameters
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")

        try:
            parameters = cls(0, 1, 1, 0, 1, 1, 2, 1)
            parameters.dx = np.load(f"{directory}/dx.npy")
            parameters.dy = np.load(f"{directory}/dy.npy")
            parameters.rf = np.load(f"{directory}/rf.npy")
            parameters.rf_array = np.load(f"{directory}/rf_array.npy")

            parameters.grid_shape, parameters.slice_shape = (
                parameters._determine_grid_and_slice_shape()
            )
            parameters.extendable = parameters._determine_extendable()
        except Exception:
            raise LoadError("parameters", directory)

        return parameters