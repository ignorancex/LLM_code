"""This module contains the ShallotGrid class"""

from __future__ import annotations
from enum import Enum

import os
import shutil
import logging
import numpy as np
from tqdm import tqdm

from beyonce.shallot.grid_names import Name
from beyonce.shallot.grid_units import Unit
from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.grid_diagnostics import Diagnostics
from beyonce.shallot.grid_components import Property, Gradient

import beyonce.validate as validate
from beyonce.shallot.errors import LoadError, OriginMissingError


TEMP_SAVE_DIR = "./.temp_shallot"


class ShallotGrid:
    """
    This class contains all the grid information required to do a full
    investigation of the circumplanetary disk parameter space. It contains
    grids that are defined by:
        dx (x-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        dy (y-coordinate of disk centre w.r.t. eclipse centre [t_ecl])
        rf (radius factor that compares with the smallest disk at a given 
            point)
    With disk parameters:
        disk_radius (maximum size of the disk [t_ecl])
        inclination (angle that relates semi-major to semi-minor axes [deg])
        tilt (angle between semi-major axis and the x-axis [deg])
        gradients (calculated after building of the grid)
    With grid parameters:   
        fx (horizontal stretch)
        fy (horizontal stretch)
        diagnostic (compares the calculated value of the disk_radius to the 
            expected radius fraction).
    It contains helper functions to:
        update_measured_gradients_scaling (used in case astrophysical
            parameters that scale the measured gradients are updated)
        generate_hill_radius_mask (used to limit the parameter space by the 
            Hill radius [t_ecl])
        determine_closest_grid_point (used to find the closest grid point to
            a given solution)
        get_information_from_grid_point (used to determine all important
            values from a coordinate in the grid)
    This all along with a logger to help maintain changes. The grid can also
    be saved and loaded.
    """

 
    def __init__(self,
            parameters: Parameters, 
            num_fxfy: int, 
            logging_level: Enum = logging.INFO,
            keep_diagnostics: bool = False, 
            intermittent_saving: bool = False
        ) -> None:
        """
        This is the constructor for the shallot grid class

        Parameters
        ----------
        parameters : Parameters
            The vectors that define each dimension of the data cube.
        num_fxfy : int
            This is the resolution of the grid in the fx and fy directions,
            which is subsequently used to interpolate the rf dimension of the
            grid.
        logging_level : Enum
            This is one of the standard logging levels [default = INFO].
        keep_diagnostics : bool
            This parameter determines whether diagnostics are saved that store
            the full set of fy values and corresponding disk radius values 
            which are used to interpolate for the rf dimension 
            [default = False].
        intermittent_saving : bool
            This parameter determines whether the grid is saved after every
            y iteration (useful for large grids with excessive build times)
            [default = False].
        """
        # generate logger
        self._set_logger(logging_level)

        self.diagnostic_map: Property = None
        self.gradients: list[Gradient] = None
        self.gradient_fit: Property = None
        self._eclipse_duration: float = None
        self._transverse_velocity: float = None
        self._limb_darkening: float = None

        # set user values
        parameters = validate.class_object(parameters, "parameters", 
            Parameters, "Parameters")
        self.parameters = parameters
        self._num_fxfy = validate.number(num_fxfy, "num_fxfy",
            check_integer=True, lower_bound=1)

        self._set_diagnostics(keep_diagnostics)

        # determine circular radius and shear
        self._determine_circular_radius()
        self._determine_shear()
        
        # get grid
        self._build_grid(intermittent_saving)

        # diagnose then extend the grid (fill the quadrants in)
        self._extend_grid()
        self._diagnose_fxfy_resolution()

        # set property contrast parameters
        self._set_grid_property_contrast_parameters()

 
    def __str__(self) -> str:
        """
        This method is used to print information about the shallot grid.
        
        Returns
        -------
        str_string : str
            String representation of the shallot grid.
        """
        return self.__repr__()

 
    def __repr__(self) -> str:
        """
        This method is used to print information about the shallot grid.
        
        Returns
        -------
        repr_string : str
            String representation of the shallot grid.
        """
        # get resolution diagnostic values
        resolution_diagnostic = self.diagnostic_map.data
        max_deviation = np.nanmax(np.abs(resolution_diagnostic))

        # get total masked percentage
        total_mask = self.get_combined_mask()
        fraction_masked = np.sum(total_mask) / np.prod(total_mask.shape)
        percentage_masked = f"{100 * fraction_masked:.2f}%"

        # print information
        lines: list[str] = [""]
        lines.append("==========================================")
        lines.append("******** SHALLOT GRID INFORMATION ********")
        lines.append("==========================================")
        lines.append(self.parameters.__str__())
        lines.append(self.disk_radius.__repr__())
        lines.append(self.tilt.__repr__())
        lines.append(self.inclination.__repr__())
        if self.gradients is not None:
            for gradient in self.gradients:
                lines.append(gradient.__repr__())
        if self.diagnostics is not None:
            lines.append(self.diagnostics.__repr__())
        lines.append("")
        lines.append("Grid Information")
        lines.append(28 * "-")
        lines.append(f"percentage masked: {percentage_masked}")
        lines.append(f"worst interpolation fit: {max_deviation:.9f}")
        lines.append("")
        lines.append("==========================================")

        return "\n".join(lines)

 
    def save(self, directory: str, y_value: int = None) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        y_value : int
            This is the value at which to start the grid constsruction.
        """
        directory = validate.string(directory, "directory")
        # allow saving into the existing temporary save folder
        if directory == TEMP_SAVE_DIR:
            if not os.path.exists(directory):
                os.mkdir(directory)
        else:
            os.mkdir(directory)

        if y_value is not None:
            y_upper_bound = self.parameters.grid_shape[0]
            y_value = validate.number(y_value, "y_value", check_integer=True,
                lower_bound=0, upper_bound=y_upper_bound)
            np.save(f"{directory}/y_value", np.array([y_value]))

        np.save(f"{directory}/num_fxfy", np.array([self._num_fxfy]))
        self.parameters.save(directory)
        self.disk_radius.save(directory)
        self.inclination.save(directory)
        self.tilt.save(directory)
        self.fx_map.save(directory)
        self.fy_map.save(directory)

        if self.gradients is not None:
            np.save(f"{directory}/eclipse_parameters", np.array([
                self._eclipse_duration, self._transverse_velocity, 
                self._limb_darkening]))

            for gradient in self.gradients:
                gradient.save_gradient(directory)

        if self.diagnostics is not None:
            self.diagnostics.save(directory)

 
    def _load_grid_attributes(self, directory: str) -> None:
        """
        This method is used to load the attributes that correspond to a grid
        based on the information stored in the provided directory

        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        # to ensure all errors returned are LoadErrors
        try:
            self._num_fxfy = np.load(f"{directory}/num_fxfy.npy")[0]
        except FileNotFoundError:
            raise LoadError("num_fxfy", directory)

        self.parameters = Parameters.load(directory)
        
        self.disk_radius = Property.load(directory, Name.DISK_RADIUS, 
            Unit.ECLIPSE_DURATION)
        self.inclination = Property.load(directory, Name.INCLINATION, 
            Unit.DEGREE)
        self.tilt = Property.load(directory, Name.TILT, Unit.DEGREE)
        
        self.fx_map = Property.load(directory, Name.FX_MAP, Unit.NONE)
        self.fy_map = Property.load(directory, Name.FY_MAP, Unit.NONE)
        
        try:
            self.diagnostics = Diagnostics.load(directory)
        except LoadError:
            self.logger.debug("no diagnostics found")
        

        self.gradients = []
        files = os.listdir(directory)
        
        for f in files:
            if f[:8] == "gradient":
                try:
                    gradient = Gradient.load(f"{directory}/{f}")
                    self.gradients.append(gradient)
                except LoadError:
                    self.logger.debug(f"{f} is corrupted gradient not loaded")
        
        if not self.gradients:
            self.gradients = None
        else:
            # to ensure all errors returned are LoadErrors.
            try:
                eclipse_parameters = np.load(f"{directory}/"
                    "eclipse_parameters.npy")
                (self._eclipse_duration, self._transverse_velocity,
                    self._limb_darkening) = eclipse_parameters
            except FileNotFoundError:
                raise LoadError("eclipse_parameters", directory)


    @classmethod
    def load(cls, 
            directory: str, 
            logging_level: Enum = logging.INFO
        ) -> ShallotGrid:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid : ShallotGrid
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")
        temp_parameters = Parameters(0, 1, 2, 0, 1, 2, 2, 2)
        grid = cls(temp_parameters, 100, logging_level=logging.CRITICAL)

        grid._load_grid_attributes(directory)
        grid._determine_circular_radius()
        grid._determine_shear()
        grid._set_logger(logging_level)
        
        if os.path.exists(f"{directory}/y_value.npy"):
            y_value = np.load(f"{directory}/y_value.npy")[0]
            grid._build_grid(intermittent_saving=True, start_y=y_value)
            grid._extend_grid()

        grid._diagnose_fxfy_resolution()

        return grid

 
    def _set_diagnostics(self, keep_diagnostics: bool) -> None:
        """
        This method sets the grid diagnostics if required
        
        Parameters
        ----------
        keep_diagnostics : bool
            This parameter determines whether diagnostics are saved that store
            the full set of fy values and corresponding disk radius values 
            which are used to interpolate for the rf dimension
        """
        keep_diagnostics = validate.boolean(keep_diagnostics, 
            'keep_diagnostics')
        
        if keep_diagnostics:
            self.diagnostics = Diagnostics(self.parameters)
        else:
            self.diagnostics = None

 
    def _set_logger(self, logging_level: Enum) -> None:
        """
        This method sets the logger for this class instance.
        
        Parameters
        ----------
        logging_level : Enum
            Determines the logging level used for this class instance.
        """
        # validate
        logging_level = validate.number(logging_level, "logging_level", 
            check_integer=True, lower_bound=10, upper_bound=50)

        # define logger
        logger = logging.getLogger(str(np.random.normal(0, 1)))
        logger.setLevel(logging_level)

        # define formatter
        format = "%(asctime)s - %(levelname)-8s - %(funcName)s: %(message)s"
        formatter = logging.Formatter(format)
        
        # define console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(formatter)

        # add console handler
        logger.addHandler(console_handler)
        logger.propagate = False
        
        # set
        self.logger = logger

 
    def _determine_circular_radius(self) -> None:
        """
        This method is used to determine the circular radius (i.e. when fy
        and fx are both equal to one).
        """
        self.circular_radius = np.hypot(1/2, self.parameters.dy)

 
    def _determine_shear(self) -> None:
        """
        This method is used to determine the shear parameter that describes
        the x-shearing of a circle to form an ellipse centred on (dx, dy) 
        that passes through (+-1, 0).
        """
        # extract parameters
        dx = self.parameters.dx
        dy = self.parameters.dy
        num_dy, num_dx = self.parameters.slice_shape

        # determine dx and dy grids
        dx_grid = np.repeat(dx, num_dy, 0)
        dy_grid = np.repeat(dy, num_dx, 1)
        mask = (dy_grid != 0)

        # determine shear
        shear = np.nan * np.ones((num_dy, num_dx))[:, :, None]
        shear[mask] = -dx_grid[mask] / dy_grid[mask]

        # correct origin
        origin = (dy_grid == 0) * (dx_grid == 0)
        shear[origin] = 0

        if np.sum(origin) == 1:
            self.logger.debug("origin corrected")

        # set shear
        self.shear = shear

 
    def _generate_fxy_array(self, 
            max_value: float, 
            shift: int = 1
        ) -> np.ndarray:
        """
        This method is used to set a scale factor in the x or the y direction.

        Parameters
        ----------
        max_value : float
            This is the maximum value of the corresponding grid direction.
        shift : int
            This is from which value the fx array starts [default = 1]

        Returns
        -------
        fxy_array : np.ndarray (float 1-D)
            Either a new fx or fy array to be used for building the grid.
        """
        # determine bounds of the array array
        min_value = (shift - 1) * max_value
        max_value = shift * max_value
        
        if shift == 1:
            min_value = 1

        fxy_array = np.linspace(min_value, max_value, 
            self._num_fxfy)[None, None, :]

        return fxy_array

 
    def _determine_fx(self, fy: np.ndarray) -> np.ndarray:
        """
        This method is used to determine the x-direction scale factor fx based
        on the fy array passed.

        Parameters
        ----------
        fy : np.ndarray (float 1-D)
            Value of y-direction scale factors for which to calculate fx.

        Returns
        -------
        fx : np.ndarray (float 2-D)
            Value of x-direction scale factors corresponding to input fy.
        """
        # extract necessary parameters
        dy = self.parameters.dy

        # determine the numerator and denominator of the fx function
        numerator = 1
        denominator = fy**2 * (1 + 4 * dy**2) - 4 * dy**2

        # determine values that are real
        mask = (denominator <= 0)

        # determine fx
        fx = np.nan * np.ones(mask.shape)
        fx[~mask] = np.sqrt(numerator / denominator[~mask])
        fx = fx * fy

        return fx
    
 
    def _determine_fy(self, fx: np.ndarray) -> np.ndarray:
        """
        This method is used to determine the y-direction scale factor fy based
        on the fx array passed.

        Parameters
        ----------
        fx : np.ndarray (float 1-D)
            Value of x-direction scale factors for which to calculate fy.

        Returns
        -------
        fy : np.ndarray (float 2-D)
            Value of y-direction scale factors corresponding to input fx.
        """
        # extract parameters
        dy = self.parameters.dy

        # determine the numerator and denominator of the fy function
        numerator = 4 * fx**2 * dy**2
        denominator = fx**2 * (1 + 4 * dy**2) - 1
        mask = (denominator <= 0)

        # determine fy
        fy = np.nan * np.ones(mask.shape)
        fy[~mask] = np.sqrt(numerator[~mask] / denominator[~mask])
        
        # correct the origin
        origin = (self.parameters.dy == 0) * (fx == 1)
        fy[origin] = 1
        if np.sum(origin) == 1:
            self.logger.debug("origin corrected")

        return fy

 
    def _calculate_vertex_and_covertex_angles(self, 
            fx: np.ndarray, 
            fy: np.ndarray, 
            shear_value: float
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to determine the angles in the parametric equation
        of the ellipse where dR/dtheta = 0 i.e. the (co)vertices.

        Parameters
        ----------
        fx : np.ndarray (float 1-D)
            Contains the x-direction scale factor for the ellipses.
        fy : np.ndarray (float 1-D)
            Contains the y-direction scale factor for the ellipses.
        shear_value : float
            The shear value for this ellipse.

        Returns
        -------
        vertex1 : np.ndarray (float 1-D)
            Contains either the vertex or covertex of the ellipse.
        vertex2 : np.ndarray (float 1-D)
            Contains either the vertex or covertex of the ellipse (opposite to
            vertex1).
        """
        # validate
        fx = validate.array(fx, "fx", dtype="float64")
        fy = validate.array(fy, "fy", dtype="float64")
        shear_value = validate.number(shear_value, "shear_value")

        # determine helper variables
        numerator = 2 * fx * fy * shear_value
        denominator = (shear_value**2 + 1) * fy**2 - fx**2

        # determine vertices
        vertex1 = 0.5 * np.arctan2(numerator, denominator)
        vertex2 = vertex1 + np.pi / 2

        return vertex1, vertex2

 
    def _calculate_xy_from_parameteric_angle(self, 
            fx: np.ndarray, 
            fy: np.ndarray, 
            parametric_angle: np.ndarray, 
            shear_value: float,
            circular_radius_value: float
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method takes the parameteric angle of an ellipse function and
        converts that into the cartesian coordinates of that point. It is
        assumed that the ellipse is centred at (0, 0) has a circular radius,
        which is then scaled by fx and fy and then sheared to an ellipse.
        
        Parameters
        ----------
        fx : np.ndarray (float 1=D)
            Contains the x-direction scale factor for the ellipses.
        fy : np.ndarray (float 1=D)
            Contains the y-direction scale factor for the ellipses.
        parametric_angle : np.ndarray (float 1-D)
            Is the angle for which to calculate the cartesian coordinate
            relative to the ellipse centre.
        shear_value : float
            The shear value for this ellipse.
        circular_radius_value : float
            The circular radius value for this ellipse.

        Returns
        -------
        x_sheared : np.ndarray (float 1-D)
            Contains the x-coordinate of the parametric location on the
            ellipse.
        y_sheared : np.ndarray (float 1-D)
            Contains the y-coordinate of the parametric location on the
            ellipse.
        """
        # validate
        fx = validate.array(fx, "fx", dtype="float64")
        fy = validate.array(fy, "fy", dtype="float64")
        parametric_angle = validate.array(parametric_angle, 
            "parametric_angle", dtype="float64")
        shear_value = validate.number(shear_value, "shear_value")
        circular_radius_value = validate.number(circular_radius_value, 
            "circular_radius_value", lower_bound=0)

        # get circle coordinates
        x_circle = circular_radius_value * np.cos(parametric_angle)
        y_circle = circular_radius_value * np.sin(parametric_angle)

        # get stretched coordinates
        x_stretched = fx * x_circle
        y_stretched = fy * y_circle

        # get sheared coordinates
        x_sheared = x_stretched - shear_value * y_stretched
        y_sheared = y_stretched

        return x_sheared, y_sheared
    
 
    def _determine_disk_parameters(self, 
            fx: np.ndarray, 
            fy: np.ndarray, 
            shear_value: float, 
            circular_radius_value: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method is used to determine the disk parameters of each grid
        point. The disk parameters are the radius, inclination and the tilt.

        Parameters
        ----------
        fx : np.ndarray (float 1-D)
            Contains the x-direction scale factor for the ellipses.
        fy : np.ndarray (float 1-D)
            Contains the y-direction scale factor for the ellipses.
        shear_value : float
            The shear value for this ellipse.
        circular_radius_value : float
            The circular radius value for this ellipse.

        Returns
        -------
        disk_radius : np.ndarray (float 3-D)
            Three dimensional cube containing disk radius for all the grid
            points (dy, dx, rf).
        inclination : np.ndarray (float 3-D)
            Three dimensional cube containing disk inclination for all the
            grid points (dy, dx, rf).
        tilt : np.ndarray (float 3-D)
            Three dimensional cube containing disk tilt for all the grid
            points (dy, dx, rf).
        """
        # get (co)vertex angles
        theta1, theta2 = self._calculate_vertex_and_covertex_angles(
            fx, fy, shear_value
        )

        # get cartesian coordinates
        x1, y1 = self._calculate_xy_from_parameteric_angle(
            fx, fy, theta1, shear_value, circular_radius_value
        )
        x2, y2 = self._calculate_xy_from_parameteric_angle(
            fx, fy, theta2, shear_value, circular_radius_value
        )

        # calculate distances
        distance1 = np.hypot(x1, y1)
        distance2 = np.hypot(x2, y2)

        # determine axes and nan mask
        semimajor_axis = np.maximum(distance1, distance2)
        semiminor_axis = np.minimum(distance1, distance2)
        nan_mask = ~np.isnan(semimajor_axis * semiminor_axis)

        # disk radius is the semi major axis
        disk_radius = semimajor_axis

        # determine the tilt (assume distance1 > distance2 then correct)
        tilt = np.arctan2(y1, x1)
        tilt_mask = distance2[nan_mask] > distance1[nan_mask]
        tilt[nan_mask] = tilt[nan_mask] + tilt_mask * np.pi / 2
        tilt = np.rad2deg(tilt)

        # determine the inclination avoiding divide by 0/inf/nan
        inclination = np.nan * np.ones_like(tilt)
        inclination[nan_mask] = np.arccos(
            semiminor_axis[nan_mask] / semimajor_axis[nan_mask]
        )
        inclination = np.rad2deg(inclination)

        return disk_radius, inclination, tilt

 
    def _initialise_grid_properties(self) -> None:
        """
        This method is used to initialise the grid properties in case that
        is necessary during the building of the grid.
        """
        empty = np.nan * np.ones(self.parameters.grid_shape)

        self.disk_radius = Property(Name.DISK_RADIUS, Unit.ECLIPSE_DURATION,
            np.copy(empty), self.parameters)
        self.inclination = Property(
            Name.INCLINATION, Unit.DEGREE, np.copy(empty), self.parameters
        )
        self.tilt = Property(
            Name.TILT, Unit.DEGREE, np.copy(empty), self.parameters
        )
        self.fx_map = Property(
            Name.FX_MAP, Unit.NONE, np.copy(empty), self.parameters
        )
        self.fy_map = Property(
            Name.FY_MAP, Unit.NONE, np.copy(empty), self.parameters
        )

 
    def _create_temp_save_dir(self) -> None:
        """
        This method is used to create the temporary save folder used for
        intermittent saving during the building of the grid.
        """
        if os.path.exists(TEMP_SAVE_DIR):
            shutil.rmtree(TEMP_SAVE_DIR)
            self.logger.debug("cleared temp folder")
        
        os.mkdir(TEMP_SAVE_DIR)

 
    def _delete_temp_save_dir(self) -> None:
        """
        This method is used to delete the temporary save folder used for
        intermittent saving during the building of the grid.
        """
        if os.path.exists(TEMP_SAVE_DIR):
            shutil.rmtree(TEMP_SAVE_DIR)
            self.logger.debug("deleted temp folder")

 
    def _extend_grid_horizontally(self, 
            fy_full: np.ndarray, 
            disk_radius_full: np.ndarray,
            x: int,
            y: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method is used to determine the fx, fy and disk radius values
        when the grid is extended in the third dimension in scale factor
        space. This is the non-trivial space governed by either fx or fy,
        which will later be converted to rf. This particular method is for
        the horizontal end of the space.

        Parameters
        ----------
        fy_full : np.ndarray
            Contains the full extent of fy values that can be used for the
            interpolation transformation from fy -> rf.
        disk_radius_full : np.ndarray
            Contains the full extent of disk radius values that can be used
            for the interpolation transformation from fy -> rf.
        x : int
            x-index of the grid currently under investigation.
        y : int
            y-index of the grid currently under investigation.

        Returns
        -------
        fy_full : np.ndarray
            Contains the full extent of fy values that can be used for the
            interpolation transformation from fy -> rf.
        disk_radius_full : np.ndarray
            Contains the full extent of disk radius values that can be used
            for the interpolation transformation from fy -> rf.
        """
        # extracting parameters
        circular_radius_value = self.circular_radius[y, 0, 0]
        shear_value = self.shear[y, x, 0]
        max_value = self.parameters.dx.max()
        max_rf = self.parameters.rf.max()
        
        # loop parameters
        shift = 1
        repeat = True
        
        while repeat:
            fx_array = self._generate_fxy_array(max_value, shift)
            fy_array = self._determine_fy(fx_array)
            
            # get extending values
            fx = np.flip(fx_array[0, 0, :])
            fy = np.flip(fy_array[y, 0, :])
            disk_radius, _, _ = self._determine_disk_parameters(
                fx, fy, shear_value, circular_radius_value
            )

            # extend input arrays
            fy_full = np.concatenate((fy, fy_full))
            disk_radius_full = np.concatenate((disk_radius, disk_radius_full))

            # continue?
            disk_radius_fraction = disk_radius / np.nanmin(disk_radius_full)
            if disk_radius_fraction[0] < max_rf:
                shift += 1
            else: 
                repeat = False

        return fy_full, disk_radius_full

 
    def _extend_grid_vertically(self, 
            fy_full: np.ndarray, 
            disk_radius_full: np.ndarray,
            x: int,
            y: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This method is used to determine the fx, fy and disk radius values
        when the grid is extended in the third dimension in scale factor
        space. This is the non-trivial space governed by either fx or fy,
        which will later be converted to rf. This particular method is for
        the vertical end of the space.

        Parameters
        ----------
        fy_full : np.ndarray
            Contains the full extent of fy values that can be used for the
            interpolation transformation from fy -> rf.
        disk_radius_full : np.ndarray
            Contains the full extent of disk radius values that can be used
            for the interpolation transformation from fy -> rf.
        x : int
            x-index of the grid currently under investigation.
        y : int
            y-index of the grid currently under investigation.

        Returns
        -------
        fy_full : np.ndarray
            Contains the full extent of fy values that can be used for the
            interpolation transformation from fy -> rf.
        disk_radius_full : np.ndarray
            Contains the full extent of disk radius values that can be used
            for the interpolation transformation from fy -> rf.
        """
        # extracting parameters
        circular_radius_value = self.circular_radius[y, 0, 0]
        shear_value = self.shear[y, x, 0]
        max_value = self.parameters.dy.max()
        max_rf = self.parameters.rf[-1]
        
        # loop parameters
        shift = 1
        repeat = True
        
        while repeat:
            fy_array = self._generate_fxy_array(max_value, shift)
            fx_array = self._determine_fx(fy_array)
            
            # get extending values
            fx = fx_array[y, 0, :]
            fy = fy_array[0, 0, :]
            disk_radius, _, _ = self._determine_disk_parameters(fx, fy, 
                shear_value, circular_radius_value)

            # extend input arrays
            fy_full = np.concatenate((fy_full, fy))
            disk_radius_full = np.concatenate((disk_radius_full, disk_radius))

            # continue?
            disk_radius_fraction = disk_radius / np.nanmin(disk_radius_full)
            if disk_radius_fraction[-1] < max_rf:
                shift += 1
            else: 
                repeat = False

        return fy_full, disk_radius_full

 
    def _interpolate_scale_values(self,
            fy_full: np.ndarray,
            disk_radius_full: np.ndarray,
            y: int
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        This method is used to convert the third dimension of the grid from
        one based on the non-trivial scale factors (fx/fy) to the intuitive
        radius scale factor, rf
        
        Parameters
        ----------
        fy_full : np.ndarray
            Contains the full extent of fy values that can be used for the
            interpolation transformation from fy -> rf.
        disk_radius_full : np.ndarray
            Contains the full extent of disk radius values that can be used
            for the interpolation transformation from fy -> rf.
        y : int
            y-index of the grid currently under investigation.

        Returns
        -------
        fx_valid : np.ndarray (float 1-D)
            Valid fx values which are used to generate the disk radii such
            that they follow the radius scale.
        fy_valid : np.ndarray (float 1-D)
            Valid fy values which are used to generate the disk radii such
            that they follow the radius scale.
        """
        disk_radius_fraction = disk_radius_full / np.nanmin(disk_radius_full)
        min_arg = np.nanargmin(disk_radius_full)
        rf = self.parameters.rf

        # horizontal (x) stretch
        disk_radius_fraction_x = np.flip(disk_radius_fraction[:min_arg+1])
        fy_x = np.flip(fy_full[:min_arg + 1])
        fy_interp_x = np.flip(np.interp(rf, disk_radius_fraction_x, fy_x))

        # vertical (y) stretch
        disk_radius_fraction_y = disk_radius_fraction[min_arg:]
        fy_y = fy_full[min_arg:]
        fy_interp_y = np.interp(rf, disk_radius_fraction_y, fy_y)

        # get valid fx and fy
        fy_valid = np.concatenate(
            (fy_interp_x[:-1], np.array([fy_full[min_arg]]), fy_interp_y[1:]),
            0
        )
            
        fx_valid = self._determine_fx(fy_valid)[y, 0, :]

        return fx_valid, fy_valid

 
    def _set_grid_property_contrast_parameters(self) -> None:
        """
        This method is used to set the contrast parameters of every grid 
        parameter.
        """
        parameters = [self.disk_radius, self.inclination, self.tilt, 
            self.fx_map, self.fy_map]
        cmaps = ["viridis", "viridis", "twilight", "viridis", "viridis"]
        
        for parameter, cmap in zip(parameters, cmaps):
            parameter.set_contrast_parameters(color_map=cmap)

        self.diagnostic_map.set_contrast_parameters()

        if self.gradients is not None:
            for gradient in self.gradients:
                gradient.set_contrast_parameters()

 
    def _build_grid(self, 
            intermittent_saving: bool, 
            start_y: int = 0
        ) -> None:
        """
        This method is used to build the shallot grid. It does this grid point
        by grid point and ensures that the extent of the grid goes from Rmin 
        to maximum Rf (rf = 1 to rf = user set value). Interpolation then 
        occurs to fill in the gaps accordingly.

        Parameters
        ----------
        intermittent_saving : bool
            This is used to determine whether to save the shallot grid build 
            progress after every y iteration. This is useful for very large
            refined grids.
        start_y : int
            This is the integer to start the y building of the grid at 
            [default = 0 --> new grid].
        """
        # validate
        intermittent_saving = validate.boolean(intermittent_saving, 
            "intermittent_saving")
        start_y = validate.number(start_y, "start_y", check_integer=True, 
            lower_bound=0, upper_bound=len(self.parameters.dy))

        # properties
        if start_y == 0:
            self._initialise_grid_properties()

        if intermittent_saving:
            self._create_temp_save_dir()

        # extract parameters
        num_y, num_x = self.parameters.slice_shape

        for y in range(start_y, num_y):
            self.logger.info(f"building y = {y+1}/{num_y}")

            for x in tqdm(range(num_x)):
                # extract relevant values
                circular_radius_value = self.circular_radius[y, 0, 0]
                shear_value = self.shear[y, x, 0]

                if np.isnan(shear_value):
                    if self.diagnostics:
                        empty = np.array([np.nan])
                        self.diagnostics.save_diagnostic(y, x, empty, empty)

                    continue

                # define fx and disk radius arrays
                fy = np.array([])
                disk_radius = np.array([])

                # generate the valid fx and fy necessary to construct rf grid
                fy, disk_radius = self._extend_grid_horizontally(
                    fy, disk_radius, x, y
                )
                fy, disk_radius = self._extend_grid_vertically(
                    fy, disk_radius, x, y
                )
                fx_valid, fy_valid = self._interpolate_scale_values(
                    fy, disk_radius, y
                )
                
                # determine proper grid parameters and set them
                disk_radius_valid, inclination_valid, tilt_valid = (
                    self._determine_disk_parameters(
                        fx_valid, fy_valid, shear_value, circular_radius_value
                    )
                )

                # set
                self.disk_radius.data[y, x] = disk_radius_valid
                self.inclination.data[y, x] = inclination_valid
                self.tilt.data[y, x] = tilt_valid
                self.fx_map.data[y, x] = fx_valid
                self.fy_map.data[y, x] = fy_valid

                # save diagnostic
                if self.diagnostics:
                    self.diagnostics.save_diagnostic(y, x, fy, disk_radius)

            if intermittent_saving:
                self.save(TEMP_SAVE_DIR, y_value=y)

        self._delete_temp_save_dir()

 
    def get_disk_radius(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the disk radius.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        disk_radius : np.ndarray (float 3-D)
            This contains the disk radius values for all disks investigated 
            [t_ecl].
        """
        disk_radius = self.disk_radius.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            disk_radius[combined_mask] = np.nan
        
        return disk_radius

 
    def get_inclination(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the inclination.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        inclination : np.ndarray (float 3-D)
            This contains the inclination values for all disks investigated
            [deg].
        """
        inclination = self.inclination.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            inclination[combined_mask] = np.nan
        
        return inclination

 
    def get_tilt(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the tilt.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        tilt : np.ndarray (float 3-D)
            This contains the tilt values for all disks investigated [deg].
        """
        tilt = self.tilt.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            tilt[combined_mask] = np.nan

        return tilt

 
    def get_fx_map(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the fx map.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        fx_map : np.ndarray (float 3-D)
            This contains the fy values for all disk parameter cube grid
            points.
        """
        fx_map = self.fx_map.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            fx_map[combined_mask] = np.nan

        return fx_map
    
 
    def get_fy_map(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the fy map.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        fy_map : np.ndarray (float 3-D)
            This contains the fy values for all disk parameter cube grid
            points.
        """
        fy_map = self.fy_map.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            fy_map[combined_mask] = np.nan

        return fy_map

 
    def _diagnose_fxfy_resolution(self) -> None:
        """
        This method generates a diagnosis cube so that one can determine how
        reliable the shallot grid radius fraction interpolation is. It does
        this by comparing the calculated radius fraction with the expected
        radius fraction. In an ideal situation the whole cube should be 0 and
        NaNs.
        """
        # generate disk radius fraction
        min_arg = len(self.parameters.rf) - 1
        minimum_disk_radius = self.disk_radius.data[:, :, min_arg][:, :, None]
        disk_radius_fraction = self.disk_radius.data / minimum_disk_radius
        expected = self.parameters.rf_array[None, None, :]
        
        diagnostic_values = disk_radius_fraction - expected
        self.diagnostic_map = Property(
            Name.DIAGNOSTIC_MAP, 
            Unit.ECLIPSE_DURATION, 
            diagnostic_values, 
            self.parameters
        )

        maximum_deviation = np.nanmax(np.abs(diagnostic_values))
        self.logger.info(f"maximum deviation is {maximum_deviation:.4f} - "
            "explore by plotting")

 
    def get_diagnostic_map(self, 
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method retrieves the diagnostic map.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        diagnostic_map : np.ndarray (float 3-D)
            This contains the fy values for all disk parameter cube grid
            points.
        """
        diagnostic_map = self.diagnostic_map.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            diagnostic_map[combined_mask] = np.nan

        return diagnostic_map

 
    def _fill_quadrants(self, 
            disk_property: Property, 
            is_tilt: bool = False
        ) -> Property:
        """
        This method is used to extend the grid (Q1) to all quadrants to obtain
        a more complete picture. This method can be used for disk radius,
        inclination, tilt, fxMap and fyMap. It is not suitable for gradients.

        Parameters
        ----------
        disk_property : Property
            Contains the property that will be extended to fill all
            quadrants. Note that this method should not be used for gradients.
        is_tilt : bool
            This parameter is defines whether or not the input parameter is
            the disk tilt, as this adjusts the values of Q2 and Q4.

        Returns
        -------
        extended_property : Property
            Contains an extended version of the input parameter that now fills
            Q2, Q3 and Q4.
        """
        # instantiate full parameter
        ny, nx, nr = disk_property.data.shape
        full_property = np.zeros((2 * ny, 2 * nx, nr))
        final_shape = (2 * ny - 1, 2 * nx -1, nr)

        # validate
        if self.parameters.grid_shape != final_shape:
            raise AttributeError(f"the new parameter shape {final_shape}"
                f" doesn't match the grid shape {self.parameters.grid_shape}")

        # determine the quadrants
        Q1 = disk_property.data
        Q2 = np.fliplr(disk_property.data)
        Q3 = np.flipud(Q2)
        Q4 = np.flipud(disk_property.data)

        # adjust for tilt
        if is_tilt:
            Q2 = 180 - Q2
            Q4 = 180 - Q4

        # fill full parameter
        full_property[ny:, nx:] = Q1
        full_property[ny:, :nx] = Q2
        full_property[:ny, :nx] = Q3
        full_property[:ny, nx:] = Q4

        # remove duplicate row and column
        full_property = np.delete(full_property, ny, axis=0)
        full_property = np.delete(full_property, nx, axis=1)

        # generate new disk property
        extended_property = Property(disk_property.name, 
            disk_property.unit, full_property, self.parameters)

        if is_tilt:
            extended_property.set_contrast_parameters(color_map="twilight")
        
        return extended_property

 
    def _extend_grid(self) -> None:
        """
        This method extends the grid to all four quadrants and extends the dx
        and dy values accordingly.
        """
        # grid parameters
        try:
            self.parameters.extend_grid()
        except OriginMissingError as ex:
            self.logger.debug(str(ex))
            return

        # helper variables recalculated
        self._determine_circular_radius()
        self._determine_shear()

        # properties are extended
        self.disk_radius = self._fill_quadrants(self.disk_radius)
        self.inclination = self._fill_quadrants(self.inclination)
        self.tilt = self._fill_quadrants(self.tilt, is_tilt=True)
        self.fx_map = self._fill_quadrants(self.fx_map)
        self.fy_map = self._fill_quadrants(self.fy_map)

        if self.diagnostics is not None:
            self.diagnostics.extend()

 
    def _determine_disk_gradients(self, positions: np.ndarray) -> None:
        """
        This method finds the local tangent of a scaled-down version of the 
        ellipse that is centred at (dx, dy) and passes through the points 
        +-(1/2, 0). It is scaled down such that you have a scaled-down 
        concentric ellipse that passes through (positions, 0). This is then
        converted to a disk gradient.

        Parameters
        ----------
        positions : np.ndarray (float 1-D)
            Position values at which to determine the local tangents [t_ecl].
        """
        positions = validate.array(
            positions, "positions", num_dimensions=1, dtype="float64"
        )

        # extract parameters
        fx_map = self.fx_map.data
        fy_map = self.fy_map.data

        # instantiate list of gradients
        if self.gradients is None:
            self.gradients: list[Gradient] = []

        # loop through positions
        for position in tqdm(positions):
            # shift coordinates to frame centred on the ellipse centre
            x = position - self.parameters.dx
            y = -self.parameters.dy

            # get the slopes
            numerator = -self.shear * fy_map**2 * y - fy_map**2 * x
            denominator = ((self.shear**2 * fy_map**2 + fx_map**2) 
                * y + self.shear * fy_map**2 * x)
            
            # prevent divide by 0 error
            tangent = np.nan * np.ones(self.parameters.grid_shape)
            mask = (denominator != 0)
            tangent[mask] = (numerator[mask] / denominator[mask])

            # set tangent
            disk_gradient = np.abs(np.sin(np.arctan2(tangent, 1)))
            
            self.gradients.append(
                Gradient(disk_gradient, self.parameters, position)
            )

 
    def _determine_orbital_scale(self) -> float:
        """
        This method is used to set a light curve gradient scale factor that is
        dependent on the transverse velocity of the occulting object and the
        limb darkening parameter of the star.

        Return
        ------
        orbital_scale : float
            This is the scale factor that is generated depending on the input
            transverse velocity of the occulter and the limb darkening of the
            star.
        """
        u = self._limb_darkening

        limb_darkening_scale = ((6 - 2 * u) / (12 - 12 * u + 3 * np.pi * u))
        velocity_scale = np.pi / self._transverse_velocity

        orbital_scale = limb_darkening_scale * velocity_scale
        return orbital_scale

 
    def _set_measured_gradients_and_masks(self, 
            light_curve_gradients: np.ndarray,
            light_curve_gradient_errors: np.ndarray = None,
            transmission_changes: np.ndarray = None
        ) -> None:
        """
        This method is used to convert the gradients measured in a light curve
        [L*/t_ecl] into the disk gradients, so that they can be compared to the 
        theoretical disk gradients determined by .determineDiskGradients() 
        method. 

        Parameters
        ----------
        light_curve_gradients : np.ndarray (float 1-D)
            Gradients measured from the light curve [L*/day].
        light_curve_gradient_errors : np.ndarray (float 1-D)
            Errors on the gradients measured from the light curve [L*/day]
            [default = None].
        transmission_changes : np.ndarray (float 1-D)
            Contains the full change in transmission over the course of the 
            measured gradient (see notes).

        Notes
        -----
        transmission_changes should be used carefully. These changes provide a
        much more restricted parameter space but there are several things to
        note.
        
        1)  The transmission change must be due to the complete occulting of
            the star by the new transiting ring. This means that the new ring
            can"t partially cover the star.
        2)  The transmission change given must be greater than or equal to the
            actual transmission change.

        The above two points are to prevent over compensation of the gradients
        that will remove potential viable solutions for the light curve fit.
        """
        if light_curve_gradient_errors is None:
            light_curve_gradient_errors = np.ones_like(light_curve_gradients)
        light_curve_gradient_errors = validate.array(
            light_curve_gradient_errors, "light_curve_gradient_errors", 
            lower_bound=0., dtype="float64", num_dimensions=1
        )

        if transmission_changes is None:
            transmission_changes = np.ones_like(light_curve_gradients)
        transmission_changes = validate.array(
            transmission_changes, "transmission_changes", lower_bound=0, 
            upper_bound=1, dtype="float64", num_dimensions=1
        )
        
        arrays_list = [light_curve_gradients, light_curve_gradient_errors, 
            transmission_changes]
        names_list = ["light_curve_gradients", "light_curve_gradient_errors",
            "transmission_changes"]
        validate.same_shape_arrays(arrays_list, names_list)

        num_new_gradients = len(light_curve_gradients)
        orbital_scale = self._determine_orbital_scale()
        measured_gradients = np.abs(light_curve_gradients)
        zip_params = (self.gradients[-num_new_gradients:], measured_gradients,
            light_curve_gradient_errors, transmission_changes)

        for gradient, measured_gradient, measured_error, transmission_change \
                in zip(*zip_params):
            gradient.determine_mask(measured_gradient, orbital_scale,
                transmission_change, measured_error)

 
    def _reset_gradient_fit(self):
        """
        This method is used to reset the gradient fit after changes are made
        to the gradients list if necessary.
        """
        if not self.gradient_fit:
            self.gradient_fit = None
            self.logger.info("gradient fit has been cleared")

 
    def add_gradients(self, 
            times: np.ndarray, 
            light_curve_gradients: np.ndarray,
            light_curve_gradient_errors: np.ndarray = None,
            transmission_changes: np.ndarray = None
        ) -> None:
        """
        This method is used to convert the gradients measured in a light curve
        [L*/t_ecl] into the right form to be compared with the theoretical
        gradients determined by ._determine_disk_gradients() method. To do 
        this other astrophysical parameters are required.

        Parameters
        ----------
        times : np.ndarray (float 1-D)
            Time values at which to lightCurveGradients were measured [day].
        light_curve_gradients : np.ndarray (float 1-D)
            Gradients measured from the light curve [L*/day].
        light_curve_gradient_errors : np.ndarray (float 1-D)
            Errors on the gradients measured from the light curve [L*/day]
            [default = None].
        transmission_changes : np.ndarray (float 1-D)
            Contains the full change in transmission over the course of the 
            measured gradient (see notes) [default = None].

        Notes
        -----
        transmission_changes should be used carefully. These changes provide a
        much more restricted parameter space but there are several things to
        note.
        
        1)  The transmission change must be due to the complete occulting of
            the star by the new transiting ring. This means that the new ring
            can't partially cover the star.
        2)  The transmission change given must be greater than or equal to the
            actual transmission change.

        The above two points are to prevent over compensation of the gradients
        that will remove potential viable solutions for the light curve fit.
        """
        if ((self._limb_darkening is None) or (self._eclipse_duration is None)
                or (self._transverse_velocity is None)):
            self.logger.info("eclipse parameters missing, "
                "use .set_eclipse_parameters()")
            return
        
        # validate
        times = validate.array(times, "times", num_dimensions=1, 
            dtype="float64")
        light_curve_gradients = validate.array(
            light_curve_gradients, "light_curve_gradients", lower_bound=0., 
            dtype="float64", num_dimensions=1
        )
               
        arrays_list = [times, light_curve_gradients]
        names_list = ["times", "light_curve_gradients"]
        validate.same_shape_arrays(arrays_list, names_list)

        self.logger.info(f"adding {len(times)} gradients")

        # convert gradients
        positions = -times / self._eclipse_duration
        self._determine_disk_gradients(positions)

        self._set_measured_gradients_and_masks(light_curve_gradients, 
            light_curve_gradient_errors, transmission_changes)
        
        self._reset_gradient_fit()

 
    def remove_gradients(self, indices: np.ndarray) -> None:
        """
        This method is used to remove gradients from the gradients list.
        
        Parameters
        ----------
        indices : np.ndarray (int 1-D)
            Contains all the indices that should be removed from the gradients
            list.
        """
        if not self.gradients:
            self.logger.info("no gradients found")
            return

        indices = validate.array(indices, "indices", num_dimensions=1, 
            lower_bound=0, upper_bound=len(self.gradients) - 1, dtype="int64")
        sorted_indices = np.flip(np.sort(indices))
        
        for index in sorted_indices:
            del self.gradients[index]

        self._reset_gradient_fit()

 
    def set_eclipse_parameters(self,
            eclipse_duration: float,
            transverse_velocity: float,
            limb_darkening: float
        ) -> None:
        """
        This method is used to set the astrophysical parameters necessary for
        adding gradients to the shallot grid.

        Parameters
        ----------
        eclipse_duration : float
            Duration of the eclipse [day].
        transverse_velocity : float
            This is the velocity of the disk, which can be determined from the
            light curve or by other means [R*/day].
        limb_darkening : float
            This is the parameter that defines the limb darkening according to
            the linear limb darkening law [-].
        """
        self._eclipse_duration = validate.number(eclipse_duration, 
            "eclipse_duration", lower_bound=0.)
        self._transverse_velocity = validate.number(transverse_velocity, 
            "transverse_velocity", lower_bound=0.)
        self._limb_darkening = validate.number(limb_darkening, 
            "limb_darkening", lower_bound=0., upper_bound=1.)
        
        # gradients and gradient fit reset due to eclipse duration dependency
        if self.gradients is not None:
            self.gradients = None
            self.logger.info("gradients set to None due to dependency on "
                "eclipse duration")
            self._reset_gradient_fit()

 
    def update_gradient_scaling(self, 
            transverse_velocity: float = None, 
            limb_darkening: float = None, 
            transmission_changes: np.ndarray = None
        ) -> None:
        """
        This method is used to update the scaling factors (light curve 
        gradient and transmission change) and then subsequently updates the
        measured gradients and the gradient masks. This is due to the fact
        that the measured gradients depend on astrophysical parameters that
        may change based on new information.

        Parameters
        ----------
        transverse_velocity : float
            This is the velocity of the disk, which can be determined from the
            light curve or by other means [R*/day] [default = None].
        limb_darkening : float
            This is the parameter that defines the limb darkening according to
            the linear limb darkening law [-] [default = None].
        transmission_changes : np.ndarray (float 1-D)
            Contains the full change in transmission over the course of the 
            measured gradient (see notes) [default = None].

        Notes
        -----
        transmission_changes should be used carefully. These changes provide a
        much more restricted parameter space but there are several things to
        note.
        
        1)  The transmission change must be due to the complete occulting of
            the star by the new transiting ring. This means that the new ring
            can't partially cover the star.
        2)  The transmission change given must be greater than or equal to the
            actual transmission change.

        The above two points are to prevent over compensation of the gradients
        that will remove potential viable solutions for the light curve fit.
        """
        # update orbital scale
        update_orbital_scale = False
        
        if transverse_velocity is not None:
            self._transverse_velocity = transverse_velocity
            update_orbital_scale = True
        
        if limb_darkening is not None:
            self._limb_darkening = limb_darkening
            update_orbital_scale = True

        # update transmission change scale
        update_transmission_change = False
        if transmission_changes is not None:
            transmission_changes = validate.array(
                transmission_changes, "transmission_changes", dtype="float64",
                num_dimensions=1, lower_bound=-1, upper_bound=1
            )

            arrays_list = [transmission_changes, np.array(self.gradients)]
            names_list = ["transmission_changes", "gradients"]
            validate.same_shape_arrays(arrays_list, names_list)
            
            update_transmission_change = True

        if not update_transmission_change and not update_orbital_scale:
            self.logger.info("no changes passed")
            return
        
        for k, gradient in enumerate(self.gradients):
            # select orbital scale
            if update_orbital_scale:
                orbital_scale = self._determine_orbital_scale()
            else:
                orbital_scale = gradient.orbital_scale

            # select transmission scale
            if update_transmission_change:
                transmission_change = transmission_changes[k]
            else:
                transmission_change = gradient.transmission_change

            # update mask
            gradient.determine_mask(gradient.measured_gradient, orbital_scale, 
                transmission_change, gradient.measured_error)
        
        self._reset_gradient_fit()
        
 
    def get_gradients(self,
            masked: bool = False,
            property_masked: bool = True
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method retrieves the disk gradients.

        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = False].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].
        
        Returns
        -------
        positions : np.ndarray (float 1-D)
            This array specifies the positions the gradients were measured at 
            [t_ecl].
        scaled_gradients : np.ndarray (float 1-D)
            Contains the scaled gradients, which have been converted from
            the measured light curve gradients by scaling by the eclipse 
            duration, transverse velocity and change in transmission.
        measured_errors : np.ndarray (float 1-D)
            Contains the measured gradient errors, which are equivalent to the
            measured light curve gradient errors.
        disk_gradients : np.ndarray (float 4-D)
            Projected gradients of the ellipses investigated at the provided 
            positions (x, 0), where x [t_ecl].
        """
        num_gradients = len(self.gradients)
        positions = np.zeros(num_gradients)
        scaled_gradients = np.zeros(num_gradients)
        measured_errors = np.zeros(num_gradients)
        disk_gradients = np.zeros((num_gradients,) + 
            self.parameters.grid_shape)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
        else:
            combined_mask = np.zeros(self.parameters.grid_shape).astype(bool)
        
        for k, gradient in enumerate(self.gradients):
            positions[k] = gradient.position
            scaled_gradients[k] = gradient.get_scaled_gradient()
            measured_errors[k] = gradient.measured_error
            disk_gradients[k] = gradient.get_data(property_masked)
            
            disk_gradients[k][combined_mask] = np.nan

        return positions, scaled_gradients, measured_errors, disk_gradients

 
    def determine_gradient_fit(self, weighted: bool = True) -> None:
        """
        This method is used to determine the rms distance between the measured
        gradients and the disk gradients as a measure of how well a given grid
        point serves as a solution to the provided data

        Parameters
        ----------
        weighted : bool
            Whether or not to weigh the individual r.m.s. values of the
            gradient fit. This is done by the square of the error.
        """
        if self.gradients is None:
            self.logger.info("no gradients present")
            return
        
        data = np.zeros(self.parameters.grid_shape)
        
        self.logger.info(f"determining {len(self.gradients)} gradient rms's")

        for gradient in tqdm(self.gradients):
            # for type hints
            gradient: Gradient
            
            if weighted:
                weight = gradient.measured_error**2
            else:
                weight = 1
            data += (gradient.get_data(masked=False) - 
                gradient.get_scaled_gradient())**2 / weight
        
        self.gradient_fit = Property(
            Name.GRADIENT_FIT, Unit.NONE, data, self.parameters
        )
        self.gradient_fit.set_mask(np.isnan(data))

 
    def get_gradient_fit(self,
            masked: bool = True,
            property_masked: bool = True
        ) -> np.ndarray:
        """
        This method is used to retrieve the data in the gradient fit
        
        Parameters
        ----------
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].
        property_masked : bool
            This determines whether the mask attached to the property is
            applied or not [default = True].

        Returns
        -------
        gradient_fit : np.ndarray (float 3-D)
            This contains the gradient fit for all disks investigated [-].
        """
        if not self.gradient_fit:
            self.logger.info("no gradient fit found")
            return

        gradient_fit = self.gradient_fit.get_data(property_masked)

        masked = validate.boolean(masked, "masked")
        if masked:
            combined_mask = self.get_combined_mask()
            gradient_fit[combined_mask] = np.nan

        return gradient_fit

 
    def extract_solutions(self, 
            num_solutions: int = None
        ) -> tuple[
                np.ndarray, 
                np.ndarray, 
                np.ndarray, 
                np.ndarray, 
                np.ndarray,
                np.ndarray
            ]:
        """
        This method is used to extract the best gradient fit solutions for
        use in analysis.

        Parameters
        ----------
        num_solutions : int
            The number of solutions to extract [default = None -> all].

        Returns
        -------
        rms : np.ndarray (float 1-D)
            This is the rms value of the given grid point
        disk_radius : np.ndarray (float 1-D)
            Disk radius for the given solution [t_ecl].
        inclination: np.ndarray (float 1-D)
            Inclination for the given solution [deg].
        tilt : np.ndarray (float 1-D)
            Tilt for the given solution [deg].
        dx : np.ndarray (float 1-D)
            Dx for the given solution [t_ecl].
        dy : np.ndarray (float 1-D)
            Dy for the given solution [t_ecl].
        """
        if self.gradient_fit is None:
            self.logger.info("no gradient fit present")
            return

        max_solutions = np.sum(~self.get_combined_mask())
        
        if num_solutions is None:
            num_solutions = max_solutions
        num_solutions = validate.number(num_solutions, "num_solutions",
            lower_bound=0, upper_bound=max_solutions, check_integer=True)
        
        self.logger.info(f'extracting {num_solutions} out of {max_solutions}'
            ' possible solutions')

        rms = np.zeros(num_solutions)
        disk_radius = np.zeros(num_solutions)
        inclination = np.zeros(num_solutions)
        tilt = np.zeros(num_solutions)
        dx = np.zeros(num_solutions)
        dy = np.zeros(num_solutions)

        gradient_fit = self.get_gradient_fit()
        for k in tqdm(range(num_solutions)):
            (y, x, r) = np.unravel_index(
                np.nanargmin(gradient_fit), gradient_fit.shape
            )
            disk_data, _, _ = self.get_grid_point_data(y, x, r, masked=False)
            
            # fill data
            rms[k] = gradient_fit[y, x, r]
            disk_radius[k] = disk_data[0]
            inclination[k] = disk_data[1]
            tilt[k] = disk_data[2]
            dx[k] = disk_data[3]
            dy[k] = disk_data[4]

            # remove best solution
            gradient_fit[y, x, r] = np.nan

        return rms, disk_radius, inclination, tilt, dx, dy

 
    def generate_hill_radius_mask(self, hill_radius: float) -> None:
        """
        This method masks the disk parameters according to the Hill radius,
        which is needed to fulfill a stability criterion of the disk.
        
        Parameters
        ----------
        hill_radius : float
            Value of the Hill radius, the maximum stable size of the disk 
            [t_ecl].
        """
        hill_radius = validate.number(
            hill_radius, "hill_radius", lower_bound=0.
        )

        # determine mask
        hill_radius_mask = np.ones(self.parameters.grid_shape).astype(
            bool)
        mask_values = ~np.isnan(self.disk_radius.data)
        hill_radius_mask[mask_values] = (self.disk_radius.data[mask_values] > 
            hill_radius)

        self.disk_radius.set_mask(hill_radius_mask)

 
    def get_combined_mask(self) -> np.ndarray:
        """
        This method is used to combine all available masks to a single mask
        that can be applied to each disk property.

        Returns
        -------
        combined_mask : np.ndarray (bool 3-D)
            Mask where True values are bad, and False values are good.
        """
        # retrieve masks
        all_masks = [self.disk_radius.mask, self.inclination.mask, 
            self.tilt.mask]
        if self.gradients is not None:
            for gradient in self.gradients:
                all_masks.append(gradient.mask)
        
        # generate combined mask
        combined_mask = np.zeros(self.parameters.grid_shape)
        for mask in all_masks:
            if mask is not None:
                combined_mask += mask.astype(float)

        return combined_mask.astype(bool)

 
    def determine_closest_grid_point(self, 
            disk_radius: float,  
            inclination: float, 
            tilt: float,
            impact_parameter: float, 
            masked: bool = True
        ) -> tuple[
                tuple[float, float, float], 
                tuple[int, int, int], 
                float
            ]:
        """
        This method determines the closest shallot grid point to the input
        parameters given. Note that the max_occultation_time (~dx) is ignored
        here as it does not influence the shape of the light curve.

        Parameters
        ----------
        disk_radius : float
            Actual size of the disk [day].
        inclination : float
            Actual inclination of the disk [deg].
        tilt : float
            Actual tilt of the disk [deg].
        impact_parameter : float
            Actual impact parameter (dy) of the disk [day].
        masked : bool
            Determines whether or not to use the combined mask if available
            [default = True].

        Returns
        -------
        closest_coordinates : tuple
            (dy, dx, rf) values of the closest solution.
        closest_indices : tuple
            Contains the indices for the closest solution.
        minimum_distance : float
            Is the minimum rms distance from input location to closest grid 
            point.
        """
        # validate
        disk_radius = validate.number(disk_radius, "disk_radius", 
            lower_bound=0)
        inclination = validate.number(inclination, "inclination", 
            lower_bound=0, upper_bound=90)
        tilt = validate.number(tilt, "tilt", lower_bound=-180, 
            upper_bound=180)
        impact_parameter = validate.number(impact_parameter, 
            "impact_parameter")
        masked = validate.boolean(masked, "masked")
        
        # get sjalot parameters
        grid_disk_radius = self.get_disk_radius(masked)
        grid_inclination = self.get_inclination(masked)
        grid_tilt = self.get_tilt(masked)
        
        # get grid parameters
        dx = self.parameters.dx.flatten()
        x_min, x_max = dx[0], dx[-1]
        x_num = len(dx)
        
        dy = self.parameters.dy.flatten()
        y_min, y_max = dy[0], dy[-1]
        y_num = len(dy)

        rf_array = self.parameters.rf_array
        
        # construct position grid
        yy, xx = np.mgrid[:y_num, :x_num]
        xx = xx / x_num * (x_max - x_min) + x_min
        yy = yy / y_num * (y_max - y_min) + y_min

        # ensure selecting the right portion of the grid
        if impact_parameter < 0:
            yy[yy>0] = 1e7
        elif impact_parameter > 0:
            yy[yy<0] = -1e7

        # determine the distance
        distance = np.sqrt(
            (grid_disk_radius - disk_radius)**2 + 
            (grid_inclination - inclination)**2 + 
            (grid_tilt - tilt)**2 + 
            (yy[:, :, None] - impact_parameter)**2
        )
        
        # determine the best indicies
        closest_indices = np.unravel_index(np.nanargmin(distance), 
            distance.shape)
        
        y_index, x_index, rf_index = closest_indices
        closest_coordinates = (dy[y_index], dx[x_index], rf_array[rf_index])
        
        minimum_distance = np.nanmin(distance)

        # log information
        self.logger.info("property_name: input_value --> grid_value")
        self.logger.info(f"disk_radius: {disk_radius:.2f} --> "
            f"{grid_disk_radius[closest_indices]:.2f}")
        self.logger.info(f"inclination: {inclination:.2f} --> "
            f"{grid_inclination[closest_indices]:.2f}")
        self.logger.info(f"tilt: {tilt:.2f} --> "
            f"{grid_tilt[closest_indices]:.2f}")
        self.logger.info(f"impact_parameter: {impact_parameter:.2f} --> "
            f"{yy[closest_indices[:2]]:.2f}")
        self.logger.info(f"'fit': {minimum_distance:.6f}")

        return closest_coordinates, closest_indices, minimum_distance

 
    def get_grid_point_data(self, 
            y_index: int, 
            x_index: int, 
            rf_index: int,
            masked: bool = True
        ) -> tuple[
                tuple[float, float, float, float, float],
                tuple[float, float, float, float],
                tuple[np.ndarray, np.ndarray]
            ]:
        """
        This method is used to extract all the relevant grid information
        for a given grid point
        
        Parameters
        ----------
        y_index : int
            The index of the first dimension that is related to the impact 
            parameter [t_ecl].
        x_index : int
            The index of the second dimension that is related to the x-shift
            parameter [t_ecl].
        rf_index : int
            The index of the third dimension that is related to the growth
            factor of the disk radius [-].
        masked : bool
            This determines whether the combined mask is applied or not 
            [default = True].

        Returns
        -------
        disk_data : tuple (float 5)
            disk_radius : float [t_ecl]
            inclination : float [deg]
            tilt : float [deg]
            dx : float [t_ecl]
            dy : float [t_ecl]
        grid_data : tuple (float 4)
            rf : float [-]
            fx : float [-]
            fy : float [-]
            diagnostic : float [t_ecl]
        gradient_data : tuple (np.ndarray 2)
            positions : np.ndarray (float 1-D) [t_ecl]
            gradients : np.ndarray (float 1-D) [-]
        """
        # validations
        max_y, max_x, max_rf = self.parameters.grid_shape
        y_index = validate.number(y_index, "y_index", check_integer=True, 
            lower_bound=0, upper_bound=max_y)
        x_index = validate.number(x_index, "x_index", check_integer=True,
            lower_bound=0, upper_bound=max_x)
        rf_index = validate.number(rf_index, "rf_index", check_integer=True,
            lower_bound=0, upper_bound=max_rf)

        # disk information
        disk_radius = self.get_disk_radius(masked)[y_index, x_index, rf_index]
        inclination = self.get_inclination(masked)[y_index, x_index, rf_index]
        tilt = self.get_tilt(masked)[y_index, x_index, rf_index]
        dx = self.parameters.dx.flatten()[x_index]
        dy = self.parameters.dy.flatten()[y_index]
        disk_data = (disk_radius, inclination, tilt, dx, dy)

        # grid information
        rf = self.parameters.rf_array[rf_index]
        fx = self.get_fx_map(masked)[y_index, x_index, rf_index]
        fy = self.get_fy_map(masked)[y_index, x_index, rf_index]
        diagnostic = self.get_diagnostic_map(masked)[y_index, x_index, 
            rf_index]
        grid_data = (rf, fx, fy, diagnostic)

        # gradient information
        if self.gradients is not None:
            positions = np.zeros(len(self.gradients))
            gradients = np.zeros(len(self.gradients))
            for k, gradient in enumerate(self.gradients):
                positions[k] = gradient.position
                gradients[k] = gradient.get_data(masked)[y_index, x_index, 
                    rf_index]
        else:
            positions = None
            gradients = None
        gradient_data = (positions, gradients)

        return disk_data, grid_data, gradient_data