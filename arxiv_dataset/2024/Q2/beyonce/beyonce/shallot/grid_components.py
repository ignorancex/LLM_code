from __future__ import annotations
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent

import os
import numpy as np
import matplotlib.pyplot as plt

from beyonce.shallot.grid_names import Name
from beyonce.shallot.grid_units import Unit
from beyonce.shallot.grid_parameters import Parameters

import beyonce.validate as validate
from beyonce.shallot.errors import LoadError


class Property:
    """
    This class is used to define the following grid properties:
        disk radius
        inclination
        tilt
        fx map
        fy map
        diagnostic map
    Data manipulation only includes:
        set_mask (set some kind of mask based on the data)
    For plotting purposes:
        set_contrast_parameters (vmin, vmax, color_map and num_colors)
        plot_cube
        plot_slice
    And finally the data can be saved and loaded.
    """

 
    def __init__(self, 
            name: Name, 
            unit: Unit, 
            data: np.ndarray, 
            parameters: Parameters
        ) -> None:
        """
        This is the class initialiser.
        
        Parameters
        ----------
        name : Name
            The name of the grid property.
        unit : Unit
            The unit of the grid property.
        data : np.ndarray (float)
            The grid property data.
        parameters : Parameters
            The vectors that define each dimension of the data cube.
        """
        validate.class_object(name, "name", Name, "Name")
        validate.class_object(unit, "unit", Unit, "Unit")
        validate.class_object(parameters, "parameters", Parameters, 
            "Parameters")
        
        self.name = name
        self.unit = unit
        self.parameters = parameters
        self.data = validate.array(data, "data", dtype="float64",
            num_dimensions=3)
        data_structure = np.ones(parameters.grid_shape)
        validate.same_shape_arrays([data, data_structure], 
            ["data", "data_structure"])
        self.mask: np.ndarray = None

        self.set_contrast_parameters()

 
    def __str__(self) -> str:
        """
        This produces a string representation for the user. It includes
        the grid parameters str representation.
        
        Returns
        -------
        str_string : str
            Representation string for grid property class.
        """
        lines = self.__repr__().split("\n")
        lines.append(self.parameters.__str__())
        
        return "\n".join(lines)

 
    def __repr__(self) -> str:
        """
        This generates a string representation of the grid gradient. 
        
        Returns
        -------
        repr_string : str
            Representation string of the grid property class. This ignores
            the grid parameters object.
        """
        lines: list[str] = [""]
        parameter = f"{self.name.property_name} [{self.unit.symbol}]"

        lines.append(parameter)
        lines.append(28 * "-")
        min_value_string = f'{f"{self.vmin:.4f}":>13}'
        max_value_string = f'{f"{self.vmax:.4f}":>13}'
        mean_value_string = f'{f"{np.nanmean(self.data):.4f}":>13}'
        median_value_string = f'{f"{np.nanmedian(self.data):.4f}":>13}'
        lines.append(f"min value:    {min_value_string}")
        lines.append(f"max value:    {max_value_string}")
        lines.append(f"mean value:   {mean_value_string}")
        lines.append(f"median value: {median_value_string}")

        if self.mask is not None:
            lines.append("")
            fraction_masked = np.sum(self.mask) / np.prod(self.mask.shape)
            fraction_masked_string = f'{f"{100 * fraction_masked:.4f}":>14}'
            lines.append(f"mask [out]:  {fraction_masked_string}%")
        
        return "\n".join(lines)

 
    def set_mask(self, mask: np.ndarray) -> None:
        """
        This method is used to set a mask that can be applied to the data.
        Note that the True values of the mask are invalid and will be replaced
        with NaN's
        
        Parameters
        ----------
        mask : np.ndarray (bool)
            Mask that points to invalid values.
        """
        mask = validate.array(mask, "mask", dtype="bool")
        validate.same_shape_arrays([mask, self.data], ["mask", "data"])
        
        self.mask = mask

 
    def get_data(self, masked: bool = True) -> np.ndarray:
        """
        This method retrieves the data of the property.

        Parameters
        ----------
        masked : boolean
            Whether or not to convert invalid values to NaN's based on the
            available mask [default = True]
        
        Returns
        -------
        data : np.ndarray (float)
            The grid property data.
        """
        masked = validate.boolean(masked, "masked")
        data = np.copy(self.data)

        if masked and self.mask is not None:
            data[self.mask] = np.nan
        
        return data

 
    def set_contrast_parameters(self, 
            vmin: float = None, 
            vmax: float = None,
            color_map: str = None, 
            num_colors: int = None
        ) -> None:
        """
        This method sets the contrast parameters for the plot_cube() and 
        plot_slice() methods.

        Parameters
        ----------
        vmin : float
            This is the lower limit of the color_map to adjust the contrast
            [default = None].
        vmax : float
            This is the upper limit of the color_map to adjust the contrast
            [default = None].
        color_map : string
            This is the name of the matplotlib color_map to be used to colour
            the image [default = "viridis"].
        num_colors : integer
            This is the number of colors the color_map should be divided into.
            This is to make the image easier to interpret [default = 11].
        """
        all_nans = np.sum(np.isnan(self.data)) == np.prod(self.data.shape)
        
        if all_nans:
            self.vmin = None
            self.vmax = None
        else:
            if vmin is None:
                vmin = np.nanmin(self.data)
            self.vmin = validate.number(vmin, "vmin")       
        
            if vmax is None:
                vmax = np.nanmax(self.data)
            self.vmax = validate.number(vmax, "vmax")
        
        if color_map is None:
            color_map = "viridis"
        self.color_map = validate.string(color_map, "color_map")
        
        if num_colors is None:
            num_colors = 11
        self.num_colors = validate.number(num_colors, "num_colors", 
            check_integer=True, lower_bound=2)

 
    def plot_cube(self, 
            axis: int = 2, 
            coordinates: list[tuple[float, float, float]] = None, 
            masked: bool = True
        ) -> None:
        """
        This method plots the values of the data cube in a scrollable grid
        property viewer.

        Parameters
        ----------
        axis : int
            This is the axis along which the cut will be made for the data
            cube. Note that these axes are ordered (dy, dx, rf) [default = 2].
        coordinates : list[tuple (float 3-D)]
            Contains the coordinates of the points to plot separately.
        masked : bool
            Determines whether or not the mask is applied to the data 
            [default = True].
        """
        fig, ax = plt.subplots()
        viewer = Viewer(ax, axis, self, masked, coordinates=coordinates)
        fig.canvas.mpl_connect("scroll_event", viewer.onscroll)
        plt.colorbar(viewer.image)
        plt.show()

 
    def plot_slice(self, 
            axis: int, 
            index: int, 
            ax: plt.Axes = None, 
            masked: bool = True
        ) -> tuple[plt.Axes, AxesImage]:
        """
        This method plots a single slice from a fixed grid property viewer.

        Parameters
        ----------
        axis : integer
            This is the axis along which the cut will be made for the data
            cube. Note that these axes are ordered (dy, dx, fy).
        index : integer
            This is the index of the slice to plot.
        ax : matplotlib.Axes
            This is the object that will contain all the plotted information
            [default = None].
        masked : bool
            Determines whether or not the mask is applied to the data 
            [default = True].

        Returns
        -------
        ax : matplotlib.Axes
            This Axes objects contains all the plot information on it.
        image : mappable
            Used for colorbars.
        """
        if ax is None:
            ax = plt.gca()
        ax = validate.class_object(ax, "ax", plt.Axes, "Axes")

        viewer = Viewer(ax, axis, self, masked, index=index, frozen=True)
        return viewer.ax, viewer.image

 
    def save(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        if not os.path.exists(directory):
            os.mkdir(directory)

        filename_data = f"{self.name.name.lower()}_{self.unit.name.lower()}"
        np.save(f"{directory}/{filename_data}", self.data)
        
        if self.mask is not None:
            filename_mask = f"{filename_data}_mask"
            np.save(f"{directory}/{filename_mask}", self.mask)

        self.parameters.save(directory)


    @classmethod
    def load(cls, directory: str, name: Name, unit: Unit) -> Property:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        grid_property : Property
            This is the loaded object.
        """
        directory = validate.string(directory, "directory")
        validate.class_object(name, "name", Name, "Name")
        validate.class_object(unit, "unit", Unit, "Unit")
        
        try:
            filepath_data = (f"{directory}/{name.name.lower()}_"
                f"{unit.name.lower()}.npy")
            data = np.load(f"{filepath_data}")
            parameters = Parameters.load(directory)

            grid_property = cls(name, unit, data, parameters)
            
            filepath_mask = (f"{directory}/{name.name.lower()}_"
                f"{unit.name.lower()}_mask.npy")
            if os.path.exists(f"{filepath_mask}"):
                mask = np.load(f"{filepath_mask}")
                grid_property.mask = mask
            
            grid_property.set_contrast_parameters()

        except Exception:
            type_string = f"{name.property_name} [{unit.property_unit}]"
            raise LoadError(type_string, directory)

        return grid_property


class Gradient(Property):
    """
    This class is used to define the following grid gradients. Data 
    manipulation only includes:
        determine_mask (set some kind of mask based on a measured_gradient)
    For plotting purposes (from super class):
        set_contrast_parameters (vmin, vmax, color_map and num_colors)
        plot_cube
        plot_slice
    And finally the data can be saved and loaded.
    """


    def __init__(self, 
            data: np.ndarray, 
            parameters: Parameters,
            position: float
        ) -> None:
        """
        This is the class initialiser.
        
        Parameters
        ----------
        data : np.ndarray (float)
            The grid property data.
        grid_parameters : GridParameters
            The vectors that define each dimension of the data cube.
        position : float
            The position associated with this gradient.
        """
        super().__init__(
            Name.GRADIENT, 
            Unit.NONE, 
            data, 
            parameters
        )
        
        self.position = validate.number(position, "position")
        self.measured_gradient: float = None
        self.measured_error: float = None
        self.orbital_scale: float = None
        self.transmission_change: float = None

 
    def __repr__(self) -> str:
        """
        This generates a string representation of the grid gradient. This has
        been overriden to include the position of the grid gradient and
        additional information about the mask.
        
        Returns
        -------
        repr_string : str
            Representation string of the grid gradient subclass. This ignores
            the grid parameters object.
        """
        lines: list[str] = super().__repr__().split("\n")
        
        position = f"{self.position:.4f}".rjust(7)
        lines[1] = lines[1] + f" @ pos = {position}"
        if self.mask is not None:
            measured_gradient = f"{self.measured_gradient:.4f}".rjust(6)
            lines.append(f"measured gradient:   {measured_gradient}")
            measured_error = f"{self.measured_error:.4f}".rjust(6)
            lines.append(f"measured error:      {measured_error}")
            orbital_scale = f"{self.orbital_scale:.4f}".rjust(9)
            lines.append(f"orbital scale:    {orbital_scale}")
            transmission_change = f"{self.transmission_change:.4f}".rjust(6)
            lines.append(f"transmission change: {transmission_change}")
            
        repr_string = "\n".join(lines)
        return repr_string

 
    def __lt__(self, other: Gradient) -> bool:
        """
        This method is used to determine sorting of grid gradients, which is
        done by the position value
        
        Parameters
        ----------
        other : GridGradient
            Another GridGradient class instance to compare with.
        """
        return self.position < other.position

 
    def get_scaled_gradient(self):
        """
        This method is used to determine the measured gradient that has been
        scaled by the orbital scale and the transmission scale.
        
        Returns
        -------
        scaled_gradient : float
            This is the measured gradient that has been scaled by the orbital
            scale and the transmission scale.
        """
        if self.measured_gradient is None:
            return

        transmission_scale = 1 / self.transmission_change
        total_scale = self.orbital_scale * transmission_scale
        
        scaled_gradient = self.measured_gradient * total_scale

        if scaled_gradient > 1:
            raise ValueError("scaled gradient is greater than one, check the"
                " measured gradient, orbital scale and transmission change")

        return scaled_gradient

 
    def determine_mask(self, 
            measured_gradient: float,
            orbital_scale: float,
            transmission_change: float = None,
            measured_error: float = None
        ) -> None:
        """
        This method is used to determine the mask based on the appropriately
        scaled, measured gradient.

        Parameters
        ----------
        measured_gradient : float
            This value is the measured light curve gradient at this particular
            position.
        orbital_scale : float
            This value is used to scale the measured gradient by some scale
            factor that depends on the transverse velocity of the occulter and
            the limb darkening parameter of the star.
        transmission_change : float
            This value scales the measured gradient by the change in 
            transmission over the gradient. If unknown use `1`, if unsure then
            use an upper limit.
        measured_error : float
            This is the error on the measurement of the measured gradient 
            [default = None].
        """
        self.measured_gradient = validate.number(measured_gradient, 
            "measured_gradient", lower_bound=0)
        
        self.orbital_scale = validate.number(orbital_scale, 
            "orbital_scale", lower_bound=0)
        
        if transmission_change is None:
            transmission_change = 1
        self.transmission_change = validate.number(transmission_change, 
            "transmission_change", lower_bound=0, upper_bound=1)

        if measured_error is None:
            measured_error = 1
        self.measured_error = validate.number(measured_error, "measured_error",
            lower_bound=0.)

        scaled_gradient = self.get_scaled_gradient()
        mask = (scaled_gradient > self.data).astype(float)
        mask += np.isnan(self.data).astype(float)
        self.mask = mask.astype(bool)

 
    def save_gradient(self, directory: str) -> None:
        """
        This method saves all the information of this object to a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
        """
        # change the directory as there can be multiple gradients
        gradient_directory = f"{directory}/gradient_{self.position:.4f}"
        if not os.path.exists(gradient_directory):
            os.mkdir(gradient_directory)

        self.save(gradient_directory)

        # save additional gradient information
        np.save(f"{gradient_directory}/position", np.array([self.position]))

        if self.measured_gradient is not None:
            mask_values = np.array([self.measured_gradient, 
                self.orbital_scale, self.transmission_change, 
                self.measured_error])
            np.save(f"{gradient_directory}/mask_values", mask_values)
        
        self.parameters.save(directory)

    
    @classmethod
    def load(cls, directory: str) -> Gradient:
        """
        This method loads all the information of this object from a specified
        directory.
        
        Parameters
        ----------
        directory : str
            File path for the saved information.
            
        Returns
        -------
        gradient : Gradient
            This is the loaded object.
        """
        name = Name.GRADIENT
        unit = Unit.NONE

        try:
            filepath_data = (f"{directory}/{name.property_name}_"
                f"{unit.property_unit}.npy")
            data = np.load(f"{filepath_data}")
            
            parameters = Parameters.load(directory)
            position = np.load(f"{directory}/position.npy")[0]
            
            gradient = cls(data, parameters, position)
            
            filepath_mask_values = f"{directory}/mask_values.npy"
            if os.path.exists(filepath_mask_values):
                mask_values = np.load(filepath_mask_values)
                gradient.determine_mask(*mask_values)

        except Exception:
            type_string = f"{name.property_name} [{unit.property_unit}]"
            raise LoadError(type_string, directory)

        return gradient


class Viewer:
    """
    This class is used to view the grid properties and gradients either as a 
    cube or a single slice. Cubes can be viewed from any of the 3 axes and a
    list of coordinates can be provided to plot throughout the cube.
    """

 
    def __init__(self, 
            ax: plt.Axes, 
            axis: int, 
            grid_property: Property,
            masked: bool, 
            index: int = None, 
            frozen: bool = False, 
            coordinates: list[tuple[float, float, float]] = None
        ) -> None:
        """
        Initialiser for the cube viewer.

        Parameters
        ----------
        ax : plt.Axes
            Axis object where plotting will occur.
        axis : int
            Which axis to use as the scrolling axis.
        grid_property : Property
            Is used to obtain data, extent and contrast properties.
        masked : bool
            Use the masked data or the unmasked data.
        index : int
            Initial index to plot [default = None -> middle of cube].
        frozen : bool
            Determines whether the plot is scrollable [default = False].
        coordinates : list[tuple]
            List of 3-D coordinates to plot if visible in the given slice.
        """
        self.axis = validate.number(axis, "axis", check_integer=True, 
            lower_bound=0, upper_bound=2)
        
        ax = validate.class_object(ax, "ax", plt.Axes, "Axes")
        self.ax = ax
        
        grid_property = validate.class_object(grid_property, "grid_property", 
            Property, "Property")

        self.data = grid_property.get_data(masked)
        self.slice_values = grid_property.parameters.get_vectors()[axis]

        if index is None:
            index = len(self.slice_values) // 2
        self.index = validate.number(index, "index", check_integer=True, 
            lower_bound=0, upper_bound=len(self.slice_values) - 1)

        if coordinates is not None:
            coordinates = validate.class_object(coordinates, "coordinates",
                list, "List")
            for coordinate in coordinates:
                if len(coordinate) != 3:
                    raise ValueError("all input coordinates must be tuples "
                        "with three values")
        self.coordinates = coordinates
        self.tolerance = np.abs(self.slice_values[1] - self.slice_values[0]) / 3

        self.slice_name = ["$y$", "$x$", "$R_f$"][axis]
        
        
        property_name = f"{grid_property.name.property_name}"
        property_unit = f"{grid_property.unit.symbol}" 
        self.title_prefix = f"{property_name} [{property_unit}]"
            
        self.update_title()
        self.set_labels()
        cmap = plt.cm.get_cmap(grid_property.color_map, 
            grid_property.num_colors)
        extent = self.determine_extent(grid_property.parameters)

        # get data slice and set image
        data_slice = np.take(self.data, self.index, axis)
        self.image = self.ax.imshow(data_slice, origin="lower", cmap=cmap, 
            vmin=grid_property.vmin, vmax=grid_property.vmax, extent=extent)

        self.set_rf_ticklabels(grid_property.parameters)

        self.frozen = False
        self.update()
        if frozen is not None:
            self.frozen = validate.boolean(frozen, "frozen")

 
    def __str__(self) -> str:
        """
        This method produces a string representation of the grid property
        viewer for the user.
        
        Returns
        -------
        str_string : str
            Representation string of the grid property viewer.
        """
        return self.__repr__()

 
    def __repr__(self) -> str:
        """
        This method produces a string representation of the grid property
        viewer.
        
        Returns
        -------
        repr_string : str
            Representation string of the grid property viewer.
        """
        lines: list[str] = [""]
        lines.append("Grid Property Viewer")
        lines.append(28 * "-")
        lines.append(self.ax.get_title())

        return "\n".join(lines)

 
    def update_title(self) -> None:
        """
        This method generates the title of the given slice.
        """
        title = (f"{self.title_prefix} - {self.slice_name} = "
            f"{self.slice_values[self.index]:.4f}")
        
        if self.axis == 2: 
            minimum_radius_index = len(self.slice_values) // 2
            if self.index < minimum_radius_index:
                title = title + " - horizontal"
            elif self.index > minimum_radius_index:
                title = title + " - vertical"
        
        self.ax.set_title(title)

 
    def set_labels(self) -> None:
        """
        This method is used to generate the axes labels.
        """
        xlabel = ["$R_f$", "$R_f$", "$x$"][self.axis]
        ylabel = ["$x$", "$y$", "$y$"][self.axis]
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

 
    def determine_extent(self, 
            parameters: Parameters
        ) -> tuple[float, float, float, float]:
        """
        This method determines the extent for the plot (rf is manipulated 
        because of its non-linear nature).
        
        Parameters
        ----------
        grid_parameters : GridParameters
            Contains all the information for the grid that is used to extract
            data.

        Returns
        -------
        extent : tuple (4-D)
            The extent for the matplotlib imshow image.
        """
        dy, dx, rf_array = parameters.get_vectors()
        x_extent = (dx[0], dx[-1])
        y_extent = (dy[0], dy[-1])
        rf_extent = (0, 2 * (rf_array[-1] - 1))

        horizontal_extent = [rf_extent, rf_extent, x_extent][self.axis]
        vertical_extent = [x_extent, y_extent, y_extent][self.axis]

        extent = (*horizontal_extent, *vertical_extent)

        return extent

 
    def set_rf_ticklabels(self, parameters: Parameters) -> None:
        """
        This method sets the labels for the x and y axes. Note that this can
        not be done with extent due to the fact that Rf is not a linear axis
        (rf_max -> 1 -> rf_max)
        
        Parameters
        ----------
        grid_parameters : GridParameters
            Contains all the information for the grid that is used to extract
            data.
        """
        if self.axis == 2:
            return
        
        rf = parameters.rf
        rf_range = rf[-1] - rf[0]
        
        locations = self.ax.get_xticks()
        
        labels = 1 + np.abs(locations - rf_range)
        labels = np.char.mod("%.2f", labels)

        self.ax.set_xticks(locations)
        self.ax.set_xticklabels(labels)

 
    def onscroll(self, event: MouseEvent) -> None:
        """
        This method determines what happens what happens when the scroll wheel
        is used to navigate through the data cube.
        
        Parameters
        ----------
        event : MouseEvent
            This is used to register scroll events.
        """
        if self.frozen:
            return

        if event.button == "up" and self.index < len(self.slice_values) - 1:
            self.index += 1
        elif event.button != "up" and self.index > 0:
            self.index -= 1

        self.update()

 
    def update(self) -> None:
        """
        This method is used to update the slice of the data cube being viewed.
        """
        data_slice = np.take(self.data, self.index, self.axis)
        self.image.set_data(data_slice)
        self.plot_coordinates()
        
        self.update_title()

        self.image.axes.figure.canvas.draw()

 
    def plot_coordinates(self) -> None:
        """
        This method is used to plot a point at a specified location.
        """
        if self.coordinates is None:
            return

        plotted = False

        for coordinate in self.coordinates:
            distance = np.abs(self.slice_values[self.index] - 
                coordinate[self.axis])

            if self.tolerance > distance:
                y, x = np.delete(coordinate, self.axis)
                self.ax.plot(x, y, "ro")
                plotted = True
        
        if not plotted:
            self.ax.lines = []