import os
import numpy as np
import importlib.util

import warnings

from matplotlib.colors import Normalize, LogNorm
from plotter import Plotter, PolarPlotter


def save_histogram_to_file(hist_data, file_name):
    """
    Saves the histogram data to a file in a format compatible with HistogramData.load().

    Parameters:
    - hist_data: HistogramData object containing histogram, edges, centers, and metadata.
    - file_name: str, the name of the file to save the histogram data to.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'w') as f:
        # Write the import statement
        f.write("import numpy as np\n\n")

        # Write metadata as comments
        f.write("# ==================================\n")
        f.write("# Metadata\n")
        f.write("# ==================================\n")
        for key, value in hist_data.metadata.items():
            if isinstance(value, str):
                f.write(f"{key} = \"{value}\"\n")
            else:
                f.write(f"{key} = {value}\n")
        f.write("\n")

        # Write the histogram data
        f.write("# ==================================\n")
        f.write("# Histogram Data\n")
        f.write("# ==================================\n")

        # Write the histogram array
        f.write("hist = np.array([\n")
        # Format the histogram values
        hist_values = ", ".join(map(str, hist_data.hist.flatten()))
        f.write(f"    {hist_values}\n")
        f.write("])\n\n")

        # Write edges and centers
        var_list = hist_data.variable_order
        if not var_list:
            var_list = hist_data.edges.keys()

        for var_name in var_list:
            # Edges
            edges_var_name = f"{var_name}_edges"
            edges_values = ", ".join(map(str, hist_data.edges[var_name]))
            f.write(f"{edges_var_name} = np.array([\n")
            f.write(f"    {edges_values}\n")
            f.write("])\n\n")

            # Centers
            centers_var_name = f"{var_name}_centers"
            centers_values = ", ".join(map(str, hist_data.centers[var_name]))
            f.write(f"{centers_var_name} = np.array([\n")
            f.write(f"    {centers_values}\n")
            f.write("])\n\n")

    print(f"Histogram data saved to '{file_name}' successfully.")


def check_normalization(hist, scheme='linear'):
    """Checks for histogram normalization."""
    hist.integrate_histogram(scheme=scheme, outflow_weight=1)

    if not np.isclose(hist.integral, 1.0, rtol=5e-2):
        warnings.warn(f"Histogram does not integrate to 1 "
                      f"({hist.integral=})")
        return False




# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
# Base Histogram Class
# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
class HistogramData:
    def __init__(self, file_name=None, hist_data=None,
                 variable_order : list = None,
                 validate=False,
                 **kwargs):
        """
        Initializes the HistogramData class.
        If a file name is provided, the 'load' method is called to
        load data from the file.
        """
        # Initializing
        self.hist = None
        self.edges = {}
        self.centers = {}
        self.metadata = {}

        # Setting up the variable names corresponding to
        # each dimension of the histogram
        self.variable_order = variable_order

        # Loading from file
        if file_name:
            if hist_data:
                raise ValueError("Cannot be given both a filename "
                                 "and additional histogram data"
                                 "for histogram initialization.")
            if kwargs:
                raise ValueError("Cannot be given both a filename "
                                 "and additional keyword arguments "
                                 "for histogram initialization.")
            self.load(file_name)

        elif hist_data is not None:
            if kwargs:
                raise ValueError("Cannot be given both histogram "
                                 "data and additional keyword args "
                                 "for histogram initialization "
                                 f"(given {kwargs=}).")
            self.hist = hist_data.hist
            self.edges = hist_data.edges
            self.centers = hist_data.centers
            self.metadata = hist_data.metadata

            if variable_order is not None \
                    and hist_data.variable_order is not None:
                if variable_order != hist_data.variable_order:
                    raise ValueError(f"Given {variable_order=} "
                                     "during initialization, but "
                                     "the given HistogramData the "
                                     "object has variable order "
                                     f"{hist_data.variable_order}.")

            self.variable_order = hist_data.variable_order or \
                                    variable_order

        elif kwargs:
            try:
                self.hist = kwargs['hist']
                self.edges = kwargs['edges']
                self.centers = kwargs['centers']
                self.metadata = kwargs['metadata']
            except Exception as exc:
                raise ValueError("Must be given all data for "
                                 "histogram initialization. "
                                 "Cannot initialize with only "
                                 "partial histogram information "
                                 f"(given {kwargs=}).") from exc

        if self.variable_order:
            if set(self.variable_order) != set(self.edges.keys()):
                raise ValueError(f"Invalid {variable_order=} does "
                                 f"not match {self.edges.keys()=}.")
        if validate:
            # After loading, validate the data
            self.validate()


    def load(self, file_name):
        """
        Loads the data from the given Python file.
        Identifies 'hist', bin edges, and bin centers,
        while storing other attributes separately.
        """
        # Load the module from the given file
        spec = importlib.util.spec_from_file_location("data_file", file_name)
        data_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_file)

        # Process attributes in the loaded data file
        for attr_name in dir(data_file):
            # Skip private attributes (starting with '__')
            # and imported modules like numpy
            attr_value = getattr(data_file, attr_name)
            if attr_name.startswith('__') \
                    or isinstance(attr_value, type(np)):
                continue

            attr_value = getattr(data_file, attr_name)

            # Identify 'hist', 'edges', 'centers', and other attributes
            if attr_name == 'hist':
                self.hist = np.array(attr_value)
            elif attr_name.endswith('_edges'):
                self.edges[attr_name[:-6]] = np.array(attr_value)
            elif attr_name.endswith('edges'):
                self.edges[attr_name[:-5]] = np.array(attr_value)
            elif attr_name.endswith('_centers'):
                self.centers[attr_name[:-8]] = np.array(attr_value)
            elif attr_name.endswith('centers'):
                self.centers[attr_name[:-7]] = np.array(attr_value)
            else:
                self.metadata[attr_name] = attr_value

        if self.hist is None or not self.edges or not self.centers:
            raise ValueError("\nNot enough info to make histogram:"
                             f"\n\t* metadata = {self.metadata}"
                             f"\n\t* hist     = {self.hist}"
                             f"\n\t* edges    = {self.edges}"
                             f"\n\t* centers  = {self.centers}"
                             f"\n\n\t* {file_name=}"
                             )


    def validate(self):
        """
        Validates that the bin edges match the shape of
        the histogram and that there are bin centers
        for each set of bin edges.
        """
        if self.hist is None:
            raise ValueError("No histogram data (self.hist) found!")

        hist_shape = self.hist.shape
        num_dims = len(hist_shape)

        # Ensure the number of edges matches the number
        # of histogram dimensions
        if len(self.edges) != num_dims:
            raise ValueError("Mismatch between number of "
                             f"histogram dimensions ({num_dims}) "
                             "and number of bin edges "
                             f"({len(self.edges)}).")

        # Validate that each set of bin edges has the
        # correct length and there are corresponding centers
        for i, (dim_name, edges) in enumerate(self.edges.items()):
            if len(edges) != hist_shape[i] + 1:
                raise ValueError("Length of bin edges for "
                                 f"'{dim_name}' does not match the "
                                 f"histogram shape along dimension "
                                 f"{i}. Expected {hist_shape[i] + 1}, "
                                 f"got {len(edges)}.")

            if dim_name not in self.centers:
                raise ValueError("No bin centers found for "
                                 f"'{dim_name}'.")

        # If we have a single dimension, then check/define
        # the variable order
        if self.variable_order:
            if set(self.variable_order) != set(self.edges.keys()):
                raise ValueError("Ordering of the variables defining "
                                 "the histogram dimensions is "
                                 "inconsistent with the names "
                                 "of the given edges.")

        elif num_dims == 1:
            self.variable_order = list(self.edges.keys())


    def integrate_histogram(self, scheme='linear',
                            outflow_weight=None, func=None):
        """
        Integrates the histogram over all bin edges, with an
        optional function applied to the bin variables.

        Parameters:
        - scheme (str):
            The scheme for bin width calculation ('linear', 'log', or 'log10').
        - outflow_weight (float or None):
            Weight for outflow bins (if applicable).
        - func (callable or None):
            Function to apply to the bin variables during integration.

        Returns:
        - float: The integrated value of the histogram.
        """
        # Check if histogram is available
        if self.hist is None:
            raise ValueError("No histogram data available "
                             "for integration.")

        # Setting up scheme for integration (i.e. linear/logarithmic)
        if isinstance(scheme, str):
            scheme = {var: scheme for var in self.edges}

        # Initialize the total integration result
        total = self.hist.copy()
        # and bin information, for integrating against a function
        bin_centers_list = []
        is_outflow_bin_list = []

        # Loop through all the dimensions (variables) of the histogram
        for index, var in enumerate(self.variable_order):
            # Calculate bin widths for this dimension
            try:
                if scheme[var] == 'linear':
                    bin_widths = np.diff(self.edges[var])
                elif scheme[var] == 'log':
                    bin_widths = np.diff(np.log(self.edges[var]))
                elif scheme[var] == 'log10':
                    bin_widths = np.diff(np.log10(self.edges[var]))
                else:
                    raise ValueError("Invalid integration scheme "
                                     f"{scheme[var]} for {var}.")
            except KeyError as exc:
                raise ValueError(f"Given {scheme=} does not "
                                 "specify and integration scheme "
                                 "for all expected variables "
                                 f"{self.variable_order=}.") from exc

            # Detect outflow bins (where bin width would be NaN or inf)
            is_outflow_bin = np.isnan(bin_widths) | np.isinf(bin_widths)
            is_outflow_bin_list.append(is_outflow_bin)
            # for later integration against func if desired

            # If using special weights for outflow
            if outflow_weight is not None:
                bin_widths = np.nan_to_num(bin_widths,
                                           nan=outflow_weight,
                                           posinf=outflow_weight)

            # Use broadcasting to multiply the histogram values
            # along this dimension by the corresponding bin widths:
            shape = np.ones(len(self.hist.shape), dtype=int)
            shape[index] = len(bin_widths)
            bin_widths = bin_widths.reshape(shape)

            # Multiply the histogram by the bin widths
            # along the current axis
            total *= bin_widths

            # Prepare to integrate a function of the bin centers
            bin_centers_list.append(self.centers[var].reshape(shape))

        # Stack all bin centers together as a grid for the function application
        # TODO: DEBUG
        if func is not None:
            # Generate a grid of bin centers across all dimensions
            bin_centers_grid = np.stack(np.meshgrid(*bin_centers_list, indexing='ij'), axis=-1)

            # Generate a mask for all outflow bins across all dimensions
            outflow_mask = np.zeros(self.hist.shape, dtype=bool)
            for i, is_outflow_bin in enumerate(is_outflow_bin_list):
                # Reshape the outflow bin mask to match the dimension's axis
                shape = np.ones(len(self.hist.shape), dtype=int)
                shape[i] = len(is_outflow_bin)
                is_outflow_bin_reshaped = is_outflow_bin.reshape(shape)
                outflow_mask |= is_outflow_bin_reshaped

            # Apply the function only to non-outflow bins
            non_outflow_values = func(bin_centers_grid)
            total *= np.where(outflow_mask, outflow_weight, non_outflow_values)

        # Finally, sum up all the elements in the weighted histogram
        self.integral = np.nansum(total)
        return np.nansum(total)


    def integrate_over_variable(self, var_name, scheme='linear',
                                outflow_weight=None, func=None):
        """
        Integrates the histogram over a single variable (dimension),
        reducing the histogram to one lower dimension.
        An optional function can be applied to the bin variables
        during integration.

        Parameters:
        - var_name (str):
            The name of the variable to integrate over.
        - scheme (str):
            The scheme for bin width calculation
            ('linear', 'log', or 'log10').
        - outflow_weight (float or None):
            Weight for outflow bins (if applicable).
        - func (callable or None):
            A function to apply to the bin variables
            during integration.
            The function should take the bin centers as input
            and return a weight.

        Returns:
        - HistogramData: A new HistogramData object with one fewer dimension.

        TODO:
            Add integration bounds
        """
        # Check if the variable is valid
        if var_name not in self.edges or \
                var_name not in self.centers:
            raise ValueError(f"Variable '{var_name}' is not "
                             "present in edges or centers.")

        # Prepare a copy of the histogram for integration
        total_hist = self.hist.copy()

        # Get the bin edges for the variable to integrate over
        edges = self.edges[var_name]
        centers = self.centers[var_name]
        var_index = list(self.variable_order).index(var_name)

        # Calculate bin widths for this dimension
        if scheme == 'linear':
            bin_widths = np.diff(edges)
        elif scheme == 'log':
            bin_widths = np.diff(np.log(edges))
        elif scheme == 'log10':
            bin_widths = np.diff(np.log10(edges))
        else:
            raise ValueError("Invalid integration scheme "
                             f"'{scheme}'.")

        # Handle outflow bins
        if outflow_weight is not None:
            bin_widths = np.nan_to_num(bin_widths,
                                       nan=outflow_weight,
                                       posinf=outflow_weight)

        # Use broadcasting to apply the bin widths to the histogram
        shape = np.ones(len(self.hist.shape), dtype=int)
        shape[var_index] = len(bin_widths)
        bin_widths = bin_widths.reshape(shape)

        # Multiply the histogram by the bin widths along the axis
        total_hist *= bin_widths

        # Sum over the axis corresponding to the integrated variable
        integrated_hist = np.sum(total_hist, axis=var_index)

        # Remove the integrated variable from the edges and centers
        new_edges = {k: v for k, v in self.edges.items()
                          if k != var_name}
        new_centers = {k: v for k, v in self.centers.items()
                            if k != var_name}

        # Update the variable order
        new_variable_order = self.variable_order.copy()
        new_variable_order.remove(var_name)

        # Create a HistogramData object with the reduced histogram
        return HistogramData(hist=integrated_hist, edges=new_edges,
                             centers=new_centers,
                             metadata=self.metadata,
                             variable_order=new_variable_order)


    def get_sub_histogram(self, var_name, value):
        """
        Extracts a sub-histogram by fixing the value of the
        given variable.

        Parameters:
        - var_name:
            The name of the variable to fix
            (must be in edges and centers)
        - value:
            The value of the variable to use for
            sub-histogram extraction

        Returns:
        A HistogramData object containing the sub-histogram.
        """
        if var_name not in self.edges or var_name not in self.centers:
            raise ValueError(f"Variable '{var_name}' is not "
                             "present in edges or centers.")

        # Find the bin corresponding to the given value
        edges = self.edges[var_name]
        if not self.variable_order:
            raise ValueError("Must give a variable order for the "
                             "row, column, etc. of the histogram.")

        # Determine the bin index for the given value
        bin_index = np.digitize([value], edges)[0] - 1

        if bin_index < 0 or bin_index >= len(edges) - 1:
            raise ValueError(f"Value {value} is out of range "
                             f"for variable '{var_name}'.")

        # Prepare new histogram and bin edges
        new_hist = np.take(self.hist, bin_index,
                       axis=list(self.variable_order).\
                                 index(var_name))

        new_edges = {k: v for k, v in self.edges.items()
                            if k != var_name}
        new_centers = {k: v for k, v in self.centers.items()
                             if k != var_name}

        # Updating metadata
        new_metadata = self.metadata.copy()
        sub_histogram_vars = new_metadata.get('sub_histogram_vars',
                                              [])
        sub_histogram_vars = [*sub_histogram_vars, var_name]
        sub_histogram_vals = new_metadata.get('sub_histogram_vals',
                                              [])
        sub_histogram_vals = [*sub_histogram_vals, var_name]

        new_metadata.update({'sub_histogram_vars':
                                sub_histogram_vars,
                             'sub_histogram_vals':
                                sub_histogram_vals})

        # Updating order in which variables appear
        new_var_order = self.variable_order.copy()
        if new_var_order is not None:
            new_var_order.remove(var_name)

        return HistogramData(hist=new_hist, edges=new_edges,
                             centers=new_centers,
                             metadata=new_metadata,
                             variable_order=new_var_order)


    def make_plot(self):
        """
        Raises NotImplementedError to ensure subclasses
        implement specific plot functionality.
        """
        raise NotImplementedError("Plot function not implemented "
                                  "for the base class.")


    def __str__(self):
        """
        Provides a string representation of the object,
        including the histogram, its dimensions,
        edges, centers, and other attributes.
        """
        repr_str = f"{self.__class__.__name__}:\n"
        if self.hist is not None:
            repr_str += f"  Histogram (hist): Available\n"
            repr_str += f"  Histogram dimensions: {self.hist.shape}\n"
        else:
            repr_str += "  Histogram (hist): None\n"

        repr_str += f"  Bin edges: {list(self.edges.keys())}\n"
        repr_str += f"  Bin centers: {list(self.centers.keys())}\n"
        repr_str += f"  Other attributes: {list(self.metadata.keys())}"
        repr_str += "\n"
        return repr_str


# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
# 1D Histogram Class
# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
class HistogramData1D(HistogramData):
    """A HistogramData subclass with 1D hists and plotting."""
    def validate(self):
        """
        Validates that the bin edges match the shape of the histogram and that
        there are bin centers for each set of bin edges.
        """
        if self.hist is None:
            raise ValueError("No histogram data (self.hist) found!")

        hist_shape = self.hist.shape
        num_dims = len(hist_shape)

        if num_dims != 1:
            raise ValueError("Histogram should be one-dimensional, "
                             f"but is {num_dims}-dimensional.")
        super().validate()


    def cumulative(self, scheme='linear', outflow_weight=None,
                   **kwargs):
        """Returns a histogram associated with the cumulative
        distribution of self, whose values are the cumulative sum
        of self.histogram multiplied by the bin widths.
        """
        # Getting attributes
        hist = self.hist
        edges = list(self.edges.values())[0]
        centers = list(self.centers.values())[0]

        # Finding bin widths
        if scheme == 'linear':
            bin_widths = np.diff(edges)
        elif scheme == 'log':
            bin_widths = np.diff(np.log(edges))
        elif scheme == 'log10':
            bin_widths = np.diff(np.log10(edges))
        else:
            raise ValueError("Invalid integration scheme "
                             f"'{scheme}'.")

        # If using special weights for outflow
        if outflow_weight is not None:
            bin_widths = np.nan_to_num(bin_widths,
                                       nan=outflow_weight,
                                       posinf=outflow_weight)

        # Getting cumulative
        cml_hist = np.cumsum(hist * bin_widths)

        # Packaging and returning
        variable_order = self.variable_order
        if variable_order:
            if len(variable_order) > 1:
                raise RuntimeError("1D hist was given several "
                                   f"variables: {variable_order}.")
            var = self.variable_order[0]
        else:
            var = self.edges.keys()[0]

        edges = {var: edges}
        centers = {var: centers}

        cml_data = ({'hist': cml_hist,
                     'edges': edges,
                     'centers': centers,
                     'variable_order': self.variable_order,
                     'metadata': dict(**{'cumulative': True},
                                      **self.metadata)})

        return HistogramData1D(**cml_data)


    def make_plot(self, **kwargs):
        """
        Plots the one-dimensional histogram using the Plotter class.
        """
        bin_centers = list(self.centers.values())[0]
        line_keys = {'color', 'ls', 'lw', 'label',
                     'linestyle', 'linewidth', 'marker'}
        line_kwargs = {key: kwargs.pop(key) for key in line_keys
                                            if key in kwargs}
        self.plot = Plotter(**kwargs)
        self.plot.axes[0].plot(bin_centers, self.hist,
                               **line_kwargs)


# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
# 2D Histogram Class
# #:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#:#
class HistogramData2D(HistogramData):
    """A HistogramData subclass with 2D hists and plotting."""
    def make_plot(self, plot_type, **kwargs):
        """
        Plots the one-dimensional histogram using the Plotter class.
        """
        if plot_type == 'density':
            self.make_density_plot(**kwargs)
            return
        if plot_type == 'bullseye':
            self.make_bullseye_plot(**kwargs)
            return

        raise NotImplementedError(f"Invalid {plot_type=}, must be "
                                  "`density` or `bullseye`.")


    def make_density_plot(self, log_norm=False, **kwargs):
        """
        Plots the 2D histogram as a density plot.

        Parameters:
        - log_norm:
            Whether to use logarithmic color normalization
        - **kwargs:
            Additional arguments for the Plotter class
        """
        # Unpack additional colorbar kwargs
        cbar_kwargs = {}
        for key in list(kwargs.keys()):
            if key.startswith('cbar_'):
                cbar_kwargs[key[5:]] = kwargs.pop(key)

        # Extract histogram and bin edges
        hist2d = self.hist
        edges = list(self.edges.values())
        if len(edges) < 2:
            raise ValueError("At least two sets of edges are required for 2D plotting.")

        # Extract bin edges
        var_names = self.variable_order
        if len(var_names) != 2:
            raise ValueError("For 2D density plot, exactly two variables must be specified.")

        var1, var2 = var_names
        edges1 = self.edges.get(var1)
        edges2 = self.edges.get(var2)

        # Prepare data for plotting
        X, Y = np.meshgrid(edges1, edges2)

        # Apply normalization
        if log_norm:
            masked_hist = np.where(hist2d == 0, np.nan, hist2d)
            vmin = kwargs.pop('vmin', np.nanmin(masked_hist))
            vmax = kwargs.pop('vmax', np.nanmax(hist2d))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = kwargs.pop('vmin', np.nanmin(hist2d))
            vmax = kwargs.pop('vmax', np.nanmax(hist2d))
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Create plot using Plotter
        self.density = Plotter(**kwargs)

        cmap = cbar_kwargs.pop('cmap', 'magma_r')
        pc = self.density.axes[0].pcolormesh(X, Y, hist2d.T,
                                          cmap=cmap, norm=norm,
                                          alpha=1.0,
                                          rasterized=True,
                                          antialiased=True)
        pc.set_edgecolor('face')

        # Add colorbar
        cbar = self.density.fig.colorbar(pc,
                               pad=cbar_kwargs.pop('pad', 0.1))
        if cbar_kwargs:
            cbar.set_label(**cbar_kwargs)


    def make_bullseye_plot(self, radial_coordinate,
                           log_norm=False, **kwargs):
        """
        Plots the 2D histogram as a bullseye plot in polar coordinates.

        Parameters:
        - radial_coordinate:
            The param to use as a radial coordinate in the bullseye
        - log_norm:
            Whether to use logarithmic color normalization.
        - **kwargs:
            Additional arguments for the Plotter class.
        """
        # Unpack additional colorbar kwargs
        cbar_kwargs = {}
        for key in list(kwargs.keys()):
            if key.startswith('cbar_'):
                cbar_kwargs[key[5:]] = kwargs.pop(key)

        # Extract histogram and bin edges
        hist2d = self.hist
        edges = list(self.edges.values())

        if len(edges) != 2:
            raise ValueError("For a bullseye plot, exactly two "
                             "sets of edges are required.")

        var1, var2 = self.variable_order
        if radial_coordinate == var1:
            radius  = self.centers.get(var1)
            phi     = self.centers.get(var2)
        elif radial_coordinate == var2:
            radius  = self.centers.get(var2)
            phi     = self.centers.get(var1)
        else:
            raise ValueError("Radial coordinate was given as "
                             f"{radial_coordinate}, but must be "
                             f"one of {(var1, var2)}.")

        # Apply normalization
        if log_norm:
            vmin = np.nanmin(hist2d[np.nonzero(hist2d)])
            vmin = kwargs.pop('vmin', vmin)
            vmax = kwargs.pop('vmax', np.nanmax(hist2d))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = kwargs.pop('vmin', np.nanmin(hist2d))
            vmax = kwargs.pop('vmax', np.nanmax(hist2d))
            norm = Normalize(vmin=hist2d.min(), vmax=hist2d.max())

        # Create plot using Plotter
        self.bullseye = PolarPlotter(**kwargs)
        fig = self.bullseye.fig
        ax = self.bullseye.axes[0]  # Polar subplot

        # Plot the data
        cmap = cbar_kwargs.pop('cmap', 'magma_r')
        ax.grid(False)
        pc = ax.pcolormesh(phi, radius, hist2d,
                           cmap=cmap, norm=norm,
                           alpha=1.0,
                           rasterized=True,
                           antialiased=True)
        pc.set_edgecolor('face')

        # Remove grid lines for clarity
        ax.set_thetagrids([])
        ax.set_rgrids([])

        if kwargs.pop('show_colorbar', True):
            cbar = fig.colorbar(pc, pad=cbar_kwargs.pop('pad', 0.1))
            if cbar_kwargs:
                cbar.set_label(**cbar_kwargs)
        #     # Set polar grid lines
        #     ax.set_thetagrids([0, 90, 180, 270],
        #         labels=[r'$\quad\phi=0$',r'$\phi=\pi/2$',
        #                 r'$\phi=\pi\quad$',r'$\phi=-\pi/2$'])
        # else:


        return
