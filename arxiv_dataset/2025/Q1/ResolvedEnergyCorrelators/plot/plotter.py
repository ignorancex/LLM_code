from os.path import dirname, abspath
from pathlib import Path
from copy import deepcopy

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PathCollection
from cycler import cycler

import logging
from warnings import warn


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Basic Directories
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
output_dir = Path(dirname(dirname(abspath(__file__)))) / "output"

property_data_dir = output_dir / "jet_properties"
property_figure_dir = output_dir / "jet_property_figures"

enc_data_dir = output_dir / "new_encs"
enc_figure_dir = output_dir / "new_enc_figures"

ewoc_data_dir = output_dir / "ewocs"
ewoc_figure_dir = output_dir / "ewoc_figures"

manim_dir = output_dir / "manim"
manim_enc_dir = manim_dir / "encs"


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Logging Flags
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Logging level (for LOGGER, defined below)
LOGGING_LEVEL = 'INFO'
# (drawn from:)
logging_levels = {'HYPER': 5,
                  'DEBUG': logging.DEBUG,  # 10
                  'INFO': logging.INFO,  # 20 'WARNING': logging.WARNING,  # 30
                  'ERROR': logging.ERROR  # 40
                  }
# Allowing 5, 15, 25, 35 for intermediate levels:
# (I have some especially verbose debug messages set to level 5)
logging_levels.update({5*(2*k+1): 5*(2*k+1) for k in range(4)})

# =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= #
# Logger Initialization
# =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+= #
# Main logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging_levels[LOGGING_LEVEL])
# Printing to stdout
LOGGER.addHandler(logging.StreamHandler())


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
# Basic Plotter Class
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#= #
def copy_attributes(obj2, obj1, attr_list):
    for i_attribute  in attr_list:
        getattr(obj2, 'set_' + i_attribute)\
            ( getattr(obj1, 'get_' + i_attribute)() )

class Plotter():
    """Base plotter class."""
    # ---------------------------------------------------
    # Formatting:
    # ---------------------------------------------------
    # Font sizes
    _BIG_size    = 16
    _Big_size    = 14
    _big_size    = 15
    _normal_size = 12
    _small_size  = 10

    # Plot styles: lines, markers, etc.
    _linewidth = 2
    _linestyle = '-'  # solid lines
    # (modstyle)
    # _linestyle = None
    _markersize = 2
    _capsizes = 2
    _capthick = 1.5

    # Figure size
    _fig_width = 6.4
    _fig_height = 4.8
    _figsize = (_fig_width, _fig_height)


    def __init__(self, **kwargs):
        """Initializes the plotter, including the axis information
        and style of the plots.

        Possible Parameters
        ----------

        Figure Parameters
        ----------
        xlabel : str
            xlabel of the plot.
        ylabel : str
            ylabel of the plot.
        title : str
            title of the plot.
        showdate : bool
            If True, adds a date to the upper right of the plot.
        xlim : tuple
            The x limits of the plot.
        ylim : tuple
            The y limits of the plot.
        ylim_ratio : tuple
            The y limits of the ratio subplot.
        ratio_plot : bool
            Determines whether there is an additional subplot
            for ratio plotting.
        ylabel_ratio : str
            ylabel of the ratio subplot, if it exists.

        Style Parameters
        ----------
        font.size : int
            default text sizes
        figure.titlesize : int
            fontsize of the figure title
        axes.titlesize : int
            fontsize of the axes title
        axes.labelsize : int
            fontsize of the x and y labels
        xtick.labelsize : int
            fontsize of the x tick labels
        ytick.labelsize : int
            fontsize of the y tick labels
        legend.fontsize : int
            fontsize of the legend
        lines.linewidth : int
            default plot linewidth
        axes.prop_cycle : cycler
            default cycle (e.g. for plot colors)
        """
        # Get plot metadata from kwargs:
        _metadata_defaults = {'figsize'      : self._figsize,
                              'title'        : None,
                              'xlabel'       : r'$x$',
                              'ylabel'       : r'$y$',
                              'xlim'         : None,
                              'ylim'         : None,
                              'x_scale'      : 'linear',
                              'y_scale'      : 'linear',
                              'ylim_ratio'   : None,
                              'ylabel_ratio' : 'Ratio',
                             }

        self.ratio_plot = kwargs.pop('ratio_plot', False)

        self.metadata = {key: kwargs.get(key, _metadata_defaults[key])
                         for key, default_value in
                         _metadata_defaults.items()}

        # Setting up custom color cycler
        prop_cycle = kwargs.get('axes.prop_cycle',
                                  'default')
        if prop_cycle == 'default':
            prop_cycle = cycler('color',
                                  ['cornflowerblue', 'lightcoral',
                                   'mediumorchid', 'mediumseagreen',
                                   'sandybrown'])
        elif prop_cycle == 'pastel':
            prop_cycle = cycler('color',
                                  ['#a4b6dd', '#d09292', '#c094cc',
                                   '#a2d0c0', '#c37892'])


        # Get plot style info for plotting with a local rc_context
        self._mpl_rc = {
            # Setting up for LaTeX-like text
            'text.usetex': kwargs.get('usetex', True),
            'font.family': kwargs.get('font.family', 'serif'),
            'font.serif': kwargs.get('font.serif',
                                     'Computer Modern'),
                                     # 'Computer Modern Roman'),
            'font.monospace': kwargs.get('font.monospace',
                                     'Computer Modern Typewriter'),
            'text.latex.preamble': kwargs.get('latex.preamble',
                                     r'\usepackage{amsmath}'),

            # Main font sizes
            'font.size': kwargs.get('font.size', self._normal_size),
            'figure.titlesize': kwargs.get('figure.titlesize',
                                           self._BIG_size),
            'axes.titlesize': kwargs.get('axes.titlesize',
                                         self._Big_size),
            'axes.labelsize': kwargs.get('axes.labelsize',
                                         self._normal_size),
            # Ticks
            'xtick.labelsize': kwargs.get('xtick.labelsize',
                                          self._small_size),
            'ytick.labelsize': kwargs.get('ytick.labelsize',
                                          self._small_size),
            # Legend options
            'legend.frameon': kwargs.get('legend.frameon', False),
            'legend.fontsize': kwargs.get('legend.fontsize',
                                           self._big_size),
            # Line options
            'lines.linewidth': kwargs.get('lines.linewidth',
                                          self._linewidth),
        }

        if prop_cycle is not None:
            self._mpl_rc['axes.prop_cycle'] = prop_cycle

        # Preparing plot behavior
        mpl.rcParams.update(self._mpl_rc)

        # Preparing figure and axes for the plotter:
        self.prepare_subplots()
        self.format_subplots()


    def combine(self, axes_list, artists=('lines')):
        """
        Combines multiple axes into the current plot,
        copying specified plot elements.

        Parameters
        ----------
        axes_list : list of matplotlib.axes.Axes
            List of axes objects whose plot elements will
            be added to the current plot.
        artists : tuple of str, optional
            Types of plot elements to copy. Options include
            'lines', 'patches', 'collections', etc.
            Default is ('lines',), which copies only
            Line2D objects (lines).
        """
        for source_ax in axes_list:
            # Copy lines if 'lines' is in the artists list
            if 'lines' in artists:
                for line in source_ax.get_lines():
                    # Extract line data (x, y)
                    x_data, y_data = line.get_data()
                    self.axes[0].plot(x_data, y_data,
                                      label=line.get_label(),
                                      color=line.get_color(),
                                      linestyle=line.get_linestyle(),
                                      linewidth=line.get_linewidth(),
                                      marker=line.get_marker())

            # Copy patches if 'patches' is in the artists list
            # (e.g., bars, polygons)
            if 'patches' in artists:
                for patch in source_ax.patches:
                    self.axes[0].add_patch(deepcopy(patch))

            # if 'collections' is in the artists list
            # (e.g., scatter points)
            if 'collections' in artists:
                for collection in source_ax.collections:
                    if isinstance(collection, PathCollection):
                        # This is likely a scatter plot
                        offsets = collection.get_offsets()
                        self.axes[0].scatter(
                            offsets[:, 0], offsets[:, 1],
                            label=collection.get_label(),
                            color=collection.get_facecolor()[0],
                            s=collection.get_sizes())
                    else:
                        # For other collections, add them directly
                        warn("Adding an explicit copy of "
                             f"{collection} to the new plot -- "
                             "this is not supported by matplotlib "
                             "and may lead to unexpected results.")
                        collection = deepcopy(collection)
                        collection._axes = self.axes[0]
                        collection.figure = None
                        self.axes[0].add_collection(collection)

            # Update axis limits
            xlim = self.axes[0].get_xlim()
            ylim = self.axes[0].get_ylim()
            source_xlim = source_ax.get_xlim()
            source_ylim = source_ax.get_ylim()
            self.axes[0].set_xlim(min(xlim[0], source_xlim[0]), max(xlim[1], source_xlim[1]))
            self.axes[0].set_ylim(min(ylim[0], source_ylim[0]), max(ylim[1], source_ylim[1]))

        self.fig.canvas.draw()


    def prepare_subplots(self, **kwargs):
        """Creates a figure and associated axes using default or
        given parameters during initialization.

        Can be used to produce a figure with a ratio subplot.

        New Parameters
        ----------
        showdate : bool
            If True, adds a date to the upper right of the plot.

        Returns
        -------
        Figure, axes.Axes
            The figure and axes/subplots specified by the
            above parameters.
        """
        # Setting up plot options
        metadata = self.metadata.copy()
        metadata.update(kwargs)
        kwargs = metadata
        del metadata

        # Get plt subplots
        gridspec_kw = {'height_ratios': (3.5, 1) if self.ratio_plot
                                                 else (1,),
                       'hspace': 0.0}
        nsubplots = 2 if self.ratio_plot else 1
        fig, axes = plt.subplots(nsubplots, gridspec_kw=gridspec_kw,
                                 figsize=kwargs.get('figsize',
                                            self.metadata['figsize']))
        if nsubplots == 1:
            axes = [axes]

        self.fig, self.axes = fig, axes


    def format_subplots(self, **kwargs):
        """Formats the figure and associated axes using default or
        given parameters during initialization.

        New Parameters
        ----------
        showdate : bool
            If True, adds a date to the upper right of the plot.

        Returns
        -------
        Figure, axes.Axes
            The figure and axes/subplots specified by the
            above parameters.
        """
        # Setting up plot options
        metadata = self.metadata.copy()
        metadata.update(kwargs)
        kwargs = metadata
        del metadata

        fig, axes = self.fig, self.axes

        # - - - - - - - - - - - - - - - -
        # Axis Formatting
        # - - - - - - - - - - - - - - - -
        # Axis limits
        # xlim
        xlim = kwargs.get('xlim', self.metadata['xlim'])
        if hasattr(xlim, '__iter__') and \
                not isinstance(xlim, str):
            xlim = [None if isinstance(lim, str) else lim
                    for lim in xlim]
            if xlim == [None, None]:
                xlim = [None]
        else:
            xlim = [xlim]

        if xlim != [None]:
            _ = [ax.set_xlim(*xlim) for ax in axes]

        # ylim (axes[0])
        ylim = kwargs.get('ylim', self.metadata['ylim'])
        if hasattr(ylim, '__iter__') and \
                not isinstance(ylim, str):
            ylim = [None if isinstance(lim, str) else lim
                    for lim in ylim]
            if ylim == [None, None]:
                ylim = [None]
        else:
            ylim = [ylim]

        if ylim is not None:
            axes[0].set_ylim(*ylim)

        # ylim_ratio (axes[1], if it exists)
        if self.ratio_plot:
            if kwargs.get('ylim_ratio', self.metadata['ylim_ratio']) \
                    is not None:
                try:
                    axes[1].set_ylim(*kwargs.get('ylim_ratio',
                                                 self.metadata['ylim_ratio'])
                                     )
                except:
                    axes[1].set_ylim('auto')
            axes[1].set_yscale('log')

        # Axes labels
        axes[-1].set_xlabel(kwargs.get('xlabel', self.metadata['xlabel']))
        axes[0].set_ylabel(kwargs.get('ylabel', self.metadata['ylabel']),
                           labelpad=5)
        if self.ratio_plot:
            axes[1].set_ylabel(kwargs.get('ylabel_ratio',
                                          self.metadata['ylabel_ratio']),
                               labelpad=-10)

        # Tick settings
        if kwargs.get('axes.minorticks_on', True):
            for ax_instance in axes:
                ax_instance.minorticks_on()
                ax_instance.tick_params(top=True, right=True, bottom=True,
                                        left=True, direction='in',
                                        which='both')

        if self.ratio_plot:
            axes[0].tick_params(labelbottom=False)
            axes[1].tick_params(axis='y')

        # Extra plot information
        pad = .01

        if kwargs.get('x_scale', self.metadata['x_scale']) == 'log':
            [ax.set_xscale('log') for ax in axes]
        if kwargs.get('y_scale', self.metadata['y_scale']) == 'log':
            [ax.set_yscale('log') for ax in axes]

        # - - - - - - - - - - - - - - - -
        # Additional Formatting
        # - - - - - - - - - - - - - - - -
        if kwargs.get('title', self.metadata['title']) is not None:
            # Main title
            axes[0].text(
                x=.12,
                y=1.005+pad,
                s=kwargs.get('title', self.metadata['title']),
                transform=axes[0].transAxes,
                ha="left",
                va="bottom",
                fontsize=self._small_size * 1.5,
                fontstyle="italic",
                fontname="Arial"
            )

        plt.tight_layout()

        self.fig = fig
        self.axes = axes
        return fig, axes


    def stamp(self, left_x, top_y,
              axes_index=0, delta_y=0.075,
              ha='left', va='center',
              fontsize=8.5, transform='default',
              text_options=None, **text):
        # Stamp formatting
        if transform == 'default':
            transform = self.axes[axes_index].transAxes
        if not text_options:
            text_options = {}
        text_options.update({'horizontalalignment': ha,
                            'verticalalignment': va,
                            'fontsize': fontsize,
                            'transform': transform})

        # Stamp text (add text line-by-line)
        for line in range(len(text)):
            y_val = top_y - line*delta_y
            if text.get(f'line_{line}') is not None:
                self.axes[axes_index].text(left_x, y_val,
                                           text.get(f'line_{line}'),
                                           **text_options)


    def show(self):
        self.fig.show()

    def savefig(self, filename, dirname=''):
        if dirname:
            dirname = str(dirname) + '/'
        self.fig.savefig(str(dirname)+str(filename))



class PolarPlotter(Plotter):
    def prepare_subplots(self, **kwargs):
        """Creates a figure and associated axes using default or
        given parameters during initialization.

        Can be used to produce a figure with a ratio subplot.

        New Parameters
        ----------
        showdate : bool
            If True, adds a date to the upper right of the plot.

        Returns
        -------
        Figure, axes.Axes
            The figure and axes/subplots specified by the
            above parameters.
        """
        # Setting up plot options
        self.metadata['xlabel'] = None
        self.metadata['ylabel'] = None
        self.metadata['axes.minorticks_on'] = False
        metadata = self.metadata.copy()
        metadata.update(kwargs)
        kwargs = metadata
        del metadata

        # Get plt subplots
        gridspec_kw = {'height_ratios': (3.5, 1) if self.ratio_plot
                                                 else (1,),
                       'hspace': 0.0}
        nsubplots = 2 if self.ratio_plot else 1
        fig, axes = plt.subplots(nsubplots, gridspec_kw=gridspec_kw,
                                 subplot_kw=dict(projection="polar"),
                                 figsize=kwargs.get('figsize',
                                            self.metadata['figsize']))
        if nsubplots == 1:
            axes = [axes]

        self.fig, self.axes = fig, axes

# =============================================


def combine_axes(axes_list, artists=('lines',), **kwargs):
    combined_plotter = Plotter(**kwargs)
    combined_plotter.combine(axes_list, artists)
    return combined_plotter


def combine_plotters(plotter_list, ax_ind=0,
                     style_from=0, **kwargs):
    if style_from is not None:
        cplotter = plotter_list[style_from]
        cplotter.combine([p.axes[0] for i, p in
                          enumerate(plotter_list)
                          if i!=style_from])
    else:
        cplotter = combine_axes([p.axes[ax_ind]
                                 for p in plotter_list],
                                **kwargs)

    return cplotter


# =============================================
# Main (rudimentary test)
# =============================================
if __name__ == "__main__":
    plotter = Plotter()
    plotter.show()
