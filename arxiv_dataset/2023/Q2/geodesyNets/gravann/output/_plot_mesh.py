from os import PathLike
from typing import Optional

import pyvista as pv
import tetgen

from gravann.input import sample_reader


def plot_polyhedral_meshes(save_path: PathLike, **kwargs):
    """Plots the mesh and saves it to a file.

    Args:
        save_path: Path to save the plot to.

    Keyword Args:
        vertices (numpy.ndarray): Vertices of the mesh.
        faces (numpy.ndarray): Faces of the mesh.
        sample (str): Name of the mesh to load. Alternative to vertices and faces.
        resolution (str): Resolution of the mesh. Default is "100%".
    """
    resolution = ["100%", "10%", "1%", "0.1%"]
    plotter = pv.Plotter(shape=(2, 2))
    pv.set_plot_theme("document")
    for i, res in enumerate(resolution):
        plot_polyhedral_mesh(
            jupyter_backend=None,
            subplot=(i // 2, i % 2),
            plotter=plotter,
            resolution=res,
            **kwargs
        )
    plotter.screenshot(save_path, window_size=(3200, 2000))


def plot_polyhedral_mesh(save_path: Optional[PathLike] = None, jupyter_backend: Optional[str] = None, **kwargs):
    """Plots the mesh and saves it optionally to a file.

    Args:
        save_path: Path to save the plot to. If None, the plot is not saved.
        jupyter_backend: Backend to use for plotting in Jupyter. If None, the plot is not shown. E.g. "panel".

    Keyword Args:
        vertices (numpy.ndarray): Vertices of the mesh.
        faces (numpy.ndarray): Faces of the mesh.
        sample (str): Name of the mesh to load. Alternative to vertices and faces.
        resolution (str): Resolution of the mesh. Default is "100%".
        subplot (int): Subplot to plot the mesh in.
    """
    if sample_name := kwargs.get("sample", None):
        vertices, faces = sample_reader.load_polyhedral_mesh(sample_name, kwargs.get("resolution", "100%"))
    elif (vertices := kwargs.get("vertices", None)) and (faces := kwargs.get("faces", None)):
        pass
    else:
        raise ValueError("Either sample or vertices and faces must be provided.")
    tgen = tetgen.TetGen(vertices, faces)
    tgen.tetrahedralize()

    plotter = kwargs.get("plotter", pv.Plotter())
    if subplot := kwargs.get("subplot", None):
        plotter.subplot(*subplot)

    plotter.add_mesh(tgen.grid, show_edges=True)
    plotter.add_axes(x_color='red', y_color='green', z_color='blue', xlabel='X', ylabel='Y', zlabel='Z', line_width=2)
    plotter.add_title(f'Resolution: {kwargs.get("resolution", "100%")}')
    plotter.show_grid()

    if jupyter_backend is not None:
        plotter.show(jupyter_backend=jupyter_backend)
    if save_path is not None:
        plotter.screenshot(save_path)
