import torch

from gravann.functions import binder as function_binder
from gravann.labels import binder as label_binder
from gravann.util import get_target_point_sampler


def compute_c_for_model(model, encoding, method, use_acc=True, **kwargs):
    """Convenience function to calculate the current c constant for a model (based on mascon or polyhedral mesh data)

    Args:
        model: trained model
        encoding (encoding): encoding to use for the points
        method (str): either 'mascon' or 'polyhedral'
        use_acc (bool): if acceleration should be used (otherwise potential)


    Keyword Args:
        sample (str): sample name (file name) instead of mascons or mesh data
        mascon_points (torch.tensor): asteroid mascon points
        mascon_masses (torch.tensor): asteroid mascon masses
        mascon_masses_nu (torch.tensor): asteroid mascon masses
        mesh_vertices ((N, 3) array-like): the vertices of a polyhedron
        mesh_faces ((N, 3) array-like): the triangular faces of a polyhedron
    """
    label_fn = label_binder.bind_label(
        method, use_acc, **kwargs
    )
    prediction_fn = function_binder.bind_integration(
        'trapezoid', use_acc, model, encoding, integration_points=10000, **kwargs
    )
    return _compute_c_for_model(label_fn, prediction_fn)


def _compute_c_for_model(label_fn, prediction_fn):
    """Common method computing the current c constant for a model.

    Args:
        label_fn: mascon or polyhedral label (only takes target points as argument)
        prediction_fn : label of the trained model (only takes target points as argument)
    """
    targets_point_sampler = get_target_point_sampler(1000, method="spherical", bounds=[0.81, 1.0])
    target_points = targets_point_sampler()
    labels = label_fn(target_points)
    predicted = prediction_fn(target_points)
    return (torch.sum(predicted * labels) / torch.sum(predicted * predicted)).item()
