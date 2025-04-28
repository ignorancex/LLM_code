from os import PathLike
from typing import List

from gravann.input import model_reader
from gravann.labels import binder as label_binder
from gravann.util import get_target_point_sampler, deep_get
from . import binder as function_binder
from ..network import encodings


def error_calculation(model_root_path: PathLike, n_list: List, max_bound: float = 1.0):
    """Calculate the relative error of the integration function for several models in
    a given directory.

    Notes:
        Experimental function! This has not been part of the study!

    Args:
        model_root_path:    Path to the directory containing the models.
        n_list:             List of how many integration points to use.
        max_bound:          Maximum bound of the sampling domain.

    Returns:
        Dictionary of the form {n: error} where n is the number of integration points

    """
    model, cfg = model_reader.read_models(model_root_path)[0]
    encoding = encodings.get_encoding(deep_get(cfg, ["Encoding", "encoding"]))
    sample = deep_get(cfg, ["Sample", "sample"])
    label_fn = label_binder.bind_label('polyhedral', True, sample=sample)
    target_points_sampler = get_target_point_sampler(
        100,
        method='altitude',
        bounds=[max_bound],
        limit_shape_to_asteroid=f"3dmeshes/{sample}_lp.pk"
    )
    target_points = target_points_sampler()
    true_value = label_fn(target_points)
    errors = {}
    for n in n_list:
        integration_fn = function_binder.bind_integration(
            method='trapezoid',
            use_acc=True,
            model=model,
            encoding=encoding,
            integration_points=n,
            integration_domain=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        )
        errors[n] = _calculate_relative_error(true_value, integration_fn, target_points)
    return errors


def _calculate_relative_error(true_value, integration_fn, target_points):
    """Calculate the relative error of the integration function compared to the ground truth."""
    approximated_value = integration_fn(target_points)

    true_error = abs(true_value - approximated_value)

    return float(abs(true_error / true_value).mean())
