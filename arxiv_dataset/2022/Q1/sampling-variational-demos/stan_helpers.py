import numpy as np
import stan
from math import prod
import pandas as pd
from io import StringIO


def constrained_dim(mdl: stan.model.Model) -> int:
    # Note: scalars have empty dimension, but prod([]) is 1
    return sum(map(prod, mdl.dims))


def unconstrained_dim(mdl: stan.model.Model) -> int:
    cdim = constrained_dim(mdl)
    for _ in range(10):
        dummy_x = np.random.rand(cdim)
        try:
            dummy_r = unconstrain_sample(mdl, dummy_x)
            return dummy_r.size
        except (ValueError, RuntimeError) as e:
            pass
    raise e


def unflatten_constrained_params(mdl: stan.model.Model, x: np.ndarray) -> dict:
    k = 0
    out = {}
    for para, dim in zip(mdl.param_names, mdl.dims):
        if len(dim) == 0:
            # Scalar case
            out[para] = x[k]
            k += 1
        else:
            # Multi-dimensional case
            out[para] = np.zeros(dim)
            for j in range(out[para].size):
                # item_name for parameter 'x' are things like x.1.2 for 1st row 2nd col
                parts = mdl.constrained_param_names[k + j].split('.')
                assert parts[0] == para, f"Mismatch! {para} != {mdl.constrained_param_names[k + j]}"
                subs = [int(idx)-1 for idx in parts[1:]]
                out[para][subs] = x[k + j]
            k += out[para].size
            out[para] = out[para].tolist()
    assert k == len(mdl.constrained_param_names), "Something went wrong in terms of dimensions"
    return out


def unconstrain_sample(mdl: stan.model.Model, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]

    if len(mdl.constrained_param_names) != x.shape[1]:
        raise ValueError(f'Only {x.shape[1]} values in x but require {len(mdl.constrained_param_names)} constrained params!')

    return np.stack([
        mdl.unconstrain_pars(unflatten_constrained_params(mdl, s))
        for s in x
    ], axis=0)


def read_summary(summary_file):
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    # Drop blank lines
    lines = [l.rstrip() for l in lines if len(l.strip()) > 0]
    # Find index of header row
    idx_header = np.where([l.endswith('R_hat') for l in lines])[0][0]
    # Find index of first non-data row
    idx_footer = np.where([l.startswith('Samples') for l in lines])[0][0]
    # Keep only the lines specifying the table of data
    lines = lines[idx_header:idx_footer]
    # (Temporary?) further drop metadata – rows whose names end with '__'
    lines = [l for l in lines if not l.lstrip().split(' ')[0].endswith('__')]
    # Construct 'CSV', but really space-separated string
    table = "\n".join(lines)
    # Replace instances like "-0.5-0.2" with spaces, "-0.5 -0.2", but not things like " 1e-2"
    table = table.replace("-", " -").replace("e -", "e-")
    return pd.read_csv(StringIO(table), sep='\s+')
