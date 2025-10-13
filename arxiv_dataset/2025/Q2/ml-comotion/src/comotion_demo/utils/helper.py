# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Miscellaneous helper functions."""

import functools

import einops as eo
import numpy as np
import torch


def px_to_world(K, pts):
    """Convert values from pixel coordinates to world coordinates."""
    x, y = pts.unbind(-1)
    return torch.stack(
        [(x - K[..., 0, 2]) / K[..., 0, 0], (y - K[..., 1, 2]) / K[..., 1, 1]], -1
    )


def project_to_2d(K, pts3d, min_depth=0.001):
    """Project 3D points to pixel coordinates.

    Args:
    ----
      K: ... x 2 x 3
      pts3d: ... x 3
      min_depth: minimum depth for clamping (default: 0.001).

    """
    # Normalize by depth
    z = pts3d[..., -1:].clamp_min(min_depth)
    pts3d_norml = pts3d / z
    pts3d_norml[..., -1].fill_(1.0)

    # Apply intrinsics
    pts_2d = eo.einsum(K, pts3d_norml, "... i j, ... j -> ... i")

    return pts_2d


def _merge_aux(arr, dim=0, n_merged=2, unmerge=False, vals=None):
    n_extra = -(dim + 1) if dim < 0 else dim
    extra_d = " ".join(map(lambda x: f"a{x}", range(n_extra)))
    di = " ".join(map(lambda x: f"d{x}", range(n_merged)))
    df = f"({di})"
    kargs = {}

    if unmerge:
        di, df = df, di
        if isinstance(vals, int):
            vals = [vals]
        for v_idx, v in enumerate(vals):
            kargs[f"d{v_idx}"] = v

    if dim < 0:
        return eo.rearrange(arr, f"... {di} {extra_d} -> ... {df} {extra_d}", **kargs)
    else:
        return eo.rearrange(arr, f"{extra_d} {di} ... -> {extra_d} {df} ...", **kargs)


def _merge_d(arr, dim=0, n_merged=2):
    return _merge_aux(arr, dim, n_merged)


def _unmerge_d(arr, vals, dim=0, n_merged=2):
    return _merge_aux(arr, dim, n_merged, True, vals)


def fixed_dim_op(fn_=None, d=4, nargs=1, is_class_fn=False):
    def _decorator(fn):
        @functools.wraps(fn)
        def run_fixed_dim(*args, **kargs):
            args = [a for a in args]
            if is_class_fn:
                self = args[0]
                args = args[1:]

            tmp_nargs = min(len(args), nargs)
            ndim = args[0].ndim
            if ndim > d:
                # If input is greater than required dims
                # flatten first dimensions together
                n = ndim - d + 1
                d_ref = args[0].shape[:n]
                for i in range(tmp_nargs):
                    args[i] = _merge_d(args[i], 0, n)

                def adjust_fn(x):
                    return _unmerge_d(x, d_ref, 0, n) if x is not None else x

            elif ndim < d:
                # If input is less than the required dims
                # prepend extra singleton dimensions
                n = d - ndim
                one_str = " ".join(["1"] * n)
                for i in range(tmp_nargs):
                    args[i] = eo.rearrange(args[i], f"... -> {one_str} ...")

                def adjust_fn(x):
                    return (
                        eo.rearrange(x, f"{one_str} ... -> ...") if x is not None else x
                    )

            else:
                # Otherwise, do nothing
                def adjust_fn(x):
                    return x

            if is_class_fn:
                args = [self, *args]
            fn_out = fn(*args, **kargs)

            if isinstance(fn_out, tuple) or isinstance(fn_out, list):
                return [adjust_fn(v) for v in fn_out]
            elif isinstance(fn_out, dict):
                return {k: adjust_fn(v) for k, v in fn_out.items()}
            else:
                return adjust_fn(fn_out)

        return run_fixed_dim

    if fn_ is not None:
        return _decorator(fn_)
    else:
        return _decorator


def get_grid(ht, wd, device="cpu"):
    """Get coordinate grid for a set of features.

    Assume pixels are normalized to 1 (aspect-ratio adjusted so 1 is max(ht,wd))
    """
    grid = torch.meshgrid(
        torch.arange(wd, device=device), torch.arange(ht, device=device), indexing="xy"
    )
    grid = torch.stack(grid, -1) / float(max(ht, wd))
    return grid


_link_ref = torch.tensor(
    [
        (23, 24),  # eyes
        (22, 16),  # nose -> shoulder
        (22, 17),  # nose -> shoulder
        (16, 17),  # shoulders
        (16, 1),  # left shoulder -> hip
        (17, 2),  # right shoulder -> hip
        (16, 18),  # left shoulder -> elbow
        (17, 19),  # right shoulder -> elbow
        (20, 18),  # left wrist -> elbow
        (21, 19),  # right wrist -> elbow
        (4, 1),  # left knee -> hip
        (5, 2),  # right knee -> hip
        (4, 7),  # left knee -> ankle
        (5, 8),  # right knee -> ankle
    ]
).T


# Calculated based on default SMPL with above links
_link_dists_ref = [
    0.07,
    0.3,
    0.3,
    0.35,
    0.55,
    0.55,
    0.26,
    0.26,
    0.25,
    0.25,
    0.38,
    0.38,
    0.4,
    0.4,
]


def _get_skeleton_scale(kps: torch.Tensor, valid=None):
    """Return approximate "size" of person in image.

    Instead of bounding box dimensions, we compare pairs of keypoints as defined
    in the links above. We ignore any distances that involve an "invalid" keypoint.
    We assume any values set to (0, 0) are invalid and should be ignored.
    """
    invalid = (kps == 0).all(-1)
    if valid is not None:
        invalid |= ~(valid > 0)

    # Calculate link distances
    dists = (kps[..., _link_ref[0], :] - kps[..., _link_ref[1], :]).norm(dim=-1)

    # Compare to reference distances
    ratio = dists / torch.tensor(_link_dists_ref, device=dists.device)

    # Zero out any invalid links
    invalid = invalid[..., _link_ref[0]] | invalid[..., _link_ref[1]]
    ratio *= (~invalid).float()

    # Return max ratio which corresponds to limb with least foreshortening
    max_ratio = ratio.max(-1)[0]
    return max_ratio.clamp_min(0.001)


def normalized_weighted_score(
    kp0,
    c0,
    kp1,
    c1,
    std=0.08,
    return_dists=False,
    ref_scale=None,
    min_scale=0.02,
    max_scale=1,
    fixed_scale=None,
):
    """Measure similarity of two sets of body pose keypoints.

    This is a modified version of the COCO object keypoint similarity (OKS)
    calculation using a Cauchy distribution instead of a Gaussian. We also
    compute a normalizing scale on the fly based on projected limb proportions.
    """
    # Combine confidence terms
    # conf: ... num_people x num_people x num_points
    conf = (c0.unsqueeze(-2) * c1.unsqueeze(-3)) ** 0.5

    # Calculate scale adjustment
    scale0 = _get_skeleton_scale(kp0, c0)
    scale1 = _get_skeleton_scale(kp1, c1)
    if ref_scale is None:
        scale = scale0.unsqueeze(-1).maximum(scale1.unsqueeze(-2))
    elif ref_scale == 0:
        scale = scale0.unsqueeze(-1)
    elif ref_scale == 1:
        scale = scale1.unsqueeze(-2)

    # Set scale bounds
    zero_mask = scale == 0
    scale.clamp_(min_scale, max_scale)
    scale[zero_mask] = 1e-6
    if fixed_scale is not None:
        scale[:] = fixed_scale

    # Scale-adjusted distance calculation
    # kp: ... num_people x num_pts x 2
    # dists: ... num_people x num_people x num_pts
    dists = (kp0.unsqueeze(-3) - kp1.unsqueeze(-4)).norm(dim=-1)
    dists = dists / scale.unsqueeze(-1)

    total_conf = conf.sum(-1)
    zero_filt = total_conf == 0
    scores = 1 / (1 + (dists / std) ** 2)
    scores = (scores * conf).sum(-1)
    scores = scores / total_conf
    scores[zero_filt] = 0

    if return_dists:
        return scores, dists
    else:
        return scores


def nms_detections(pred_2d, conf, conf_thr=0.2, iou_thr=0.4, std=0.08):
    """Compare keypoint estimates and return indices of nonoverlapping estimates.

    Note: input predictions are expected to be normalized (e.g. ranging from 0 to 1),
    not in pixel coordinate space.
    """
    conf_sigmoid = torch.sigmoid(conf)

    # Get indices of estimates above confidence threshold, sorted by confidence
    sorted_idxs = (-conf).argsort()
    sorted_idxs = sorted_idxs[conf_sigmoid[sorted_idxs] > conf_thr]

    # Compare all pairs of keypoint estimates
    p = pred_2d[sorted_idxs]
    c = torch.ones_like(p[..., 0])
    ious = normalized_weighted_score(p, c, p, c, std=std).cpu()

    # Identify estimates that are lower confidence and too similar to another estimate
    triu = torch.triu_indices(len(p), len(p), offset=1)
    ious = ious[triu[0], triu[1]]
    to_remove = triu[1][ious > iou_thr]

    # Return the subset of estimates to keep
    to_keep = np.setdiff1d(np.arange(len(p)), to_remove.unique().numpy())
    final_idxs = sorted_idxs.cpu()[to_keep]
    return final_idxs


def check_inbounds(kps, res):
    """Return binary mask indicating which keypoints are inbounds."""
    in_x = (kps[..., 0] > 0) & (kps[..., 0] < res[1])
    in_y = (kps[..., 1] > 0) & (kps[..., 1] < res[0])
    inbounds = in_x & in_y
    return inbounds


def points_to_bbox2d(pts, pad_dims=None):
    """Get min and max range of set of keypoints to define 2d bounding-box.

    Output format is ((x0, y0), (x1, y1)).

    Args:
    ----
    pts: ... x K x 2
    pad_dims: optional padding (as percentage of bounding-box size)

    Output:
    ----
    bboxes: ... x 2 x 2

    """
    p = torch.stack([pts.min(-2)[0], pts.max(-2)[0]], -2)

    if pad_dims is not None:
        dimensions = p[..., 1, :] - p[..., 0, :]
        scale = dimensions.max(dim=-1)[0]
        pad_x = scale * pad_dims[0]
        pad_y = scale * pad_dims[1]

        p[..., 0, 0] -= pad_x
        p[..., 0, 1] -= pad_y
        p[..., 1, 0] += pad_x
        p[..., 1, 1] += pad_y

    return p


def hsv2rgb(hsv):
    """Convert a tuple from HSV to RGB."""
    h, s, v = hsv
    vals = np.tile(v, [3] + [1] * v.ndim)
    vals[1:] *= 1 - s[None]

    h[h > (5 / 6)] -= 1
    diffs = np.tile(h, [3] + [1] * h.ndim) - (np.arange(3) / 3).reshape(
        3, *[1] * h.ndim
    )
    max_idx = np.abs(diffs).argmin(0)

    final_rgb = np.zeros_like(vals)

    for i in range(3):
        tmp_d = diffs[i] * (max_idx == i)
        dv = tmp_d * 6 * s * v
        vals[1] += np.maximum(0, dv)
        vals[2] += np.maximum(0, -dv)

        final_rgb += np.roll(vals, i, axis=0) * (max_idx == i)

    return final_rgb.transpose(*list(np.arange(h.ndim) + 1), 0)


def init_color_ref(n, seed=12345):
    """Sample N colors in HSV and convert them to RGB."""
    rand_state = np.random.RandomState(seed)
    rand_hsv = rand_state.rand(n, 3)
    rand_hsv[:, 1:] = 1 - rand_hsv[:, 1:] * 0.3
    color_ref = hsv2rgb(rand_hsv.T)

    return color_ref


color_ref = init_color_ref(2000)
