import cv2 as cv
from skimage.measure import find_contours
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from typing import Union
from scipy import interpolate
from itertools import combinations
import warnings


warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_3d_bbox(mask: np.ndarray):
    assert mask.dtype == np.bool_
    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min
    bbox = (x_min, y_min, z_min, lx, ly, lz)
    mask_cropped = mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    return bbox, mask_cropped


# noinspection SpellCheckingInspection
# noinspection PyUnresolvedReferences
def confidence_ellipse(x: np.ndarray, y: np.ndarray, ax: mpl.axes.Axes, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError('x and y must be the same size')
    cov = np.cov(x, y)
    coe = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    el_ra_x = np.sqrt(1 + coe)
    el_ra_y = np.sqrt(1 - coe)
    ellipse = Ellipse((0, 0), width=el_ra_x * 2, height=el_ra_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = float(np.mean(x))
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = float(np.mean(y))
    trf = Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(trf + ax.transData)
    # ax.plot([mean_x, _com[0]], [0, _com[1]], label=f'component{i}')
    return ax.add_patch(ellipse)


def is_2d_bool_mask_at_boundary(_mask: np.ndarray, atol=1, print_warning: bool = False):
    # The main purpose of this function is to determine whether the mask is at the border. It is not important however
    # whether it finds all borders.
    assert _mask.dtype == np.bool_
    height, width = _mask.shape
    # Flip the coordinates to be consistent with displayed image using matplotlib
    coord = np.argwhere(_mask)[:, [1, 0]]
    if np.isclose(np.min(coord[:, 1]), atol, atol=atol):
        border_line = 'top'
        flag = 1
    elif np.isclose(np.max(coord[:, 0]), width - 1 - atol, atol=atol):
        border_line = 'right'
        flag = 2
    elif np.isclose(np.max(coord[:, 1]), height - 1 - atol, atol=atol):
        border_line = 'bottom'
        flag = 3
    elif np.isclose(np.min(coord[:, 0]), atol, atol=atol):
        border_line = 'left'
        flag = 4
    else:
        flag = 0
    if flag > 0 and print_warning:
        # noinspection PyUnboundLocalVariable
        warnings.warn(f'Mask is at {border_line} border')
    return flag


def get_2d_bool_mask_boundary_points(_mask: np.ndarray, atol=1) -> Union[int, dict]:
    # This function needs to find all boundary points and output them in a dictionary specifying each side.
    assert _mask.dtype == np.bool_
    height, width = _mask.shape
    # Flip the coordinates to be consistent with displayed image using matplotlib
    coord = np.argwhere(_mask)[:, [1, 0]]
    bound_top = coord[coord[:, 1] <= atol, :]
    bound_right = coord[coord[:, 0] >= width - 1 - atol, :]
    bound_bottom = coord[coord[:, 1] >= height - 1 - atol, :]
    bound_left = coord[coord[:, 0] <= atol, :]
    result = dict()
    for i, _bound in enumerate([bound_top, bound_right, bound_bottom, bound_left]):
        if len(_bound) > 0:
            result[i + 1] = _bound
    return result if len(result) > 0 else -1
    

def get_boundary_flag_dict(_flag: Union[int, None] = None):
    dict_flag = {1: 'top', 2: 'right', 3: 'bottom', 4: 'left'}
    if _flag is not None:
        return {_flag: dict_flag[_flag]}
    else:
        return dict_flag


# noinspection SpellCheckingInspection
def get_2d_oriented_bbox_from_mask(_mask: np.ndarray, label: Union[int, None] = None, plot_result: bool = False,
                                   save_file_name: Union[str, None] = None):
    try:
        assert _mask.dtype == np.uint8
    except AssertionError:
        if _mask.dtype == np.bool_:
            _mask = _mask.astype(np.uint8)
        else:
            raise ValueError('Mask array should either be boolean or uint8 type!')
    labels = np.unique(_mask[_mask > 0])
    if len(labels) > 1 and label is None:
        raise ValueError('For none-binary input mask, an integer label should be provided!')
    else:
        if len(labels) > 1:
            _mask = (_mask == label).astype(np.uint8)
    # It is possible to use CV to find contours and then calculate the oriented bounding box.
    # However, it seems that the contour coordinates obtained using CV contain fewer elements than expected. Therefore,
    # it is decided to switch to skimage for getting the contours. For compatibility issue, all retrieved contour
    # coordinates need to reverse the x and y coordinates, and convert to int32 values for computing the bbox.
    #### The following commented codes can work without triggering errors.
    # contours, _ = cv.findContours(_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnt = np.squeeze(contours[0], axis=1)
    flag = is_2d_bool_mask_at_boundary(_mask == 1)
    contours = find_contours(_mask, 0.5)
    cnt = contours[0].astype(np.int32)[:, [1, 0]]
    rect = cv.minAreaRect(cnt)  # Notice that the coordinates have to be swapped.
    bbox = np.intp(cv.boxPoints(rect))
    if plot_result:
        plt.imshow(_mask)
        bbox_closed = np.vstack((bbox, bbox[0, :]))
        plt.plot(bbox_closed[:, 0], bbox_closed[:, 1], color='gray')
        plt.axis('off')
        if save_file_name is None:
            plt.show()
        else:
            plt.savefig(save_file_name)
    return bbox, cnt, flag


def get_slope_and_intercept_from_2points(points: np.ndarray):
    assert points.shape == (2, 2)
    if points[1, 0] != points[0, 0]:
        k = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0])  # slope
        b = points[0, 1] - k * points[0, 0]  # intercept
    else:
        k = np.nan
        b = np.nan
    return k, b


def smooth_curve_with_spline(curve: np.ndarray):
    _, idx = np.unique(curve, return_index=True, axis=0)
    # Sort the indices of unique coordinates are important, otherwise the interpolated results will be completely
    # random when plotted.
    idx.sort()
    tck, u = interpolate.splprep(curve[idx, :].T, s=0)
    u_new = np.linspace(u.min(), u.max(), len(curve))
    out = interpolate.splev(u_new, tck, der=0)
    return np.vstack(out).T


def get_intersection_between_2d_line_and_curve(curve: np.ndarray, points: np.ndarray, k=None):
    assert min(curve.shape) == 2
    assert len(points.flatten()) in [2, 4]  # two point line or single point with 2D coordinates.
    if curve.shape[1] > curve.shape[0]:
        curve = curve.T
    if k is None:
        k, b = get_slope_and_intercept_from_2points(points)
    else:
        b = points[1] - k * points[0]
    if not np.isnan(k):
        # Project all points of the curve to the fitted straight line and calculate the differences of the projected
        # points from the true y coordinates.
        diff = (k * curve[:, 0] + b) - curve[:, 1]
        # Get the indices of points where the next point is on the other side of the line by checking the signs of
        # differences.
        idx_cross = np.where(diff[1:] * diff[:-1] < 0)[0]
        # Use linear interpolation to obtain crossing points
        ratio_diff = diff[idx_cross] / (diff[idx_cross] - diff[idx_cross + 1])
        pt_cross = np.zeros((len(idx_cross), 2))
        pt_cross[:, 0] = curve[idx_cross, 0] + ratio_diff * (curve[idx_cross + 1, 0] - curve[idx_cross, 0])
        pt_cross[:, 1] = curve[idx_cross, 1] + ratio_diff * (curve[idx_cross + 1, 1] - curve[idx_cross, 1])
        if any(diff == 0):
            # Useful for detecting points directly on the curve
            idx_on_curve = np.where(diff == 0)[0]
            pt_cross = np.vstack((pt_cross, curve[idx_on_curve, :]))
    else:
        pt_cross = np.zeros((2, 2))
        point = points[0, :] if len(points.flatten()) == 4 else points
        indices = np.argwhere(np.isclose(curve[:, 0], point[0], atol=0.5))
        pt_cross[0, :] = curve[indices[np.argmin(curve[indices, 1])], :]
        pt_cross[1, :] = curve[indices[np.argmax(curve[indices, 1])], :]
    return pt_cross


def get_line_from_nearest_neighbors(_line, curve):
    points_nn = np.zeros((2, 2))
    for _i in range(2):
        dist = np.linalg.norm(_line[_i, :].reshape(1, -1) - curve, axis=1)
        points_nn[_i, :] = curve[np.argmin(dist), :]
    points_candidates = get_intersection_between_2d_line_and_curve(curve, points_nn)
    points = np.vstack((points_nn, points_candidates))
    return get_line_from_candidates(points)


def get_line_from_candidates(_points):
    comb = list(combinations(np.arange(_points.shape[0]), 2))
    _dist = np.zeros(len(comb))
    for _i, _c in enumerate(comb):
        _dist[_i] = np.linalg.norm(_points[_c[0], :] - _points[_c[1], :])
    idx_comb = np.argmax(_dist)
    return _points[np.array(comb[idx_comb]), :], _dist[idx_comb]


def get_side_from_closed_curve(curve: np.ndarray, side: str = 'left'):
    assert len(curve.shape) == 2 and min(curve.shape) == 2
    if curve.shape[0] < curve.shape[1]:
        curve = curve.T
    num_points = curve.shape[0]
    if len(side) > 1:
        assert side in ['left', 'top', 'bottom', 'right']
    else:
        assert side in ['l', 't', 'b', 'r']
    centroid = np.mean(curve, axis=0)
    if side[0] == 'l':
        idx = np.array([i for i in range(num_points) if curve[i, 0] < centroid[0]])
    elif side[0] == 't':
        idx = np.array([i for i in range(num_points) if curve[i, 1] < centroid[1]])
    elif side[0] == 'b':
        idx = np.array([i for i in range(num_points) if curve[i, 1] > centroid[1]])
    else:
        idx = np.array([i for i in range(num_points) if curve[i, 0] > centroid[0]])
    return curve[idx, :]


def get_point_angle_and_dist_cw(point: np.ndarray, origin: np.ndarray, vec_ref: Union[np.ndarray, None] = None):
    assert len(point) == len(origin) == 2
    if vec_ref is None:
        vec_ref = np.array([1, 0])
    vec = point - origin
    vec_len = np.linalg.norm(vec)
    if vec_len == 0:
        angle = -math.pi
    else:
        vec_n = vec/vec_len
        angle = math.atan2(np.dot(vec_n, vec_ref), np.cross(vec_n, vec_ref))
        if angle < 0:
            # Convert negative angle to positive one.
            angle += 2*math.pi
    return angle, vec_len


def sort_curve_points_cw(curve: np.ndarray, vec_ref=None):
    if vec_ref is None:
        vec_ref = np.array([1, 0])

    assert len(curve.shape) == 2 and min(curve.shape) == 2
    if curve.shape[0] < curve.shape[1]:
        curve = curve.T
    centroid = np.mean(curve, axis=0)
    num_points = curve.shape[0]
    angles = np.zeros((num_points, 1))
    for i in range(num_points):
        angle, _ = get_point_angle_and_dist_cw(curve[i, :], centroid, vec_ref)
        angles[i] = angle
    curve_with_angle = np.hstack((curve, np.degrees(angles)))
    curve_sorted = curve_with_angle[curve_with_angle[:, 2].argsort()][:, [0, 1]]
    return curve_sorted


def mask_bbox_dists_with_midpoints(_mask: np.ndarray, label: Union[int, None] = None, smooth_contour: bool = False):
    bbox, cnt, flag = get_2d_oriented_bbox_from_mask(_mask, label)
    if flag == 0:
        if smooth_contour:
            cnt = smooth_curve_with_spline(cnt)
        line_left = get_side_from_closed_curve(bbox, 'l')
        line_right = get_side_from_closed_curve(bbox, 'r')
        k, _ = get_slope_and_intercept_from_2points(line_left)
        line_x_l, dist_l = get_line_from_nearest_neighbors(line_left, cnt)
        line_x_r, dist_r = get_line_from_nearest_neighbors(line_right, cnt)
        dists = np.zeros(3)
        dists[0] = dist_l
        # Get the line crossing the centroid point of the bbox
        points_x_m = get_intersection_between_2d_line_and_curve(cnt, np.mean(bbox, axis=0), k)
        line_x_m, dist_m = get_line_from_candidates(points_x_m)
        dists[1] = dist_m
        dists[2] = dist_r
        result = {'BBox': bbox, 'Distances': dists, 'Landmarks': np.vstack((line_x_l, line_x_m, line_x_r))}
    else:
        result = get_boundary_flag_dict(flag)
    return result

