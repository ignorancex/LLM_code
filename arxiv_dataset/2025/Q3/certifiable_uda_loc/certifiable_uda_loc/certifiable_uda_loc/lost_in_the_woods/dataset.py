from typing import Dict, Hashable, List, Tuple, Union

import numpy as np
import scipy
import scipy.io as sio

# from pymlg.numpy import SO2, SE2
from pymlg import SE2, SO2
from scipy.stats import multivariate_normal

from certifiable_uda_loc.lost_in_the_woods.residuals import (
    RangeBearingResidual,
)
from certifiable_uda_loc.lost_in_the_woods.utils import (
    egovehicle_key_at_stamp,
    index_list_by_numpy_index,
    landmark_key_at_idx,
    round_timestamp,
)

# from navlie.batch.gaussian_mixtures import GaussianMixtureResidual
# from navlie.batch.residuals import ProcessResidual
# from navlie.lib.models import BodyFrameVelocity, VectorInput
from navlie.lib.states import SE2State
from navlie.lib.states import VectorInput

# TODO: Need to convert random sections of this dataset into
# a factor graph form with pose transformations and landmark measurements.

from dataclasses import dataclass


class LandmarkMeasurement:
    def __init__(
        self,
        stamp: float,
        landmark_id: str,
        range_meas: float,
        bearing_meas: float,
        landmark_position: np.ndarray,
    ):
        self.stamp: float = stamp
        self.landmark_id: str = landmark_id
        self.range: float = range_meas
        self.bearing: float = bearing_meas
        self.landmark_position = landmark_position


class LostInTheWoodsDataset:

    def __init__(
        self,
        t: List[float],
        ranges: np.ndarray,
        bearings: np.ndarray,
        landmarks: List[np.ndarray],
        d_to_rangefinder: float,
        r_var: float,
        b_var: float,
        v_var: float,
        om_var: float,
        x_true: np.ndarray,
        y_true: np.ndarray,
        th_true: np.ndarray,
        om: np.ndarray,
        v: np.ndarray,
        time_bounds: List[float],
    ):
        idx = np.logical_and(
            np.array(t) >= time_bounds[0], np.array(t) <= time_bounds[1]
        )
        self.t = index_list_by_numpy_index(t, idx)
        self.t = [round_timestamp(stamp) for stamp in self.t]
        self.ranges = ranges[:, idx]
        self.bearings = bearings[:, idx]
        self.landmarks = landmarks
        self.d_to_rangefinder = d_to_rangefinder
        self.r_var = r_var
        self.b_var = b_var
        self.v_var = v_var
        self.om_var = om_var
        self.x_true = x_true[idx]
        self.y_true = y_true[idx]
        self.th_true = th_true[idx]
        self.om = index_list_by_numpy_index(om, idx)
        self.v = index_list_by_numpy_index(v, idx)
        # self.X_gt = self.ground_truth_states()
        # self.Qd, self.R = self.assemble_covariances()
        self.timestamp_to_index = {self.t[lv1]: lv1 for lv1 in range(len(self.t))}
        self.index_to_timestamp = {lv1: self.t[lv1] for lv1 in range(len(self.t))}

    def landmark_measurements(
        self, r_max=1000
    ) -> Tuple[List[List[RangeBearingResidual]], List[str]]:
        """Returns a list of lists of residuals corresponding to the range-bearing
        measurements to each landmark.
        residual_list_of_lists[lv1] is a list for landmark lv1.
        residual_list_of_lists[lv1][lv2] contains the lv2 residual for landmark lv1.
        Each residual requires 2 keys, robot and landmark.
        The robot one is set properly.
        The landmark one is set to None, and has to be set separately depending on
        if the data associations are known or not.
        Also return which landmarks are visible during the time range.
        Parameters
        ----------
        r_max : int, optional
            _description_, by default 1000
        spacing : int, optional
            _description_, by default 1

        Returns
        -------
        List[np.ndarray]
            _description_
        """
        # List of lists corresponding to each landmark.
        measurement_list_of_lists = []
        for lv_stamp in range(len(self.t)):
            measurement_list_of_lists.append([])
            range_all_landmarks = self.ranges[:, lv_stamp]
            bearings_all_landmarks = self.bearings[:, lv_stamp]
            timestamp = self.t[lv_stamp]

            for lv_lndmrk in range(range_all_landmarks.shape[0]):
                r = range_all_landmarks[lv_lndmrk]
                b = bearings_all_landmarks[lv_lndmrk]
                if r < r_max and r != 0.0:
                    measurement_list_of_lists[lv_stamp].append(
                        LandmarkMeasurement(
                            stamp=timestamp,
                            landmark_id=f"L{lv_lndmrk}",
                            range_meas=r,
                            bearing_meas=b,
                            landmark_position=self.landmarks[lv_lndmrk],
                        )
                    )
                else:
                    measurement_list_of_lists[lv_stamp].append(None)
        return measurement_list_of_lists

    def plot(self):
        import navlie.utils.plot as nv_plot
        import matplotlib.pyplot as plt

        lndmrks = np.hstack([l.reshape(-1, 1) for l in self.landmarks])

        fig, axs = plt.subplots(2, 1)
        axs: List[plt.Axes] = axs
        fig, axs[0] = nv_plot.plot_poses(
            self.ground_truth_states,
            axs[0],
            arrow_length=0.03,
            step=40,
            label="GT",
        )
        axs[0].scatter(lndmrks[0, :], lndmrks[1, :])

        dr_states = [self.ground_truth_states[0]]
        axs: List[plt.Axes] = axs
        fig, axs[1] = nv_plot.plot_poses(
            self.dead_reckoned_states,
            axs[1],
            arrow_length=0.03,
            step=40,
            label="DR",
        )
        axs[1].scatter(lndmrks[0, :], lndmrks[1, :])

        return fig, axs

    @property
    def dead_reckoned_states(self) -> List[SE2State]:
        from navlie.filters import ExtendedKalmanFilter
        from navlie.lib.models import BodyFrameVelocity
        import navlie.lib.states as nv_states

        process_model = BodyFrameVelocity(np.eye(3))  # covariance here doesnt matter
        kf = ExtendedKalmanFilter(process_model)
        x_k_hat = nv_states.StateWithCovariance(
            self.ground_truth_states[0].copy(), covariance=np.eye(3)
        )
        dr_states = [x_k_hat.copy()]

        for lv1 in range(len(self.inputs) - 1):
            dt = self.inputs[lv1 + 1].stamp - self.inputs[lv1].stamp
            x_k_hat, details_dict = kf.predict(
                x_k_hat.copy(), self.inputs[lv1], dt, output_details=True
            )
            dr_states.append(x_k_hat)
        dr_states = [x.state for x in dr_states]
        return dr_states

    @property
    def inputs(self) -> List[VectorInput]:
        input_list = [
            VectorInput(np.array([om_val, v_val, 0]), stamp)
            for om_val, v_val, stamp in zip(self.om, self.v, self.t)
        ]
        return input_list

    @property
    def ground_truth_states(self) -> List[SE2State]:
        X_gt = []
        for lv1 in range(len(self.t)):
            X_gt.append(
                SE2State(
                    SE2.from_components(
                        SO2.Exp(self.th_true[lv1]),
                        np.array([self.x_true[lv1], self.y_true[lv1]]),
                    ),
                    self.t[lv1],
                    direction="right",
                )
            )
        return X_gt

    @staticmethod
    def from_mat_file(fname: str, time_bounds: List[float]):
        mat_contents = sio.loadmat(fname)
        ranges = mat_contents["r"].T
        bearings = mat_contents["b"].T
        landmarks = mat_contents["l"].T
        landmarks = [landmarks[:, lv1] for lv1 in range(landmarks.shape[1])]
        d_to_rangefinder = mat_contents["d"].squeeze()
        r_var = mat_contents["r_var"].squeeze()
        b_var = mat_contents["b_var"].squeeze()
        v_var = mat_contents["v_var"].squeeze()
        om_var = mat_contents["om_var"].squeeze()
        x_true = mat_contents["x_true"].squeeze()
        y_true = mat_contents["y_true"].squeeze()
        th_true = mat_contents["th_true"].squeeze()
        om = mat_contents["om"].squeeze().tolist()
        v = mat_contents["v"].squeeze().tolist()

        return LostInTheWoodsDataset(
            t=mat_contents["t"].squeeze().tolist(),
            ranges=ranges,
            bearings=bearings,
            landmarks=landmarks,
            d_to_rangefinder=d_to_rangefinder,
            r_var=r_var,
            b_var=b_var,
            v_var=v_var,
            om_var=om_var,
            x_true=x_true,
            y_true=y_true,
            th_true=th_true,
            om=om,
            v=v,
            time_bounds=time_bounds,
        )
