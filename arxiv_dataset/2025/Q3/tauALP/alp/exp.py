import os
import numpy as np
import pandas as pd
import glob
import vector
import pickle

from collections import defaultdict
from . import models


def save_events_to_pickle(data_dict, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)


def load_events_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_pythia_file_with_attrs(file_path, n_header_lines=8, dataframe=False):
    attrs = {}
    column_names = None
    data_dict = None

    with open(file_path, "r") as f:
        for i in range(n_header_lines):
            line = next(f)
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip().lower().replace(" ", "_")[2:]
                value = value.strip()
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass
                attrs[key] = value

        if dataframe:
            df = pd.read_csv(file_path, sep=" ", skiprows=n_header_lines)
            df.attrs.update(attrs)
            return df

        # Continue reading file
        data_dict = {}
        for line in f:
            if line.startswith("#") and "event_number" in line:
                column_names = line.strip("#").strip().split()
                data_dict = {name: [] for name in column_names}
                continue

            if column_names is None:
                continue

            if line.startswith("#") or not line.strip():
                continue
            values = line.strip().split()
            for name, value in zip(column_names, values):
                data_dict[name].append(float(value))

    if not data_dict:
        raise ValueError(f"No event data found in file {file_path}")

    for key in data_dict:
        if key in ["event_number", "particle_count", "mother_pid", "pid"]:
            data_dict[key] = np.array(data_dict[key], dtype=np.int32)
        else:
            data_dict[key] = np.array(data_dict[key], dtype=np.float64)
    data_dict.update(attrs)
    return data_dict


def load_events(
    file_names,
    apply_new_tauBR_weights=False,
    apply_xsec_weights=False,
):
    if not isinstance(file_names, list):
        files = glob.glob(f"{file_names}_*.txt")
    else:
        files = []
        for file_name in file_names:
            files += glob.glob(f"{file_name}_*.txt")

    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise FileNotFoundError(f"No valid files matching {file_names}_*.txt")

    print(f"Found {len(files)} files matching {file_names}_*.txt")

    # Tau decay BR corrections
    PARENTS = [411, 431, 100443]
    BRANCHINGS = [1.20e-3, 5.36e-2, 3.1e-3]

    # Accumulators
    data_acc = defaultdict(list)
    total_tau_xsec = 0.0
    total_xsec = 0.0
    total_events = 0

    for file_path in files:
        data_dict = read_pythia_file_with_attrs(file_path)
        if data_dict.get("mode", "soft") != "soft":
            raise ValueError(f"Expected only 'soft' mode files. Found: {file_path}")

        tau_xsec = data_dict.get("tau_xsec_mb", 0.0)
        total_xsec_mb = data_dict.get("total_xsec_mb", 0.0)

        # Adjust event_number to be continuous
        event_numbers = data_dict["particle_count"]
        data_dict["event_number"] = event_numbers + total_events
        n_events = int(np.max(data_dict["particle_count"]) + 1)
        total_events = n_events

        # Apply weights
        if apply_new_tauBR_weights:
            weights = np.ones(len(data_dict["weights"]))
            if apply_xsec_weights:
                weights *= tau_xsec
            if apply_new_tauBR_weights:
                mother_pids = data_dict["mother_pid"]
                for pid, br in zip(PARENTS, BRANCHINGS):
                    mask = np.abs(mother_pids) == pid
                    weights[mask] *= br
            data_dict["weights"] = weights

        # Accumulate
        for key, arr in data_dict.items():
            if isinstance(arr, np.ndarray):
                data_acc[key].append(arr)

        total_tau_xsec += tau_xsec / len(files)
        total_xsec += total_xsec_mb / len(files)

    # Final assembly
    full_data = {key: np.concatenate(arrays) for key, arrays in data_acc.items()}
    full_data["tau_xsec_mb"] = total_tau_xsec
    full_data["total_xsec_mb"] = total_xsec
    if apply_xsec_weights:
        full_data["weights"] /= total_events * total_xsec

    return full_data


class Experiment:
    def __init__(
        self, file_paths, exp_dic, alp=None, duplicate_taus=None, savemem=True
    ):
        """
        Initializes the experimental setup with the given parameters.

        Args:
            file_paths (list of str): List of file paths containing Pythia8 tau events.

            exp_dic (dict): Dictionary containing experimental attributes.

                    Required keys:
                    - name (str): Name of the experiment.
                    - L (float): Distance to the detector.
                    - dZ (float): Detector depth.
                    - norm (float): Normalization factor.
                    - Emin (float): Minimum energy threshold.
                    - final_states (list): List of ALP decay final states (e.g., ["ee", "em", "me", "mm","gg"]).

                Optional keys:

                    - R (float): Radius of the detector (used for circular detectors).
                    - dX (float): Width of the detector (used for rectangular detectors).
                    - dY (float): Height of the detector (used for rectangular detectors).
                    - theta0 (float): Angle of the detector center in the plane of the detector.
                    - x0 (float): X-coordinate of the detector center.
                    - y0 (float): Y-coordinate of the detector center.

            alp (object, optional): ALP (Axion-Like Particle) model instance.

                If not provided, a default ALP model is initialized.

        Raises:
            ValueError: If any required key is missing from `exp_dic`.

        Attributes:
            df_taus (DataFrame): DataFrame containing loaded event data.
            nevents (int): Number of events in the dataset.
            weight (Series): Normalized weights for the events.
            dtheta (float): Angular width of the detector in the theta direction.
            dphi (float): Angular width of the detector in the phi direction.
            x0 (float): X-coordinate of the detector center.
            y0 (float): Y-coordinate of the detector center.
            alp (object): ALP model instance.
        """

        self.savemem = savemem
        # All experimental attributes
        required_keys = ["name", "L", "dZ", "norm", "Emin", "final_states"]
        for key, value in exp_dic.items():
            try:
                required_keys.pop(required_keys.index(key))
            except ValueError:
                pass
            setattr(self, key, value)
        if required_keys:
            raise ValueError(f"Missing keys: {required_keys}")

        if "R" in exp_dic:
            self.R = exp_dic["R"]
            self.dtheta = np.arctan(self.R / self.L)
            self.dphi = np.arctan(self.R / self.L)
        else:
            self.dX = exp_dic["dX"]
            self.dY = exp_dic["dY"]
            self.dtheta = np.arctan(self.dX / self.L)
            self.dphi = np.arctan(self.dY / self.L)

        if "theta0" in exp_dic:
            # detector center at the plane of the detector
            self.x0 = self.L * np.tan(self.theta0)
            self.y0 = 0
        else:
            # detector center at the plane of the detector
            self.x0 = exp_dic["x0"]
            self.y0 = exp_dic["y0"]

        if alp is not None:
            self.alp = alp
        else:
            self.alp = models.ALP(0.5, 1e5)
        self.tau_BRs = {}

        if isinstance(file_paths, list) or (
            isinstance(file_paths, str) and ".pkl" not in file_paths
        ):
            df_taus = load_events(file_paths)
        elif isinstance(file_paths, str) and ".pkl" in file_paths:
            df_taus = load_events_from_pickle(file_paths)

        # Reutilize tau events
        if duplicate_taus is not None:
            if duplicate_taus >= 1:
                for key, value in df_taus.items():
                    if isinstance(value, np.ndarray):
                        df_taus[key] = np.repeat(value, duplicate_taus, axis=0)
            else:
                raise ValueError("duplicate_taus must be an integer >= 1")

        # Event weights (from Pythia8 and the concatenation of files)
        self.tau_weights = df_taus["weights"]
        self.tau_weights /= self.tau_weights[~np.isnan(self.tau_weights)].sum()

        # 4-momenta of taus
        self.p4_taus = vector.array(
            {
                "E": df_taus["E"],
                "px": df_taus["px"],
                "py": df_taus["py"],
                "pz": df_taus["pz"],
            }
        )
        if not savemem:
            self.df_taus = df_taus
        del df_taus

        # Total number of events in this simulation
        self.nevents = len(self.tau_weights)

        self.event_rate = self.get_event_rate(self.alp)

    def get_parent_p4(self):
        p4_parent = vector.array(
            {
                "E": self.df_taus["E_mother"],
                "px": self.df_taus["px_mother"],
                "py": self.df_taus["py_mother"],
                "pz": self.df_taus["pz_mother"],
            }
        )
        return p4_parent

    def sample_alp_energy_spectrum(self, production_channel, alp, nevents=None):
        if nevents is None:
            nevents = self.nevents
        ECM_alp = np.random.uniform(
            alp.Ea_min[production_channel], alp.Ea_max[production_channel], nevents
        )
        return ECM_alp

    def sample_alp_daughter_4momenta(self, p4_alp, decay_channel, alp, nevents=None):
        if nevents is None:
            nevents = self.nevents

        # Angular variables
        phi_1 = np.random.uniform(0, 2 * np.pi, nevents)
        ctheta_1 = np.random.uniform(-1, 1, nevents)
        stheta_1 = np.sqrt(1 - ctheta_1**2)
        cos_phi = np.cos(phi_1)
        sin_phi = np.sin(phi_1)

        # Daughter masses
        m1 = models.LEPTON_MASSES[models.LEPTON_INDEX[decay_channel[0]]]
        m2 = models.LEPTON_MASSES[models.LEPTON_INDEX[decay_channel[1]]]

        # CM energies
        ECM_1 = (alp.m_a**2 + m1**2 - m2**2) / (2 * alp.m_a)
        ECM_2 = (alp.m_a**2 - m1**2 + m2**2) / (2 * alp.m_a)
        ECM_1_arr = np.full(nevents, ECM_1)
        ECM_2_arr = np.full(nevents, ECM_2)

        # CM momentum
        pCM_1 = np.where(ECM_1 > m1, np.sqrt(ECM_1**2 - m1**2), 0.0)

        px = pCM_1 * stheta_1 * cos_phi
        py = pCM_1 * stheta_1 * sin_phi
        pz = pCM_1 * ctheta_1

        # Build CM-frame four-momenta
        p4_CM_1 = vector.array({"px": px, "py": py, "pz": pz, "E": ECM_1_arr})
        p4_CM_2 = vector.array({"px": -px, "py": -py, "pz": -pz, "E": ECM_2_arr})

        # Boost to lab frame
        p4_1 = p4_CM_1.boost_p4(p4_alp)
        p4_2 = p4_CM_2.boost_p4(p4_alp)

        return p4_1, p4_2

    def generate_alp_events(
        self, alp, production_channel, decay_channel, tau_mask=None
    ):
        """
        Generate ALP events for a given tau decay and ALP decay channel.
        """
        if tau_mask is None:
            tau_mask = np.ones(self.nevents, dtype=bool)

        p4_tau = self.p4_taus[tau_mask]
        n_taus = len(p4_tau)

        # Vectorized sampling of angles
        phi_alp = np.random.uniform(0, 2 * np.pi, n_taus)
        ctheta_alp = np.random.uniform(-1, 1, n_taus)
        stheta_alp = np.sqrt(1.0 - np.square(ctheta_alp))

        # Sample ALP energy spectrum
        ECM_alp = self.sample_alp_energy_spectrum(
            production_channel, alp, nevents=n_taus
        )
        m_a2 = alp.m_a**2
        pCM_alp = np.sqrt(np.maximum(ECM_alp**2 - m_a2, 0))

        # Build CM-frame ALP 4-momentum
        cos_phi = np.cos(phi_alp)
        sin_phi = np.sin(phi_alp)
        px = pCM_alp * stheta_alp * cos_phi
        py = pCM_alp * stheta_alp * sin_phi
        pz = pCM_alp * ctheta_alp

        p4_alp_CM = vector.array({"px": px, "py": py, "pz": pz, "E": ECM_alp})

        # Boost to lab frame
        p4_alp = p4_alp_CM.boost_p4(p4_tau)

        # Generate ALP daughters
        p4_1, p4_2 = self.sample_alp_daughter_4momenta(
            p4_alp, decay_channel=decay_channel, alp=alp, nevents=n_taus
        )

        # Calculate weights based on tau decay branching ratios and tau-event weights
        if production_channel in {"tau>e+a", "tau>mu+a"}:
            tau_BR = alp.tau_BR(production_channel)
            self.tau_BRs[production_channel] = tau_BR
            weights = self.tau_weights[tau_mask] * tau_BR
        else:
            diff_BR_vals = alp.tau_diff_BR(ECM_alp, production_channel)
            e_range = alp.Ea_max[production_channel] - alp.Ea_min[production_channel]
            tau_BR = (
                np.mean(diff_BR_vals) * e_range
            )  # faster and equivalent to sum/N * range
            self.tau_BRs[production_channel] = tau_BR
            weights = self.tau_weights[tau_mask] * diff_BR_vals

        return p4_alp, p4_1, p4_2, weights

    def get_alp_events(self, alp=None):
        """
        Generate ALP events from all tau decay channels.
        """
        if alp is None:
            alp = self.alp

        # Preload branching ratios for ALP decays
        br_decay_dict = {
            decay: getattr(alp, f"BR_a_to_{decay}") for decay in self.final_states
        }

        # Compute tau production × ALP decay channel weights
        tau_channels = [
            "tau>e+a",
            "tau>mu+a",
            "tau>nu+pi+a",
            "tau>nu+rho+a",
            "tau>nu+nu+e+a",
            "tau>nu+nu+mu+a",
        ]
        channel_weights = {
            (prod, decay): alp.tau_BR(prod) * br_decay_dict[decay]
            for prod in tau_channels
            for decay in self.final_states
        }

        # Normalize and keep only significant channels
        total_weight = sum(channel_weights.values())
        channel_weights = {
            k: w / total_weight
            for k, w in channel_weights.items()
            if w / total_weight > 1e-3
        }

        # Assign events to channels
        total_events = self.nevents

        channel_keys = list(channel_weights.keys())
        weight_values = list(channel_weights.values())

        n_events_in_channel = (np.array(weight_values) * total_events).astype(int)
        n_events_in_channel[-1] += (
            total_events - n_events_in_channel.sum()
        )  # fix rounding error

        self.event_splits = {}
        i = 0
        for (prod, decay), n_events_this_channel in zip(
            channel_keys, n_events_in_channel
        ):
            self.event_splits[(prod, decay)] = np.zeros(self.nevents, dtype=bool)
            self.event_splits[(prod, decay)][i : i + n_events_this_channel] = True
            i += n_events_this_channel

        # Preallocate component arrays
        E_alp = np.empty(total_events)
        px_alp = np.empty(total_events)
        py_alp = np.empty(total_events)
        pz_alp = np.empty(total_events)

        E_d1 = np.empty(total_events)
        px_d1 = np.empty(total_events)
        py_d1 = np.empty(total_events)
        pz_d1 = np.empty(total_events)

        E_d2 = np.empty(total_events)
        px_d2 = np.empty(total_events)
        py_d2 = np.empty(total_events)
        pz_d2 = np.empty(total_events)

        weights = np.empty(total_events)

        # Generate events per channel
        for (prod, decay), channel_mask in self.event_splits.items():
            p4_alp, p4_d1, p4_d2, w = self.generate_alp_events(
                alp, prod, decay, tau_mask=channel_mask
            )

            # Store components directly using vector.array access
            px_alp[channel_mask] = p4_alp.px
            py_alp[channel_mask] = p4_alp.py
            pz_alp[channel_mask] = p4_alp.pz
            E_alp[channel_mask] = p4_alp.E

            px_d1[channel_mask] = p4_d1.px
            py_d1[channel_mask] = p4_d1.py
            pz_d1[channel_mask] = p4_d1.pz
            E_d1[channel_mask] = p4_d1.E

            px_d2[channel_mask] = p4_d2.px
            py_d2[channel_mask] = p4_d2.py
            pz_d2[channel_mask] = p4_d2.pz
            E_d2[channel_mask] = p4_d2.E

            weights[channel_mask] = w
            weights[channel_mask] /= weights[channel_mask].sum()

        # Build vector arrays from components (fast, vectorized)
        self.p4_alp = vector.array(
            {"px": px_alp, "py": py_alp, "pz": pz_alp, "E": E_alp}
        )
        self.p4_daughter1 = vector.array(
            {"px": px_d1, "py": py_d1, "pz": pz_d1, "E": E_d1}
        )
        self.p4_daughter2 = vector.array(
            {"px": px_d2, "py": py_d2, "pz": pz_d2, "E": E_d2}
        )
        self.weights = weights

        v_alp = self.p4_alp.to_3D().unit()

        self.z_alp = np.random.uniform(self.L, self.L + self.dZ, size=total_events)
        dz_to_exit = self.L + self.dZ - self.z_alp

        self.x_alp = v_alp["x"] * self.z_alp
        self.y_alp = v_alp["y"] * self.z_alp

        # If the detector is not fully active, calculate the exit point of the daughters to see if they hit the back of the detector
        if self.active_volume or not self.savemem:
            v_d1 = self.p4_daughter1.to_3D().unit()
            v_d2 = self.p4_daughter2.to_3D().unit()

            self.x_daughter1_exit = self.x_alp + v_d1["x"] * dz_to_exit
            self.y_daughter1_exit = self.y_alp + v_d1["y"] * dz_to_exit

            self.x_daughter2_exit = self.x_alp + v_d2["x"] * dz_to_exit
            self.y_daughter2_exit = self.y_alp + v_d2["y"] * dz_to_exit

        return self.p4_alp, self.weights

    def get_alps_in_acceptance(self, generate_events=True, alp=None):

        if generate_events or hasattr(self, "p4_alp") is False:
            self.get_alp_events(alp=alp)

        # NOTE: Selection of events in a square assuming a cuboid
        if hasattr(self, "R"):
            if self.active_volume:
                self.mask_alp_in_acc = (
                    (
                        (
                            (self.x_daughter1_exit - self.x0) ** 2
                            + (self.y_daughter1_exit - self.y0) ** 2
                        )
                        < self.R**2
                    )
                    & (
                        (
                            (self.x_daughter2_exit - self.x0) ** 2
                            + (self.y_daughter2_exit - self.y0) ** 2
                        )
                        < self.R**2
                    )
                    & (self.p4_daughter1["E"] > self.Emin / 2)
                    & (self.p4_daughter2["E"] > self.Emin / 2)
                )
            else:
                self.mask_alp_in_acc = (
                    (
                        ((self.x_alp - self.x0) ** 2 + (self.y_alp - self.y0) ** 2)
                        < self.R**2
                    )
                    & (self.p4_daughter1["E"] > self.Emin / 2)
                    & (self.p4_daughter2["E"] > self.Emin / 2)
                )

        else:
            if self.active_volume:
                self.mask_alp_in_acc = (
                    (
                        ((np.abs(self.x_daughter1_exit - self.x0)) < self.dX / 2)
                        & ((np.abs(self.y_daughter1_exit - self.y0)) < self.dY / 2)
                        & ((np.abs(self.x_daughter2_exit - self.x0)) < self.dX / 2)
                        & ((np.abs(self.y_daughter2_exit - self.y0)) < self.dY / 2)
                    )
                    & (self.p4_daughter1["E"] > self.Emin / 2)
                    & (self.p4_daughter2["E"] > self.Emin / 2)
                )
            else:
                self.mask_alp_in_acc = (
                    (np.abs(self.x_alp - self.x0) < self.dX / 2)
                    & (np.abs(self.y_alp - self.y0) < self.dY / 2)
                    & (self.p4_daughter1["E"] > self.Emin / 2)
                    & (self.p4_daughter2["E"] > self.Emin / 2)
                )

        self.eff = self.weights[self.mask_alp_in_acc].sum() / self.weights.sum()

        return (
            self.p4_alp[self.mask_alp_in_acc],
            self.weights[self.mask_alp_in_acc],
        )

    def get_alp_momentum_spectrum(
        self, alp=None, selection=True, bins=40, generate_events=True
    ):
        """Get histogram of ALP momenta produced in a given tau decay channel

                * returns the histogram of ALP momenta, the bin edges, and the geom acceptance
        Returns:
            np.ndarray: the area normalized histogram of ALP momenta
            np.ndarray: the bin edges
        """

        p_alp = self.p4_alp.to_3D().mag
        if generate_events:
            # Get ALP events for all channels
            self.get_alp_events(alp=alp)
            self.get_alps_in_acceptance(alp=alp)

        # If less than 3 generated events were within acceptance, dont even try to compute rate
        if self.mask_alp_in_acc.sum() < 3:
            if isinstance(bins, int):
                p_bins = np.linspace(p_alp.min(), p_alp.max(), bins)
            elif isinstance(bins, np.ndarray):
                p_bins = bins
            self.geom_acceptance = 0
            return np.zeros(len(p_bins) - 1), p_bins

        # Else, histogram event rate in alp energy
        else:
            if isinstance(bins, int):
                p_bins = np.linspace(p_alp.min(), p_alp.max(), bins)
            elif isinstance(bins, np.ndarray):
                p_bins = bins

            if selection:
                h, p_bins = np.histogram(
                    p_alp[self.mask_alp_in_acc],
                    bins=p_bins,
                    weights=self.weights[self.mask_alp_in_acc],
                )
            else:
                h, p_bins = np.histogram(
                    p_alp,
                    bins=p_bins,
                    weights=self.weights,
                )
            self.geom_acceptance = (
                self.weights[self.mask_alp_in_acc].sum() / self.weights.sum()
            )

            return h, p_bins

    def get_event_rate(self, alp, generate_events=True):

        if generate_events:
            self.get_alps_in_acceptance(generate_events=generate_events, alp=alp)

        self.total_rate = 0.0
        for (
            prod_channel,
            decay_channel,
        ), channel_mask in self.event_splits.items():

            mask_accepted_and_channel = self.mask_alp_in_acc & channel_mask
            weights = self.weights[mask_accepted_and_channel]
            probs = self.get_signal_prob_decay(
                alp,
                prod_channel=prod_channel,
                decay_channel=decay_channel,
                mask=mask_accepted_and_channel,
            )
            # Remove NaNs from both arrays
            valid = ~(np.isnan(weights) | np.isnan(probs))
            self.total_rate += np.sum(weights[valid] * probs[valid])

        # normalization factor from POT*(tau/POT)
        self.total_rate *= self.norm

        if self.savemem:
            # del self.p4_alp
            del self.p4_daughter1
            del self.p4_daughter2
            del self.x_alp
            del self.y_alp
            del self.z_alp
            # del self.p4_taus
            if self.active_volume:
                del self.x_daughter1_exit
                del self.x_daughter2_exit
                del self.y_daughter1_exit
                del self.y_daughter2_exit
        return self.total_rate

    def get_signal_prob_decay(self, alp, prod_channel, decay_channel, mask=None):
        """Calculate the probability of decay for a given ALP momentum and distance"""
        if mask is None:
            return (
                self.tau_BRs[prod_channel]
                * alp.alp_BR(decay_channel)
                * alp.prob_decay(self.p4_alp_in_acc["E"], self.L, self.dZ)
            )
        elif mask is not None:
            return (
                self.tau_BRs[prod_channel]
                * alp.alp_BR(decay_channel)
                * alp.prob_decay(self.p4_alp["E"][mask], self.L, self.dZ)
            )

    def reweight(self, alp_old, alp_new):
        """Reweight the event rate for a new ALP decay constant

            For each decay channel, calculate the new rate by summing the flux times the probability of decay for the new ALP

            NOTE: Both ALPs *MUST* have the same mass and have the same 'c_lepton' matrix.

            Here we assume that the tau-->branching ratios are the same.


        Args:
            alp_old (alp.models.ALP): starting ALP
            alp_new (alp.models.ALP): desired ALP

        Returns:
            float: the new event rate
        """
        new_rate = 0.0

        for (
            prod_channel,
            decay_channel,
        ), channel_mask in self.event_splits.items():

            mask_accepted_and_channel = self.mask_alp_in_acc & channel_mask

            weights = self.weights[mask_accepted_and_channel]
            probs = self.get_signal_prob_decay(
                alp_new,
                prod_channel=prod_channel,
                decay_channel=decay_channel,
                mask=mask_accepted_and_channel,
            )
            # Remove NaNs from both arrays
            valid = ~(np.isnan(weights) | np.isnan(probs))
            new_rate += np.sum(weights[valid] * probs[valid])
        return self.norm * new_rate * (alp_old.f_a / alp_new.f_a) ** 2
