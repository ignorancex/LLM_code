from typing import Optional, Callable

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.stats import gamma

from plato.instrument.noise import NoiseModel
from plato.planets.transit import TransitModel


class DetectionModel:
    """
    A class to calculate the signal-to-noise ratio (SNR) for
    a transiting exoplanet, as described in Matuszewski+2023
    (Section 4.2).
    """

    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        transit_model: Optional[TransitModel] = None,
    ) -> None:
        """
        Initializes the SNR model with a NoiseModel and
        a TransitModel.

        Parameters
        ----------
        noise_model : Optional[CDPPModel], optional
            A NoiseModel instance, by default None, in which
            case a new NoiseModel instance will be created with
            default parameters.
        transit_model : Optional[TransitModel], optional
            A TransitModel instance, by default None, in which
            case a new TransitModel instance will be created with
            default parameters.
        """
        self.noise_model = NoiseModel() if noise_model is None else noise_model
        self.transit_model = TransitModel() if transit_model is None else transit_model

    @u.quantity_input
    def calculate_extra_transit_probability(
        self,
        t_mission: u.year,
        porb: u.day,
    ) -> float | np.ndarray:
        """
        Calculates the probability an additional transit
        during the mission.

        Parameters
        ----------
        t_mission : u.year
            The duration of the mission.
        porb : u.day
            The orbital period of the planet.

        Returns
        -------
        float | np.ndarray
            The probability
            of having N+1 transits.

        """
        remainder = t_mission % porb  # additional time after N transits
        return (remainder / porb).decompose().value  # probability of extra transits

    def calculate_snr_parameter(
        self,
        data: pd.DataFrame,
        calculate_non_transiting: bool = False,
        fill_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the transit depth, transit duration and
        lightcurve noise for a given set of parameters. Parameters
        are given in a DataFrame.
        Non-transiting targets are assigned a transit depth and
        lightcurve noise of 0, unless calculate_non_transiting is True.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the following columns:
            - "R_planet": the radius of the planet, in Earth radii.
            - "P_orb": the orbital period of the planet, in days.
            - "R_star": the radius of the star, in solar radii.
            - "M_star": the mass of the star, in solar masses.
            - "Magnitude_V": the apparent magnitude of the star.
            - "sigma_star": the standard deviation of the
                variability of the star.
            - "cos_i": the cosine of the inclination angle of the
                    star's planetary system.
            - "u1": the first limb darkening coefficient.
            - "u2": the second limb darkening coefficient.
            - "n_cameras": the number of cameras observing the star.
        calculate_non_transiting : bool, optional
            If True, calculate the values for non-transiting targets as well,
            by default False.
        fill_value : float, optional
            Value to fill non-transiting targets with, by default 0.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the transit depth, transit duration
            and lightcurve noise. Transit duration is given in hours.
        """

        # add semimajor axis column if not present (to copy of df, so
        # original is not modified)
        if "a" not in data.columns:
            data = data.copy()
            data["a"] = (
                self.transit_model.calculate_semimajor_axis(
                    porb=data["P_orb"].to_numpy() * u.day,
                    m_star=data["M_star"].to_numpy() * u.Msun,
                )
                .to(u.AU)
                .value
            )

        # calculate mask for transiting targets
        if calculate_non_transiting:
            transiting_mask = np.full(len(data), True)
        else:
            transiting_mask = self.transit_model.is_transiting(
                a=data["a"].to_numpy() * u.AU,
                r_p=data["R_planet"].to_numpy() * u.Rearth,
                r_star=data["R_star"].to_numpy() * u.Rsun,
                cos_i=data["cos_i"].to_numpy(),
            )
        data_masked = data[transiting_mask]

        # initialize arrays
        transit_duration = np.full(len(data), float(fill_value)) * u.hour
        transit_depth = np.full(len(data), float(fill_value))
        lightcurve_noise = np.full(len(data), float(fill_value))

        # calculate transit depth, duration and lightcurve noise for targets
        transit_duration_masked = self.transit_model.calculate_transit_duration(
            a=data_masked["a"].to_numpy() * u.AU,
            r_p=data_masked["R_planet"].to_numpy() * u.Rearth,
            r_star=data_masked["R_star"].to_numpy() * u.Rsun,
            m_star=data_masked["M_star"].to_numpy() * u.Msun,
            cos_i=data_masked["cos_i"].to_numpy(),
        )
        transit_depth_masked = self.transit_model.calculate_transit_depth(
            a=data_masked["a"].to_numpy() * u.AU,
            r_p=data_masked["R_planet"].to_numpy() * u.Rearth,
            r_star=data_masked["R_star"].to_numpy() * u.Rsun,
            cos_i=data_masked["cos_i"].to_numpy(),
            u1=data_masked["u1"].to_numpy(),
            u2=data_masked["u2"].to_numpy(),
        )
        lightcurve_noise_masked = self.noise_model.calculate_noise(
            magnitude_v=data_masked["Magnitude_V"].to_numpy(),
            n_cameras=data_masked["n_cameras"].to_numpy(),
            stellar_variability=data_masked["sigma_star"].to_numpy(),
        )

        # fill arrays with calculated values
        transit_duration[transiting_mask] = transit_duration_masked
        transit_depth[transiting_mask] = transit_depth_masked
        lightcurve_noise[transiting_mask] = lightcurve_noise_masked

        return (
            np.array(transit_depth),
            np.array(transit_duration.to(u.hour).value),
            np.array(lightcurve_noise),
        )

    @u.quantity_input
    def calculate_snr(
        self,
        transit_depth: float | np.ndarray,
        transit_duration: float | np.ndarray,
        lightcurve_noise: float | np.ndarray,
        p_orb: u.Quantity[u.day],
        t_mission: u.year = 2 * u.year,
        extra_transit: bool = False,
    ) -> float | np.ndarray:
        """
        Calculates the signal-to-noise ratio (SNR) for a
        transiting exoplanet.

        Parameters
        ----------
        transit_depth : float | np.ndarray
            The transit depth.
        transit_duration : float | np.ndarray
            The transit duration, in hours.
        lightcurve_noise : float | np.ndarray
            The lightcurve noise.
        p_orb : u.Quantity[u.day]
            The orbital period of the planet.
        t_mission : u.year, optional
            The duration of the mission, by default 2 years.
        extra_transit : bool, optional
            If True, an additional transit is assumed to be observed,
            by default False.

        Returns
        -------
        float | np.ndarray
            The signal-to-noise ratio (SNR).

        """
        transit_depth = np.asarray(transit_depth)
        transit_duration = np.asarray(transit_duration)
        lightcurve_noise = np.asarray(lightcurve_noise)

        N_transits = t_mission.to(u.year).value // p_orb.to(u.year).value
        N_transits = N_transits + 1 if extra_transit else N_transits

        # find trasiting targets
        mask = transit_duration > 0
        snr = np.zeros_like(transit_depth)
        snr[mask] = (
            transit_depth[mask]
            / (lightcurve_noise[mask])
            * np.sqrt(N_transits[mask] * transit_duration[mask])
        )
        return snr

    @staticmethod
    def linear_detection_efficiency(
        snr: float | np.ndarray | np.ndarray,
        lower_threshold: float | np.ndarray = 6,
        upper_threshold: float | np.ndarray = 10,
    ) -> float | np.ndarray | np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR) using a linear model.
        The efficiency is a linear function of the SNR, with a lower
        threshold below which the efficiency is 0 and an upper threshold
        above which the efficiency is 1, from Fressin+2013.

        Parameters
        ----------
        snr : float | np.ndarray
            The signal-to-noise ratio.
        lower_threshold : float | np.ndarray, optional
            The lower threshold below which the detection
            efficiency is 0, by default 6 (Fressin+2013).
        upper_threshold : float | np.ndarray, optional
            The upper threshold above which the detection
            efficiency is 1, by default 10 (taken from the
            Plato Red book, which assumes a 100% efficiency
            for an snr>10).

        Returns
        -------
        float | np.ndarray
            The detection efficiency as a function of the SNR.
        """
        if snr < lower_threshold:
            return 0.0
        elif snr > upper_threshold:
            return 1.0
        else:
            return (snr - lower_threshold) / (upper_threshold - lower_threshold)

    @staticmethod
    def gamma_detection_efficiency(
        snr: float | np.ndarray | np.ndarray,
        gamma_a: float | np.ndarray = 30.87,
        gamma_b: float | np.ndarray = 0.271,
        gamma_c: float | np.ndarray = 0.940,
    ) -> float | np.ndarray | np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR) using a gamma distribution model.
        The efficiency is a gamma cumulative distribution function (CDF) of the SNR,
        from Christiansen+2016, Christiansen+2017.

        Parameters
        ----------
        snr : float | np.ndarray | np.ndarray
            The signal-to-noise ratio.
        gamma_a : float | np.ndarray, optional
            The shape parameter of the gamma distribution,
            by default 30.87 (Christiansen+2017).
        gamma_b : float | np.ndarray, optional
            The scale parameter of the
            gamma distribution, by default 0.271 (Christiansen+2017).
        gamma_c : float | np.ndarray, optional
            The overall normalisation parameter of the
            gamma distribution, by default 0.940 (Christiansen+2017).


        Returns
        -------
        float | np.ndarray | np.ndarray
            The detection efficiency as a function of the SNR.
        """
        return gamma_c * gamma.cdf(snr, a=gamma_a, loc=0, scale=gamma_b)

    @u.quantity_input
    def detection_efficiency(
        self,
        data: pd.DataFrame,
        *,
        t_mission: u.year = 2 * u.year,
        min_transits: int = 2,
        mode: str = "gamma",
        kwargs_detection_efficiency: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Calculates the detection efficiency as a function
        of the signal-to-noise ratio (SNR). Two models are
        implemented:
        - A linear model, where the efficiency is a linear function
        of the SNR, with a lower threshold below which the efficiency
        is 0 and an upper threshold above which the efficiency is 1,
        from Fressin+2013.
        - A gamma model, where the efficiency is a gamma cumulative
        distribution function (CDF) of the SNR, from
        Christiansen+2016, Christiansen+2017.

        The detection efficiency is calculated for N and N+1 transits,
        and the final efficiency is a weighted average of the two
        efficiencies, where the weight is the probability of observing
        an extra transits. If the number of transits is below a certain
        threshold (min_transits), the efficiency is set to 0.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the following columns:
                - "R_planet": the radius of the planet, in Earth radii.
                - "R_star": the radius of the star, in solar radii.
                - "M_star": the mass of the star, in solar masses.
                - "Magnitude_V": the apparent magnitude of the star.
                - "sigma_star": the standard deviation of the
                    variability of the star.
                - "cos_i": the cosine of the inclination angle of the
                        star's planetary system.
                - "u1": the first limb darkening coefficient.
                - "u2": the second limb darkening coefficient.
                - "n_cameras": the number of cameras observing the star.
            Additionally, either
                - "a": the semimajor axis of the planet's orbit, in AU, or
                - "P_orb": the orbital period of the planet, in days.
            must be present.
        t_mission : u.year, optional
            The duration of the mission, by default 2 years.
        min_transits : int, optional
            The minimum number of transits required to detect planet, by default 2.
            If the number of transits is below this threshold, the efficiency is
            set to 0.
        mode : str, optional
            The mode to calculate the detection efficiency, either
            "linear" or "gamma", by default "linear".
        kwargs_detection_efficiency : Optional[dict], optional
            Additional keyword arguments for the detection efficiency model,
            by default None.
        Returns
        -------
        np.ndarray
            The detection efficiency for the given parameters.

        Raises
        ------
        ValueError
            Raised if mode is not "linear" or "gamma".
        """
        if kwargs_detection_efficiency is None:
            kwargs_detection_efficiency = {}

        # check if necessary columns are present in data
        required_columns = [
            "R_planet",
            "R_star",
            "M_star",
            "Magnitude_V",
            "sigma_star",
            "cos_i",
            "u1",
            "u2",
            "n_cameras",
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing column {col!r} in data.")

        # check if either a or P_orb is present in data, and calculate the other
        if "a" not in data.columns and "P_orb" not in data.columns:
            raise ValueError("Either 'a' or 'P_orb' must be present in data.")

        elif "P_orb" in data.columns and "a" not in data.columns:
            if "M_star" not in data.columns:
                raise ValueError(
                    "Column 'M_star' needed to calculate 'a' from 'P_orb'."
                )
            data = data.copy()
            data["a"] = (
                self.transit_model.calculate_semimajor_axis(
                    porb=data["P_orb"].to_numpy() * u.day,
                    m_star=data["M_star"].to_numpy() * u.Msun,
                )
                .to(u.AU)
                .value
            )

        elif "a" in data.columns and "P_orb" not in data.columns:
            if "M_star" not in data.columns:
                raise ValueError(
                    "Column 'M_star' needed to calculate 'P_orb' from 'a'."
                )
            data = data.copy()
            data["P_orb"] = (
                self.transit_model.calculate_orbital_period(
                    a=data["a"].to_numpy() * u.AU,
                    m_star=data["M_star"].to_numpy() * u.Msun,
                )
                .to(u.day)
                .value
            )
        else:
            raise NotImplementedError(
                "Both 'a' and 'P_orb' are present in data. "
                "This is not currently considered."
            )

        # calculate SNR for N and N+1 transits
        transit_depth, transit_duration, lightcurve_noise = (
            self.calculate_snr_parameter(data)
        )
        snr_values = [
            self.calculate_snr(
                transit_depth,
                transit_duration,
                lightcurve_noise,
                data["P_orb"].to_numpy() * u.day,
                t_mission,
                extra_transit=x,
            )
            for x in [False, True]
        ]

        # Calculate detection efficiencies for N and N+1 transits, depending on
        # detection efficiency model
        if mode == "linear":
            detection_efficiency_model: Callable = self.linear_detection_efficiency
        elif mode == "gamma":
            detection_efficiency_model = self.gamma_detection_efficiency
        else:
            raise ValueError("Invalid mode. Choose 'linear' or 'gamma.")

        # calculate detection efficiency for N and N+1 transits
        detection_eff: list | np.ndarray = [
            detection_efficiency_model(
                snr,
                **kwargs_detection_efficiency,
            )
            for snr in snr_values
        ]

        # calculate number of certain transits and probability of N+1 transits
        N_transits = (
            t_mission.to(u.year).value
            // (data["P_orb"].to_numpy() * u.day).to(u.year).value
        )
        p_extra = self.calculate_extra_transit_probability(
            t_mission, data["P_orb"].to_numpy() * u.day
        )

        # set detection efficiency to 0 if number of transits is below threshold,
        # check for N and N+1 transits
        detection_eff = np.array(
            [
                np.where(N_transits + i >= min_transits, de, 0)
                for i, de in zip([0, 1], detection_eff)
            ]
        )

        # Return weighted average of detection efficiencies
        return np.average(detection_eff, weights=[1 - p_extra, p_extra], axis=0)
