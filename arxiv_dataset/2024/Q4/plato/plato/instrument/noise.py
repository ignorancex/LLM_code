import numpy as np


class NoiseModel:
    """
    A class to model the noise in the PLATO mission, following
    the description in Boerner+2022.
    """

    def __init__(
        self,
        n_photoelectrons_ref: int = 177100,
        n_photoelectrons_brightest_pixel_ref: int = 47400,
        n_images: int = 144,
        n_mask: float | np.ndarray = 9.5,
        ccd_readout_noise_factor: float | np.ndarray = 44.3,
        fee_readout_noise_factor: float | np.ndarray = 37.0,
        asd_jitter: float | np.ndarray = 0.54e-6,
        observation_frequency: int = 278,
        signal_spread_factor: float | np.ndarray = 0.27,
        background_rate: float | np.ndarray = 60,
        stray_light_rate: float | np.ndarray = 64,
        contaminating_star_rate: float | np.ndarray = 0.01,
        exposure_time: int = 21,
        smearing_rate: float | np.ndarray = 45,
        charge_transfer_time: float | np.ndarray = 4,
        psf_breathing_noise_factor: float | np.ndarray = 20e-6,
        fee_offset_sensitivity: float | np.ndarray = 1,
        adc_gain: float | np.ndarray = 25,
        fee_temp_stability: float | np.ndarray = 1.0,
        fee_temp_knowledge: float | np.ndarray = 0.01,
        fee_offset_converstion: float | np.ndarray = 0.66,
    ) -> None:
        """
        Initializes the noise model with reference parameters.

        Parameters
        ----------
        n_photoelectrons_ref : int, optional
            Reference number of photoelectrons in the entire mas,
            by default 177100.
        n_photoelectrons_brightest_pixel_ref : int, optional
            Reference number of photoelectrons in the brightest pixel,
            by default 47400.
        n_images : int,
            Number of images taken within 1 hour,
            by default 144 (cadence of 25s, with 21s
            exposure time and 4s signal transfer time).
        n_mask : float | np.ndarray, optional
            Effective number of pixels in the mask,
            by default 9.5.
        ccd_readout_noise_factor : float | np.ndarray, optional
            CCD readout noise factor, by default 44.3.
        fee_readout_noise_factor : float | np.ndarray, optional
            FEE readout noise factor, by default 37.0.
        asd_jitter : float | np.ndarray, optional
            ASD (amplitude spectral density) jitter factor, by default 0.54e-6.
        observation_frequency : int, optional
            Observation frequency in microHz, by default 278.
        signal_spread_factor : float | np.ndarray, optional
            Fraction of energy in the brightest pixel,
            by defaults 0.27.
        background_rate : float | np.ndarray, optional
            Background photoelectron rate per pixel per second,
            by default 60.
        stray_light_rate : float | np.ndarray, optional
            Stray light photoelectron rate per pixel per second,
            by default 64.
        contaminating_star_rate : float | np.ndarray, optional
            Contaminating star signal as a fraction of target star signal,
            by default 0.01.
        exposure_time : int, optional
            Exposure time in seconds, by default 21.
        smearing_rate : float | np.ndarray, optional
            Median photoelectron rate for smearing noise,
            by default 45.
        charge_transfer_time : float | np.ndarray, optional
            Charge transfer time in seconds, by default 4.
        psf_breathing_noise_factor : float | np.ndarray, optional
            PSF breathing noise factor in ppm on camera level,
            by default 20e-6.
        fee_offset_sensitivity : float | np.ndarray, optional
            FEE offset sensitivity to temperature in ADU/K,
            by default 1.
        adc_gain : float | np.ndarray, optional
            ADC gain factor, by default 25.
        fee_temp_stability : float | np.ndarray, optional
            FEE temperature stability in K, by default 1.0.
        fee_temp_knowledge : float | np.ndarray, optional
            FEE temperature knowledge in K, by default 0.01.
        fee_offset_converstion : float | np.ndarray, optional
            FEE offset conversion factor, by default 0.66.
        """
        self.n_photoelectrons_ref = n_photoelectrons_ref
        self.n_photoelectrons_brightest_pixel_ref = n_photoelectrons_brightest_pixel_ref
        self.n_images = n_images
        self.n_mask = n_mask
        self.ccd_readout_noise_factor = ccd_readout_noise_factor
        self.fee_readout_noise_factor = fee_readout_noise_factor
        self.asd_jitter = asd_jitter
        self.observation_frequency = observation_frequency
        self.signal_spread_factor = signal_spread_factor
        self.background_rate = background_rate
        self.stray_light_rate = stray_light_rate
        self.contaminating_star_rate = contaminating_star_rate
        self.exposure_time = exposure_time
        self.smearing_rate = smearing_rate
        self.charge_transfer_time = charge_transfer_time
        self.psf_breathing_noise_factor = psf_breathing_noise_factor
        self.fee_offset_sensitivity = fee_offset_sensitivity
        self.adc_gain = adc_gain
        self.fee_temp_stability = fee_temp_stability
        self.fee_temp_knowledge = fee_temp_knowledge
        self.fee_offset_converstion = fee_offset_converstion

    def calculate_flux(
        self,
        magnitude_v: float | np.ndarray,
        a: float | np.ndarray = -0.02393861,
        b: float | np.ndarray = 1.06645729,
    ) -> float | np.ndarray:
        """
        Calculates the flux of a star given its magnitude.
        The fluxes are calculated using the following formula:
        flux = 10^(a - 0.4 * b * (magnitude_v - 11)),
        where a and b are correction factors so that the flux
        is the same as as the ones given in Boerner+2022 (
        calculated from the values of the photon noise
        given in table 5-2 and the equation 3-13)

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        a : float | np.ndarray, optional
            Normalisation correction factor,
            by default -0.02393861.
        b : float | np.ndarray, optional
            Slope correction factor, by
            default 1.06645729.

        Returns
        -------
        float | np.ndarray
            The flux of the star.
        """
        log_flux = a - 0.4 * b * (magnitude_v - 11)
        return np.power(10, log_flux)

    def calculate_n_photoelectrons(
        self,
        magnitude_v: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the number of photoelectrons in the entire mask, based
        on the V band magnitude of the star. (Very approximate, WIP)

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.

        Returns
        -------
        float | np.ndarray
            The number of photoelectrons in the entire mask.
        """
        return self.calculate_flux(magnitude_v) * self.n_photoelectrons_ref

    def calculate_n_photoelectrons_brightest_pixel(
        self,
        magnitude_v: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the number of photoelectrons in the brightest pixel, based
        on the V band magnitude of the star. (Very approximate, WIP)

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.

        Returns
        -------
        float | np.ndarray
            The number of photoelectrons in the brightest pixel.
        """
        return (
            self.calculate_flux(magnitude_v) * self.n_photoelectrons_brightest_pixel_ref
        )

    def ccd_readout_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the CCD readout noise for a star given its magnitude.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The CCD readout noise.
        """
        return (
            self.ccd_readout_noise_factor
            / self.calculate_n_photoelectrons(magnitude_v)
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def fee_readout_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the FEE readout noise for a star given its magnitude.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The FEE readout noise.
        """
        return (
            self.fee_readout_noise_factor
            / self.calculate_n_photoelectrons(magnitude_v)
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def jitter_noise(self) -> float | np.ndarray:
        """
        Calculates the jitter noise.

        Returns
        -------
        float | np.ndarray
            The jitter noise.
        """
        return self.asd_jitter * np.sqrt(self.observation_frequency)

    def misc_photon_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the miscellaneous photon noise, consistent of
        background noise, stray light noise, and contaminating star noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The miscellaneous photon noise.
        """
        background_noise = self.background_rate * self.exposure_time
        stray_light_noise = self.stray_light_rate * self.exposure_time
        contaminating_star_noise = (
            self.contaminating_star_rate
            * self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
        )
        return (
            np.sqrt(background_noise + stray_light_noise + contaminating_star_noise)
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def ccd_smearing_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the CCD smearing noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The CCD smearing noise.
        """
        return (
            np.sqrt(self.smearing_rate * self.charge_transfer_time)
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * np.sqrt(self.n_mask / n_cameras / self.n_images)
        )

    def psf_breathing_noise(
        self,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the PSF breathing noise.

        Parameters
        ----------
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The PSF breathing noise.
        """
        return self.psf_breathing_noise_factor / np.sqrt(n_cameras * self.n_images)

    def fee_offset_stability_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the FEE offset stability noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The FEE offset stability noise.
        """
        return (
            self.fee_offset_sensitivity
            * self.adc_gain
            * self.fee_temp_knowledge
            / self.calculate_n_photoelectrons_brightest_pixel(magnitude_v)
            * self.signal_spread_factor
            * self.fee_offset_converstion
            / np.sqrt(n_cameras * self.n_images)
        )

    def photon_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the photon noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The photon noise.
        """
        return 1 / np.sqrt(
            self.calculate_n_photoelectrons(magnitude_v) * n_cameras * self.n_images
        )

    def random_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the random noise, which is the (Pythagorean) sum of the CCD readout
        noise, FEE readout noise, miscellaneous photon noise, and CCD
        smearing noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The random noise.
        """
        return np.sqrt(
            self.ccd_readout_noise(magnitude_v, n_cameras) ** 2
            + self.fee_readout_noise(magnitude_v, n_cameras) ** 2
            + self.misc_photon_noise(magnitude_v, n_cameras) ** 2
            + self.ccd_smearing_noise(magnitude_v, n_cameras) ** 2
        )

    def systematic_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculates the systematic noise, which is the (Pythagorean) sum of the jitter
        noise, PSF breathing noise, and FEE offset stability noise.

        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.

        Returns
        -------
        float | np.ndarray
            The systematic noise.
        """
        return np.sqrt(
            self.jitter_noise() ** 2
            + self.psf_breathing_noise(n_cameras) ** 2
            + self.fee_offset_stability_noise(magnitude_v, n_cameras) ** 2
        )

    def calculate_noise(
        self,
        magnitude_v: float | np.ndarray,
        n_cameras: int | np.ndarray,
        stellar_variability: float | np.ndarray = 0,
    ) -> float | np.ndarray:
        """
        Calculate the relative lightcurve noise.
        The noise consists of photometric precision (which in turn consists of
        photon noise, random noise, and systematic noise), and the
        stellar variability.
        The photometric precision is relative noise integrated over one hour. The
        value depends on the magnitude of the star and the number of cameras
        observing the star.
        The stellar variability is the standard deviation of the variability
        of the star on timescales of ~1hr, assumed to be white noise. Correlations
        and trends are not considered.


        Parameters
        ----------
        magnitude_v : float | np.ndarray
            The apparent V magnitude of the star.
        n_cameras : int
            The number of cameras observing the star.
        stellar_variability : float | np.ndarray, optional
            The standard deviation of the stellar variability, by default 0.

        Returns
        -------
        float | np.ndarray
            The noise-to-signal ratio.
        """

        photon_noise = self.photon_noise(magnitude_v, n_cameras)
        random_noise = self.random_noise(magnitude_v, n_cameras)
        systematic_noise = self.systematic_noise(magnitude_v, n_cameras)
        photometric_precision_squared = (
            photon_noise**2 + random_noise**2 + systematic_noise**2
        )

        return np.sqrt(photometric_precision_squared + stellar_variability**2)
