import numpy as np
from astropy import units as u
from astropy import constants as const


class TransitModel:
    """
    A class to represent a transit model.

    """

    def __init__(self) -> None:
        pass

    @u.quantity_input
    def calculate_Roche_limit(
        self,
        m_p: u.Quantity[u.Mearth],
        r_p: u.Quantity[u.Rearth],
        m_star: u.Quantity[u.Msun],
        r_star: u.Quantity[u.Rsun],
    ) -> u.Quantity[u.AU]:
        """
        Calculate the Roche limit for the star-planet system.

        Parameters
        ----------
        m_p : u.Quantity[u.Mearth]
            Mass of the planet, in Earth masses.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.

        Returns
        -------
        u.Quantity[u.AU]
            Roche limit of the star-planet system, in astronomical units.
        """

        star_density = m_star / (4 / 3 * np.pi * r_star**3)
        planet_density = m_p / (4 / 3 * np.pi * r_p**3)
        q = planet_density / star_density

        return 2.44 * r_star.to(u.AU) * q ** (1 / 3)

    @u.quantity_input
    def calculate_orbital_period(
        self,
        a: u.Quantity[u.AU],
        m_star: u.Quantity[u.Msun],
    ) -> u.Quantity[u.day]:
        """
        Calculate the orbital period of the planet, following
        Kepler's third law. Planet mass is assumed to be negligible.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.

        Returns
        -------
        u.Quantity[u.day]
            Orbital period of the planet, in days.

        """
        numerator = 4 * np.pi**2 * a**3
        denominator = const.G * m_star  # type: ignore

        return ((numerator / denominator) ** 0.5).to(u.day)

    @u.quantity_input
    def calculate_semimajor_axis(
        self,
        porb: u.Quantity[u.day],
        m_star: u.Quantity[u.Msun],
    ) -> u.Quantity[u.AU]:
        """
        Calculate the semimajor axis of the planet's orbit, following
        Kepler's third law. Planet mass is assumed to be negligible.

        Parameters
        ----------
        porb : u.Quantity[u.day]
            Orbital period of the planet, in days.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.

        Returns
        -------
        u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.

        """
        numerator = porb**2 * const.G * m_star  # type: ignore
        denominator = 4 * np.pi**2

        return ((numerator / denominator) ** (1 / 3)).to(u.AU)

    @u.quantity_input
    def calculate_transit_duration(
        self,
        a: u.Quantity[u.AU],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        m_star: u.Quantity[u.Msun],
        cos_i: float | np.ndarray,
    ) -> u.Quantity[u.hour]:
        """
        Calculate the transit duration for a given set of parameters,
        following Eq. 3 from Seager & Mallén-Ornelas (2003) assuming
        a circular orbit.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        m_star : u.Quantity[u.Msun]
            Mass of the star, in solar masses.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.

        Returns
        -------
        u.Quantity[u.hour]
            Transit duration, in hours.

        """
        porb = self.calculate_orbital_period(a, m_star)

        term1 = 1 + (r_p / r_star).decompose().value
        term2 = (a / r_star).decompose().value * cos_i

        inner_term = np.clip(term1**2 - term2**2, 0, None)
        outer_term = np.where(
            (np.abs(cos_i) < 1),
            (r_star / a).decompose().value * np.sqrt(inner_term / (1 - cos_i**2)),
            0,
        )
        outer_term[outer_term > 1] = 0  # deals with planets that are too close

        t_transit = porb / np.pi * np.arcsin(outer_term)

        return t_transit.to(u.hour)

    @u.quantity_input
    def calculate_transit_depth(
        self,
        a: u.Quantity[u.AU],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
        u1: float | np.ndarray,
        u2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the transit depth for a given set of parameters,
        following Eq. 4 from Heller et al. (2019) using the quadratic
        limb darkening law.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.
        u1 : float | np.ndarray
            First limb darkening coefficient.
        u2 : float | np.ndarray
            Second limb darkening coefficient.

        Returns
        -------
        float | np.ndarray
            Transit depth.
        """

        radius_ratio = (r_p / r_star).decompose()

        return radius_ratio**2 * self.calculate_limb_darkening_correction(
            a, r_star, cos_i, u1, u2
        )

    @u.quantity_input
    def calculate_impact_parameter(
        self,
        a: u.Quantity[u.AU],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the impact parameter for a given set of parameters.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.

        Returns
        -------
        float | np.ndarray
            Impact parameter.
        """

        return (a / r_star).decompose().value * cos_i

    @u.quantity_input
    def calculate_limb_darkening_correction(
        self,
        a: u.Quantity[u.AU],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
        u1: float | np.ndarray,
        u2: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Calculate the limb darkening correction for a given set of parameters,
        following Heller et al. (2019), using the quadratic limb darkening law.
        For parameter values where the planet does not transit the star, the
        correction factor is set to zero.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.
        u1 : float | np.ndarray
            First limb darkening coefficient.
        u2 : float | np.ndarray
            Second limb darkening coefficient.

        Returns
        -------
        float | np.ndarray
            Limb darkening correction factor.

        """
        cos_i = np.asarray(cos_i)
        u1 = np.asarray(u1)
        u2 = np.asarray(u2)

        b = self.calculate_impact_parameter(a, r_star, cos_i)
        b = np.asarray(b)

        mask = b < 1
        corr_factor = np.full(a.shape, 0)

        coeff = np.sqrt(1 - b[mask] ** 2)
        I_p = 1 - u1[mask] * (1 - coeff) - u2[mask] * (1 - coeff) ** 2
        I_A = 1 - u1[mask] / 3 - u2[mask] / 6

        corr_factor[mask] = I_p / I_A
        return corr_factor

    @u.quantity_input
    def is_transiting(
        self,
        a: u.Quantity[u.AU],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
        cos_i: float | np.ndarray,
    ) -> bool | np.ndarray:
        """
        Check if a planet is transiting the star for a given
        set of parameters. This is done by checking if the
        impact parameter is greater than the 1 + (r_p / r_star).
        This is the same as assuring that the inner term of the
        transit duration equation is positive.

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        cos_i : float | np.ndarray
            Cosine of the inclination angle of the planet's orbit.

        Returns
        -------
        bool | np.ndarray
            Boolean indicating if the planet is transiting the star.
        """

        term1 = 1 + (r_p / r_star).decompose().value
        return term1 > self.calculate_impact_parameter(a, r_star, cos_i)

    @u.quantity_input
    def transit_probability(
        self,
        a: u.Quantity[u.AU],
        r_p: u.Quantity[u.Rearth],
        r_star: u.Quantity[u.Rsun],
    ) -> float | np.ndarray:
        """
        Calculate the transit probability for a given set of parameters,
        assuming uniform distribution of cos(i).

        Parameters
        ----------
        a : u.Quantity[u.AU]
            Semimajor axis of the planet's orbit, in astronomical units.
        r_p : u.Quantity[u.Rearth]
            Radius of the planet, in Earth radii.
        r_star : u.Quantity[u.Rsun]
            Radius of the star, in solar radii.

        Returns
        -------
        float | np.ndarray
            Transit probability.
        """

        return ((r_star + r_p) / a).decompose().value  # type: ignore
