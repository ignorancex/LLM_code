from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.stats import norm

from plato.instrument.detection import DetectionModel
from plato.utils.paths import get_abspath


class PopulationModel:
    """
    A class to represent the NGPPS planet populations.
    """

    def __init__(
        self,
        stellar_population: pd.DataFrame,
        num_embryos: int,
        snapshot_age: int = int(1e8),
        detection_model: Optional[DetectionModel] = None,
        additional_columns: Optional[str | list[str]] = None,
        keep_all_columns: bool = False,
        keep_invalid: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the planet population model.

        Parameters
        ----------
        stellar_population : pd.DataFrame
            A dataframe containing the properties of the stellar
            population. Must contain  the columns
                - "R_star": the radius of the star, in solar radii.
                - "M_star": the mass of the star, in solar masses.
                - "[Fe/H]": the metallicity of the star.
                - "Magnitude_V": the apparent magnitude of the star.
                - "sigma_star": the standard deviation of the
                                variability of the star.
                - "u1": the first limb darkening coefficient.
                - "u2": the second limb darkening coefficient.
                - "n_cameras": the number of cameras observing the star.
            Additionally, the column "gaiaID_DR3" will be used if present.
        num_embryos : int
            The number of planetary embryos in the NGPPS run, used to
            select the appropriate NGPPS snapshot. Can be 10, 20, 50,
            or 100.
        snapshot_age : int, optional
            The age of the planetary system in the NGPPS run, used to select
            the appropriate NGPPS snapshot, by default 100Myr.
        detection_model : Optional[DetectionModel], optional
            A detection model to use for the mock observations, by default None.
            If None, detection model with default parameters is used.
        additional_columns : Optional[str | list[str]], optional
            A list of additional columns to include in the stellar population
            DataFrame, by default None.
        keep_columns : bool, optional
            If True, all columns in the stellar and NGPPS population DataFrames are
            kept. Otherwise, only the required columns are kept, by default False.
        keep_invalid : bool, optional
            If True, invalid planetary systems are kept in the NGPPS population.
            Otherwise, invalid systems are removed, by default False.
        **kwargs : Any
            Additional keyword arguments to pass to the get_system_metallicities
            method.

        """

        self.detection_model = detection_model if detection_model else DetectionModel()

        # create stellar population attribute
        stellar_population_columns = [
            "R_star",
            "M_star",
            "[Fe/H]",
            "Magnitude_V",
            "sigma_star",
            "u1",
            "u2",
            "n_cameras",
        ]
        if additional_columns:
            additional_columns = (
                [additional_columns]
                if isinstance(additional_columns, str)
                else additional_columns
            )
            stellar_population_columns += additional_columns

        if "gaiaID_DR3" in stellar_population.columns:
            stellar_population_columns += ["gaiaID_DR3"]

        for col in stellar_population_columns:
            if col not in stellar_population.columns:
                raise ValueError(
                    f"Stellar population DataFrame must contain the column '{col}'."
                )
        if not keep_all_columns:
            stellar_population = stellar_population[stellar_population_columns]

        self.stellar_population = stellar_population

        # create planetary system population attribute
        ngpps_mapping = {
            10: "ng96",
            20: "ng74",
            50: "ng75",
            100: "ng76",
        }

        ngpps_population = pd.read_csv(
            get_abspath() + "data/external/NGPPS/"
            f"{ngpps_mapping[num_embryos]}/"
            f"snapshot_{snapshot_age}.csv"
        )

        if not keep_invalid:
            ngpps_population = ngpps_population[ngpps_population["valid"]]

        if not keep_all_columns:
            ngpps_population = ngpps_population[
                [
                    "system_id",
                    "planet_id",
                    "total_radius",
                    "total_mass",
                    "semi_major_axis",
                ]
            ]

        ngpps_population = ngpps_population.rename(
            columns={
                "total_radius": "R_planet",
                "total_mass": "M_planet",
                "semi_major_axis": "a",
            }
        )
        ngpps_population["R_planet"] = (
            ngpps_population["R_planet"].to_numpy() * u.Rjup
        ).to(u.Rearth)

        self.ngpps_population = ngpps_population

        self.system_populations, self.system_num_planets = self.group_systems()

        # create planetary system metallicity attribute(s)
        system_metallicity = self.get_system_metallicities(**kwargs)
        self.system_metallicity = system_metallicity.loc[
            self.ngpps_population["system_id"].unique() - 1
        ]  # only use the ones in ngpps_population

        # placeholder attributes for probability caching
        self.log_probs: Optional[np.ndarray] = None
        self.log_cum_probs: Optional[np.ndarray] = None
        self.decay_parameter_check: Optional[float] = None
        self.correct_for_initial_check: Optional[bool] = None

    def group_systems(
        self,
    ) -> tuple[list[pd.DataFrame], list[int]]:
        """
        Group the planetary systems in the NGPPS population by
        system_id. Returns a list of DataFrames, where each
        DataFrame contains the planets in a single planetary
        system. Also returns a list of the number of planets
        in each system.

        Returns
        -------
        tuple[list[pd.DataFrame], list[int]]
            A tuple containing the list of DataFrames, and the
            list of the number of planets in each system.
        """

        grouping = self.ngpps_population.groupby("system_id")
        groups = [grouping.get_group(name) for name in grouping.groups]
        len_groups = [len(group) for group in groups]

        return groups, len_groups

    def get_system_metallicities(
        self,
        solar_gas_to_dust: float = 0.0149,
        get_all_columns: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve planetary system metallicities from the NGPPS
        metadata.

        Parameters
        ----------
        solar_gas_to_dust : float, optional
            The solar gas-to-dust ratio, used to convert the
            gas-to-dust ratio of the planetary system to a
            metallicity, by default 0.0149.
        get_all_columns : bool, optional
            Whether to return all columns in the NGPPS metadata
            DataFrame, or just the 'system_id' and '[Fe/H]'. By
            default False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the system_id and metallicity
            of each planetary system in the NGPPS. If get_all_columns
            is True, all metadata columns are returned.
        """

        # read planetary system metadata
        metadata = pd.read_csv(
            get_abspath() + "data/external/NGPPS/NGPPS_variables.txt",
            header=None,
            sep=r"\s+",
            names=[
                "system_id",
                "mstar",
                "sigma",
                "expo",
                "ain",
                "aout",
                "fpg",
                "mwind",
            ],
        )

        # modify the 'system_id' column to remove 'system_id' prefix
        metadata["system_id"] = metadata["system_id"].str[3:].astype(int)

        # convert columns to float
        for col in metadata.columns[1:]:
            metadata[col] = metadata[col].map(lambda x: float(x.split("=")[1]))

        # calculate the metallicity of the planetary systems
        metadata["[Fe/H]"] = np.log10(
            metadata["fpg"] / solar_gas_to_dust
        )  # NGPPS II paper Eq. 2

        if get_all_columns:
            return metadata
        else:
            return metadata[["system_id", "[Fe/H]"]]

    def update_stellar_variability(self, sigma_star: float | np.ndarray) -> None:
        """
        Update the standard deviation of the variability of the
        stars in the stellar population.

        Parameters
        ----------
        sigma_star : float | np.ndarray
            The new standard deviation of the variability of the
            stars in the stellar population. If a float, the same
            value is used for all stars. If an array, the values
            are used for each star in the stellar population. The array
            must have the same length as the stellar population.
        """
        stellar_population = self.stellar_population.copy()
        stellar_population["sigma_star"] = sigma_star
        self.stellar_population = stellar_population

    def calculate_log_probabilities(
        self,
        decay_parameter: float = 10 * np.log(10),
        correct_for_initial_distribution: bool = True,
        return_cumulative_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Calculate the log probability (to base e) system assigment probabilities
        for each star in the stellar population. The probability distribution are
        calculated for all stars in the stellar population, based on the
        metallicities of the star and NGPPS systems.

        The probabilties are calculated using an exponential distribution, as
        log_e p = -decay_parameter * |[Fe/H]_star - [Fe/H]_system|, where
        decay_parameter controls the rate at which the probability decreases
        with metallicity difference.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star. With a value of 10 * log_e(10), the
            probability of assigning a system drops by a factor of
            10 for each 0.1 dex difference in metallicity.
        correct_for_initial_distribution : bool, optional
            Whether to correct the probabilities for the initial
            metallicity distribution of the NGPPS systems, which
            is given by a normal distribution with mean -0.02 and
            standard deviation 0.22 (NGPPS paper 2), by default True.
        return_cumulative_probabilities : bool, optional
            If True, the cumulative probabilities are returned, by
            default False.

        Returns
        -------
        np.ndarray
            A 2D array containing the log probabilities (to base e) of
            assigningany system to any star in the stellar population. If
            return_cumulative_probabilities is True, the log cumulative
            probabilities are returned instead.

        """

        # Efficiently compute the probability matrix using broadcasting
        diff_matrix = (
            self.stellar_population["[Fe/H]"].to_numpy(dtype=np.float32)[:, None]
            - self.system_metallicity["[Fe/H]"].to_numpy(dtype=np.float32)[None, :]
        )

        log_probabilities = -decay_parameter * np.abs(diff_matrix)

        if correct_for_initial_distribution:
            original_distribution = norm(loc=-0.02, scale=0.22)
            log_probabilities -= original_distribution.logpdf(  # type: ignore
                self.system_metallicity["[Fe/H]"]
            )

        # normalise the probabilities
        log_probabilities -= np.logaddexp.reduce(
            log_probabilities, axis=1, keepdims=True
        )

        if return_cumulative_probabilities:
            cumulative_log_probabilities = np.ufunc.accumulate(
                np.logaddexp,
                log_probabilities,
                axis=1,
            )
            return cumulative_log_probabilities
        return log_probabilities

    def assign_random_systems(
        self,
        decay_parameter: float = 10 * np.log(10),
        correct_for_initial_distribution: bool = True,
        get_random_indices: bool = False,
    ) -> np.ndarray:
        """
        Assign a random planetary system to each star in the stellar
        population, based on the metallicity distribution of the
        NGPPS systems. The log probability of a system being assigned
        to a star is proportional to the absolute difference in
        metallicity between the system and the star.

        The `decay_parameter` controls the rate at which the
        probability of assigning a system to a star decreases with
        the metallicity difference. A higher value results in a faster
        decrease in probability with metallicity difference.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star. With a value of 10 * log_e(10), the
            probability of assigning a system drops by a factor of
            10 for each 0.1 dex difference in metallicity.
        correct_for_initial_distribution : bool, optional
            Whether to correct the probabilities for the initial
            metallicity distribution of the NGPPS systems, which
            is given by a normal distribution with mean -0.02 and
            standard deviation 0.22 (NGPPS paper 2), by default True.
        get_random_indices : bool, optional
            If True, the random indices used for assigning the systems are
            returned instead of the system ids, by default False.

        Returns
        -------
        np.ndarray
            A random sample of system ids assigned to each star in
            the stellar population.
        """
        # calculate the log probabilities with caching
        if (
            self.log_cum_probs is None
            and self.decay_parameter_check != decay_parameter
            and self.correct_for_initial_check != correct_for_initial_distribution
        ):
            log_cumulative_probabilities = self.calculate_log_probabilities(
                decay_parameter=decay_parameter,
                correct_for_initial_distribution=correct_for_initial_distribution,
                return_cumulative_probabilities=True,
            )  # type: ignore
            self.log_cum_probs = log_cumulative_probabilities
            self.decay_parameter_check = decay_parameter
            self.correct_for_initial_check = correct_for_initial_distribution
        else:
            assert isinstance(self.log_cum_probs, np.ndarray)
            log_cumulative_probabilities = self.log_cum_probs

        # generate random values for each row
        log_random_values = np.log(
            np.random.rand(log_cumulative_probabilities.shape[0], 1)
        )

        # use broadcasting to compare random values with cumulative probabilities
        random_indices = (log_random_values < log_cumulative_probabilities).argmax(
            axis=1
        )

        if get_random_indices:
            return random_indices

        # get corresponding system ids
        system_ids = self.ngpps_population["system_id"][random_indices].to_numpy()
        return system_ids

    def add_planet_category(
        self,
        dataframe: pd.DataFrame,
        category_dict: Optional[dict[str, Callable]] = None,
    ) -> pd.DataFrame:
        """
        Add a column to the DataFrame containing the planet population
        with the category of each planet, based on the mass of the
        planet. The categories are defined by the category_dict
        argument, which is a dictionary mapping the category name
        to a function that takes a row of the DataFrame and returns
        a boolean value indicating whether the planet belongs to
        that category.

        The default categories are:
            - Dwarf: M_planet < 0.5
            - Earth: 0.5 <= M_planet < 2
            - Super-Earth: 2 <= M_planet < 10
            - Neptunian: 10 <= M_planet < 30
            - Sub-Giant: 30 <= M_planet < 300
            - Giant: 300 <= M_planet

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame containing the planet population,
            must contain the 'M_planet' column.
        category_dict : , optional
            A dictionary mapping the category name to a function
            that takes a row of the DataFrame and returns a boolean
            value indicating whether the planet belongs to that
            category. If categories are overlapping, the first
            category that returns True is assigned to the planet,
            by default None. If None, the default categories
            are used.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the planet population, with
            the 'Planet Category' column added. If no category is
            assigned, the category is set to 'Unknown'.

        """
        if category_dict is None:
            conditions = [
                dataframe["M_planet"] < 0.5,
                (dataframe["M_planet"] >= 0.5) & (dataframe["M_planet"] < 2),
                (dataframe["M_planet"] >= 2) & (dataframe["M_planet"] < 10),
                (dataframe["M_planet"] >= 10) & (dataframe["M_planet"] < 30),
                (dataframe["M_planet"] >= 30) & (dataframe["M_planet"] < 300),
                dataframe["M_planet"] >= 300,
            ]
            categories = [
                "Dwarf",
                "Earth",
                "Super-Earth",
                "Neptunian",
                "Sub-Giant",
                "Giant",
            ]
        else:
            conditions = [
                dataframe.apply(condition, axis=1)
                for condition in category_dict.values()
            ]
            categories = list(category_dict.keys())

        dataframe["Planet Category"] = np.select(
            conditions, categories, default="Unknown"
        )
        return dataframe

    def _format_results(
        self,
        dataframe: pd.DataFrame,
        result_format: str,
        additional_columns: Optional[str | list[str]] = None,
    ) -> pd.DataFrame:
        """
        Format the results of the mock population or observation
        to the specified format.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame containing the mock population or observation.
        result_format : str,
            The format of the result DataFrame. Must be one of
            'minimal', 'reconstructable', or 'full'.
            The available formats are:
                - 'minimal': contains the planet radius, mass and orbital period,
                             and the TargetID.
                - 'reconstructable': contains the 'minimal' columns,
                                     plus cos_i, system_id,
                                     planet_id, and gaiaID_DR3. Can be
                                     used to reconstruct all columns. The
                                     system_id and planet_id are the indices
                                     of the system and planet in the NGPPS
                                     population.
                - 'full': contains all columns in the mock population,
                          including the stellar parameters.
        additional_columns : Optional[str | list[str]], optional
            A list of additional columns to include in the result DataFrame,
            by default None.

        Returns
        -------
        pd.DataFrame
            The formatted DataFrame containing the mock population
            or observation.

        """
        if result_format == "minimal":
            columns = [
                "TargetID",
                "R_planet",
                "M_planet",
                "Porb",
            ]

        elif result_format == "reconstructable":
            columns = [
                "TargetID",
                "R_planet",
                "M_planet",
                "Porb",
                "cos_i",
                "system_id",
                "planet_id",
                "gaiaID_DR3",
            ]
        elif result_format == "full":
            columns = dataframe.columns
        else:
            raise ValueError(
                "return_full must be one of 'minimal', 'reconstructable', or 'full'."
            )

        if additional_columns:
            additional_columns = (
                [additional_columns]
                if isinstance(additional_columns, str)
                else additional_columns
            )
        else:
            additional_columns = []

        unique_columns = set(list(columns) + additional_columns)  # remove duplicates
        return dataframe[list(unique_columns)]

    def create_mock_population(
        self,
        decay_parameter: float = 10 * np.log(10),
        metallicity_limit: Optional[float] = None,
        add_planet_category: bool = True,
        result_format: str = "minimal",
        additional_columns: Optional[str | list[str]] = None,
    ) -> pd.DataFrame:
        """
        Create a mock population of planetary systems by assigning
        random planetary systems to stars in the stellar population.
        The assigment probability is [Fe/H]-dependent, and calculated
        in the calculate_probabilities method. The planetary systems
        are also assigned random inclination angles.

        A TargetID column is added to the mock population, to identify
        each unique system. This is in contrast to the system_id and
        planet_id columns, which are the indices of the system and
        planet in the NGPPS population, and can have duplicates.

        The assigned planets are checked for Roche limit crossing,
        and any planets that cross the Roche limit are removed from
        the mock population.

        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star. With a value of 10 * log_e(10), the
            probability of assigning a system drops by a factor of
            10 for each 0.1 dex difference in metallicity.
        metallicity_limit : Optional[float], optional
            The minimum metallicity of the star to be included in
            the mock population. If None, no limit is
            applied, by default None.
        add_planet_category : bool, optional
            If True, a column containing the planet category is
            added to the mock population, by default True.
        result_format : str, optional
            The format of the result DataFrame. Must be one of
            'minimal', 'reconstructable', or 'full'.
            The available formats are:
                - 'minimal': contains the planet radius, mass and orbital period,
                             and the TargetID.
                - 'reconstructable': contains the 'minimal' columns,
                                     plus cos_i, system_id,
                                     planet_id, and gaiaID_DR3. Can be
                                     used to reconstruct all columns. The
                                     system_id and planet_id are the indices
                                     of the system and planet in the NGPPS
                                     population.
                - 'full': contains all columns in the mock population,
                          including the stellar parameters,
            by default "full".
        additional_columns : Optional[str | list[str]], optional
            A list of additional columns to include in the result DataFrame,
            by default None.

        Returns
        -------
        pd.DataFrame
            The mock population of planetary systems, with the assigned
            planetary systems, stellar properties, and random inclination
            angles.
        """

        # assign random systems to stars
        rand_systems = self.assign_random_systems(
            decay_parameter=decay_parameter,
            get_random_indices=True,
        )

        # assign random inlcination angles (cos(i) is uniform between 0 and 1)
        random_cos_i = np.random.rand(self.stellar_population.shape[0])

        # get the systems corresponding to the assigned system ids
        # (random indices differ by 1 from system ids)
        systems = pd.concat([self.system_populations[ind] for ind in rand_systems])
        num_planets = [self.system_num_planets[ind] for ind in rand_systems]

        stellar_properties = self.stellar_population.copy()
        stellar_properties["cos_i"] = random_cos_i

        stellar_properties = stellar_properties.loc[
            np.repeat(
                stellar_properties.index,
                num_planets,
            )
        ].reset_index(drop=False, names="TargetID")

        systems = pd.concat(
            [
                systems.reset_index(drop=True),
                stellar_properties,
            ],
            axis=1,
        )

        # remove impossible planets
        roche_limit = (
            self.detection_model.transit_model.calculate_Roche_limit(
                m_p=systems["M_planet"].to_numpy() * u.Mearth,
                r_p=systems["R_planet"].to_numpy() * u.Rearth,
                m_star=systems["M_star"].to_numpy() * u.Msun,
                r_star=systems["R_star"].to_numpy() * u.Rsun,
            )
            .to(u.AU)
            .value
        )
        systems = systems[systems["a"] > roche_limit]

        if metallicity_limit:
            systems = systems[systems["[Fe/H]"] > metallicity_limit]

        systems["Porb"] = self.detection_model.transit_model.calculate_orbital_period(
            a=systems["a"].to_numpy() * u.AU,
            m_star=systems["M_star"].to_numpy() * u.Msun,
        ).to(u.day)

        systems = self._format_results(
            systems,
            result_format,
            additional_columns=additional_columns,
        )

        if add_planet_category:
            systems = self.add_planet_category(systems)

        return systems.reset_index(drop=True)

    def create_mock_observation(
        self,
        decay_parameter: float = 10 * np.log(10),
        mass_cut: Optional[float] = 0.5,
        metallicity_limit: Optional[float] = None,
        draw_sample: bool = True,
        add_detection_efficiency: bool = False,
        add_planet_category: bool = False,
        remove_zeros: bool = True,
        result_format: str = "minimal",
        additional_columns: Optional[str | list[str]] = None,
    ) -> pd.DataFrame:
        """
        Create a mock observation of the NGPPS population.
        First creates a mock population using the create_mock_population
        method, then calculates the detection efficiency of each planet
        in the mock population using the detection model. If draw_sample
        is True, a random sample of planets is drawn from the mock
        population, based on the detection efficiency of each planet.


        Parameters
        ----------
        decay_parameter : float
            The rate parameter of the exponential distribution
            used to calculate the probabilities of assigning a
            system to a star. With a value of 10 * log_e(10), the
            probability of assigning a system drops by a factor of
            10 for each 0.1 dex difference in metallicity.
        mass_cut : Optional[float], optional
            The minimum mass of the planet to be included in the
            mock population. If None, no limit is applied, by default 0.5.
        metallicity_limit : Optional[float], optional
            The minimum metallicity of the star to be included in
            the mock population. If None, no limit is
            applied, by default None.
        draw_sample : bool, optional
            If True, a random sample of planets is drawn from the
            mock population, based on the detection efficiency of
            each planet, by default False.
        add_detection_efficiency : bool, optional
            If True, a column containing the detection efficiency
            of each planet is added to the mock population, by
            default False.
        add_planet_category : bool, optional
            If True, a column containing the planet category is
            added to the mock population, by default False.
        remove_zeros : bool, optional
            If True, planets with a detection efficiency of 0 are
            removed from the mock population, by default True.
        result_format : str, optional
            The format of the result DataFrame. Must be one of
            'minimal', 'reconstructable', or 'full'.
            The available formats are:
                - 'minimal': contains the planet radius, mass and orbital period,
                             and the TargetID.
                - 'reconstructable': contains the 'minimal' columns,
                                     plus cos_i, system_id,
                                     planet_id, and gaiaID_DR3. Can be
                                     used to reconstruct all columns. The
                                     system_id and planet_id are the indices
                                     of the system and planet in the NGPPS
                                     population.
                - 'full': contains all columns in the mock population,
                          including the stellar parameters,
            by default "minimal".
        additional_columns : Optional[str | list[str]], optional
            A list of additional columns to include in the result DataFrame,
            by default None.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the mock observation of the NGPPS
            population, at the specified level of detail.
        """

        mock_population = self.create_mock_population(
            decay_parameter=decay_parameter,
            metallicity_limit=metallicity_limit,
            result_format="full",
        )

        if mass_cut:
            mock_population = mock_population[mock_population["M_planet"] > mass_cut]

        mock_population["detection_efficiency"] = (
            self.detection_model.detection_efficiency(mock_population)
        )

        if remove_zeros:
            mock_population = mock_population[
                mock_population["detection_efficiency"] > 0
            ]

        if draw_sample:
            # draw a random sample of planets based on the detection efficiency
            random_val = np.random.rand(mock_population.shape[0])
            mock_population = mock_population[
                random_val <= mock_population["detection_efficiency"]
            ]

        additional_columns = (
            [additional_columns]
            if isinstance(additional_columns, str)
            else (additional_columns or [])
        )
        if add_detection_efficiency:
            additional_columns.append("detection_efficiency")

        mock_population = self._format_results(
            mock_population,
            result_format,
            additional_columns=additional_columns,
        )

        if add_planet_category:
            mock_population = self.add_planet_category(mock_population)

        return mock_population.reset_index(drop=True)
