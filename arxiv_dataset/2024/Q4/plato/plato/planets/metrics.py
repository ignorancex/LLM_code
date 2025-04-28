from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas._typing import AggFuncTypeBase


class PlanetPopulationMetrics:
    """
    Class to calculate planet population metrics.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the PlanetPopulationMetrics class.

        """

    def calculate_number_of_planets(
        self,
        planet_population: pd.DataFrame,
        by_population: bool = False,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the number of planets for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        by_population : bool, optional
            If True, calculate the number of planets for each category
            within each population. In this case, the "Population" column
            must be present in the DataFrame, by default False.
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the number of planets for each category.
        """
        if by_population:
            num_planets = planet_population.groupby(
                ["Population", "Planet Category"]
            ).size()
        else:
            num_planets = planet_population.groupby("Planet Category").size()

        if remove_dwarfs and "Dwarf" in num_planets.index:
            num_planets = num_planets.drop(index="Dwarf")
        assert isinstance(num_planets, pd.Series)
        return num_planets

    def calculate_fraction_of_systems_with_planet(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int | dict[str, int],
        by_population: bool = False,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the fraction of systems with at least one planet
        (per category).

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int | dict[str, int]
            The number of systems for which the population was
            determined. If by_population is True, this should be a
            dictionary with the population name as key and the number
            of systems as value.
        by_population : bool, optional
            If True, calculate the fraction of systems with at least one
            planet for each category within each population. In this case,
            the "Population" column must be present in the DataFrame,
            by default False.
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the fraction of systems with at least one planet
            for each category.
        """
        if by_population and not isinstance(number_of_systems, dict):
            raise ValueError(
                "If by_population is True, number_of_systems must be a dictionary, "
                "with the population name as key and the number of systems as value."
            )
        if not by_population and not isinstance(number_of_systems, int):
            raise ValueError(
                "If by_population is False, number_of_systems must be an integer."
            )

        if by_population:
            systems_with_planet_type = planet_population.groupby(
                ["Population", "Planet Category"]
            )["TargetID"].nunique()
        else:
            systems_with_planet_type = planet_population.groupby("Planet Category")[
                "TargetID"
            ].nunique()

        if not by_population:
            assert isinstance(number_of_systems, int)
            fraction_systems_with_planet = systems_with_planet_type / number_of_systems
        else:
            assert isinstance(number_of_systems, dict)
            fraction_systems_with_planet = systems_with_planet_type.copy().astype(float)
            for population in fraction_systems_with_planet.index.get_level_values(
                "Population"
            ).unique():
                fraction_systems_with_planet[population] = (
                    fraction_systems_with_planet[population]
                    / number_of_systems[population]
                )

        if remove_dwarfs and "Dwarf" in fraction_systems_with_planet.index:
            fraction_systems_with_planet = fraction_systems_with_planet.drop(
                index="Dwarf"
            )

        return fraction_systems_with_planet

    def calculate_occurrence_rate(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int | dict[str, int],
        by_population: bool = False,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the occurrence rate (average number of planets per system)
        for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int | dict[str, int]
            The number of systems for which the population was
            determined. If by_population is True, this should be a
            dictionary with the population name as key and the number
            of systems as value.
        by_population : bool, optional
            If True, calculate the occurrence rate for each category
            within each population. In this case, the "Population" column
            must be present in the DataFrame, by default False.
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the occurrence rate for each category.

        """
        if by_population and not isinstance(number_of_systems, dict):
            raise ValueError(
                "If by_population is True, number_of_systems must be a dictionary, "
                "with the population name as key and the number of systems as value."
            )
        if not by_population and not isinstance(number_of_systems, int):
            raise ValueError(
                "If by_population is False, number_of_systems must be an integer."
            )

        number_of_planets = self.calculate_number_of_planets(
            planet_population,
            by_population=by_population,
            remove_dwarfs=remove_dwarfs,
        )
        if not by_population:
            assert isinstance(number_of_systems, int)
            occurrence_rate = number_of_planets / number_of_systems
        else:
            assert isinstance(number_of_systems, dict)
            occurrence_rate = number_of_planets.copy().astype(float)
            for population in occurrence_rate.index.get_level_values(
                "Population"
            ).unique():
                occurrence_rate[population] = (
                    number_of_planets[population] / number_of_systems[population]
                )

        if remove_dwarfs and "Dwarf" in occurrence_rate.index:
            occurrence_rate = occurrence_rate.drop(index="Dwarf")

        return occurrence_rate

    def calculate_multiplicity(
        self,
        planet_population: pd.DataFrame,
        by_population: bool = False,
        remove_dwarfs: bool = True,
    ) -> pd.Series:
        """
        Calculate the multiplicity (average number of planets per system
        with at least one planet) for each category.

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        population : Optional[str], optional
            Name of the population to filter by,
            by default None
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.Series
            Series containing the multiplicity for each category.
        """

        number_of_planets = self.calculate_number_of_planets(
            planet_population,
            by_population,
            remove_dwarfs,
        )

        if not by_population:
            number_of_containing_systems = planet_population.groupby("Planet Category")[
                "TargetID"
            ].nunique()
        else:
            number_of_containing_systems = planet_population.groupby(
                ["Population", "Planet Category"]
            )["TargetID"].nunique()

        multiplicity = number_of_planets / number_of_containing_systems

        if remove_dwarfs and "Dwarf" in multiplicity.index:
            multiplicity = multiplicity.drop(index="Dwarf")
        return multiplicity

    def calculate_metrics(
        self,
        planet_population: pd.DataFrame,
        number_of_systems: int | dict[str, int],
        by_population: bool = False,
        remove_dwarfs: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all metrics for the planet population, meaning the
            - number of planets
            - fraction of systems with planet
            - occurrence rate
            - multiplicity

        Parameters
        ----------
        planet_population : pd.DataFrame
            The planet population data, must contain columns "TargetID",
            "Planet Category", and optionally "Population",
            if filtering by population is desired.
        number_of_systems : int | dict[str, int]
            The number of systems for which the population was
            determined. If by_population is True, this should be a
            dictionary with the population name as key and the number
            of systems as value.
        by_population : bool, optional
            If True, calculate the metrics for each category within
            each population. In this case, the "Population" column
            must be present in the DataFrame, by default False.
        remove_dwarfs : bool, optional
            Whether to remove the "Dwarf" category from the results,
            by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated metrics.
        """

        num_planets = self.calculate_number_of_planets(
            planet_population,
            by_population,
            remove_dwarfs,
        )
        fraction_systems_with_planet = self.calculate_fraction_of_systems_with_planet(
            planet_population,
            number_of_systems,
            by_population,
            remove_dwarfs,
        )
        occurrence_rate = self.calculate_occurrence_rate(
            planet_population,
            number_of_systems,
            by_population,
            remove_dwarfs,
        )
        multiplicity = self.calculate_multiplicity(
            planet_population,
            by_population,
            remove_dwarfs,
        )

        metrics = pd.DataFrame(
            {
                "Number of Planets": num_planets,
                "System Fraction": fraction_systems_with_planet,
                "Occurrence Rate": occurrence_rate,
                "Multiplicity": multiplicity,
            }
        )

        return metrics

    def calculate_metrics_stats(
        self,
        metric_dataframes: list[pd.DataFrame | pd.Series],
        grouping_column: str | list[str],
        format_uncertainties: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Calculate the statistics of the metrics for a list
        of metric dataframes. Specifically, calculate the
        median, 16th, and 84th percentiles of the metrics.

        Parameters
        ----------
        metric_dataframes : list[pd.DataFrame | pd.Series]
            List of DataFrames containing the metrics for each
            planet population, calculated using the calculate_metrics
            method.
        grouping_column : str | list[str]
            Column(s) to group the results by.
        format_uncertainties : bool, optional
            Whether to format the uncertainties in the results
            dataframe for latex, by default False.
        **kwargs : Any
            Additional keyword arguments passed to the format_uncertainties
            method.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the statistics of the metrics.

        """
        dataframe = pd.concat(metric_dataframes)

        agg_functions: list[AggFuncTypeBase] = [
            lambda x: x.quantile(0.50),
            lambda x: x.quantile(0.50) - x.quantile(0.16),
            lambda x: x.quantile(0.84) - x.quantile(0.50),
        ]
        rename_dict = {
            "<lambda_0>": "Median",
            "<lambda_1>": "16th",
            "<lambda_2>": "84th",
        }

        agg_results = dataframe.groupby(grouping_column).agg(agg_functions)

        agg_results = agg_results.rename(columns=rename_dict)

        if format_uncertainties:
            agg_results = self.format_uncertainties(agg_results, **kwargs)

        return agg_results

    def format_uncertainties(
        self,
        dataframe: pd.DataFrame,
        median_col: str = "Median",
        upper_col: str = "84th",
        lower_col: str = "16th",
        new_col_name: str = "Formatted",
        round_decimals: Optional[int] = None,
        to_int: bool = False,
    ) -> pd.DataFrame:
        """
        Format the uncertainties in the results dataframe for latex.
        This method takes a DataFrame containing the median, 16th, and 84th
        percentiles of the metrics, and formats the uncertainties in a
        latex-friendly way. The formatted uncertainties are stored in a new
        column in the DataFrame. The uncertainties are formatted as
        $median^{+upper}_{-lower}$.
        The dataframe can have a single-level or multi-level column structure.
        In the latter case, the formatting is applied to each top-level column.


        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame containing the data and uncertainies.
        median_col : str, optional
            Column name of the central values, by default "Median".
        upper_col : str, optional
            Column name of the upper uncertainties, by default "84th".
        lower_col : str, optional
            Column name of the lower uncertainties, by default "16th".
        new_col_name : str, optional
            Name of the new column to store the formatted uncertainties,
            by default "Formatted".
        round_decimals : Optional[int], optional
            Number of decimals to round the values to, by default None.
        to_int : bool, optional
            Whether to convert the values to integers, by default False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the formatted uncertainties.
        """
        dataframe = dataframe.copy()
        if isinstance(round_decimals, int):
            dataframe = dataframe.round(round_decimals)
        elif round_decimals == "ceil":
            dataframe = np.ceil(dataframe)  # type: ignore
        elif round_decimals == "floor":
            dataframe = np.floor(dataframe)  # type: ignore
        elif round_decimals is not None:
            pass
        else:
            raise ValueError("round_decimals must be an integer, 'ceil', or 'floor'")

        dataframe = dataframe.astype(int) if to_int else dataframe

        def apply_formatting(df: pd.DataFrame) -> pd.DataFrame:
            df[new_col_name] = df.apply(
                lambda row: (
                    f"${row[median_col]}"
                    f"^{{+{row[upper_col]}}}_{{-{row[lower_col]}}}$"
                ),
                axis=1,
            )
            return df.drop(columns=[median_col, upper_col, lower_col])

        if isinstance(dataframe.columns, pd.MultiIndex):
            # create a dictionary to hold formatted DataFrames for each top level column
            formatted_dfs = {
                top_level: apply_formatting(dataframe[top_level])
                for top_level in dataframe.columns.levels[0]
            }
            # Concatenate the formatted DataFrames along the columns axis
            dataframe = pd.concat(formatted_dfs, axis=1)
            # remove multiindexing
            dataframe.columns = dataframe.columns.droplevel(1)
        else:
            # Apply formatting for single-level column structure
            dataframe = apply_formatting(dataframe)

        return dataframe
