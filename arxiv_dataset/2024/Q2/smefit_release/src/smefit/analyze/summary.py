# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..coefficients import Coefficient
from .latex_tools import latex_packages, multicolum_table_header


def add_bounded_coeff(coeff_dict, coeff_df):
    """Build constrain formula"""
    line = r" & & = "
    temp_c = Coefficient(
        "temp_c",
        coeff_dict["min"],
        coeff_dict["max"],
        constrain=coeff_dict["constrain"],
        value=coeff_dict.get("value", None),
    )
    if temp_c.constrain is None:
        fact = temp_c.value
        sign = "+" if fact >= 0 else "$-$"
        line += rf"{sign}\ {fact}"
        return line
    for constrain in temp_c.constrain:
        for op, val in constrain.items():
            fact, exp = val
            latex_name = coeff_df[:, op].values[0]
            sign = "+" if fact > 0 else "$-$"
            if exp > 1:
                line += rf"{sign}\ {np.abs(fact):.3f}\ $(${latex_name}$)^{exp}$"
            else:
                line += rf"{sign}\ {np.abs(fact):.3f}\ {latex_name}"
    return line


class SummaryWriter:
    """
    Provides a summary of the fits included in the report: the fit settings,
    fitted parameters (including any parameter constraints), and
    datasets.

    Uses data_references.yaml, data_groups.yaml, and coeff_groups.yaml
    YAML files. The first gives the references used for hyperlinks,
    and the other two organizes the data and parameters into groups
    (top, higgs, etc.).

    Parameters
    ----------

    """

    def __init__(self, fits, data_groups, coeff_config):
        self.fits = fits
        self.data_info = data_groups
        self.coeff_info = coeff_config
        self.nfits = len(self.fits)
        # Get names of datasets for each fit
        self.dataset_fits = []
        for fit in self.fits:
            self.dataset_fits.append([data["name"] for data in fit.config["datasets"]])

    def fit_settings(self):
        """Fit settings table.

        Returns
        -------
        pd.DataFrame:
            table with the most relevant fit settings
        """
        summaries = {}
        for fit in self.fits:
            summary_dict = {}
            summary_dict["EFT order"] = (
                "Qudratic" if fit.config["use_quad"] else "Linear"
            )
            summary_dict["Replicas"] = fit.n_replica
            label = fit.label.replace(r"\ ", "").replace(r"\rm", "")
            summaries[label] = summary_dict
        return pd.DataFrame(summaries)

    def write_dataset_table(self):
        """Write the summary tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = latex_packages()
        L.extend(
            [
                r"\usepackage{underscore}",
                r"\begin{document}",
                r"\begin{longtable}{|c|l|" + "c|" * self.nfits + "}",
                r"\hline",
                r"\footnotesize",
            ]
        )
        temp = " Type & Datasets "
        for fit in self.fits:
            temp += f" & {fit.label}"
        temp += r" \\ \hline"
        L.append(temp)

        for group, datasets in self.data_info.groupby(level=0):
            datasets = datasets.droplevel(0)
            L.append(f"\\multirow{{{datasets.shape[0]}}}{{*}}{{{group}}}")
            for isub, (dataset, link) in enumerate(datasets.items()):
                temp = r" & \href{" + link + "}{" + dataset + "} "
                for data in self.dataset_fits:
                    temp += " & "
                    if dataset in data:
                        temp += r"$\checkmark$"
                if isub != datasets.shape[0] - 1:
                    temp += f"\\\\ \\cline{{2-{(2 + self.nfits)}}}"
                L.append(temp)
            L.append(r"\\ \hline")

        # TODO: load dataset to print this info ??
        # temp = r"\hline & Total number of data points"
        # for fit in self.fits:
        #     temp += f" & {np.sum(fit.datasets.NdataExp)}"
        # temp += r" \\ \hline"
        # L.extend(
        #     [temp, r"\end{tabular}", r"\caption{Dataset comparison}", r"\end{table}"]
        # )
        L.extend([r"\caption{Dataset comparison}", r"\end{longtable}"])
        return L

    def write_coefficients_table(self):
        """Write the coefficients tables

        Returns
        -------
            L : list(str)
                list of the latex commands
        """
        L = latex_packages()
        L.extend(
            [
                r"\usepackage{underscore}",
                r"\begin{document}",
                r"\begin{table}[H]",
                r"\centering",
                r"\begin{tabular}{|c|c|" + "c|c|" * self.nfits + "}",
            ]
        )
        L.extend(multicolum_table_header([fit.label for fit in self.fits]))
        L.append(
            r"Class & Coefficients" + r" & Fitted & Fixed " * self.nfits + r" \\ \hline"
        )

        # Build lists for grouped coefficients
        n_fitted = np.zeros(self.nfits, dtype=int)

        # loop on coeff groups
        for group, coefficients in self.coeff_info.groupby(level=0):
            coefficients = coefficients.droplevel(0)
            L.append(f"\\multirow{{{coefficients.shape[0]}}}{{*}}{{{group}}}")

            # loop on coefficients
            for isub, (coeff, latex_name) in enumerate(coefficients.items()):
                temp = f" & {latex_name}"

                # loop on fits
                for i, fit in enumerate(self.fits):
                    if coeff in fit.config["coefficients"]:
                        coeff_dict = fit.config["coefficients"][coeff]
                        if "constrain" in coeff_dict:
                            temp += add_bounded_coeff(coeff_dict, self.coeff_info)
                        elif "value" in coeff_dict:
                            temp += f" & & = {coeff_dict['value']}"
                        else:
                            temp += r" & $\checkmark$ & "
                            n_fitted[i] += 1
                    else:
                        temp += r" & & "

                if isub != coefficients.shape[0] - 1:
                    temp += f"\\\\ \\cline{{2-{(2 + 2 * self.nfits)}}}"
                L.append(temp)

            L.append(r"\\ \hline")

        temp = r"\hline & Number fitted coefficients"

        # closing line
        for n in n_fitted:
            temp += f" & {n} & "
        temp += r" \\ \hline"

        L.extend(
            [
                temp,
                r"\end{tabular}",
                r"\caption{Coefficient comparison}",
                r"\end{table}",
            ]
        )
        return L
