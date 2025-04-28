# -*- coding: utf-8 -*-
import importlib
import json
import pathlib
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from histogram_tools import find_xrange
from latex_dicts import inv_param_dict, mod_dict, uv_param_dict
from matplotlib import rc, use

# import arviz as az
from sigfig import round

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
rc("text", usetex=True)

collections = ["Granada"]


here = pathlib.Path(__file__).parent

# result dir
result_dir = here / "results_fcc"
# Path.mkdir(result_dir, parents=True, exist_ok=True)
#
# mod_list = []
# for col in collections:
#     base_path = pathlib.Path(f"{here.parent}/runcards/uv_models/UV_scan/{col}/")
#     sys.path = [str(base_path)] + sys.path
#     for p in base_path.iterdir():
#         if p.name.startswith("InvarsFit") and p.suffix == ".py":
#             mod_list.append(importlib.import_module(f"{p.stem}"))
#
#
# use("PDF")
# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
# rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amssymb}"})
#
# # compute the invariants
# pQCD = ['NLO']
# EFT = ['NHO']
#
# for model in mod_list:
#     for pQCD in ['LO', 'NLO']:
#         for EFT in ['NHO', 'HO']:
#             model.MODEL_SPECS['pto'] = pQCD
#             model.MODEL_SPECS['eft'] = EFT
#             invariants = []
#             for k, attr in model.__dict__.items():
#                 if k.startswith('inv'):
#                     invariants.append(attr)
#             try:
#                 model.inspect_model(model.MODEL_SPECS, invariants)
#             except FileNotFoundError:
#                 print("File not found", model)
#                 continue


# Specify the path to the JSON file
posterior_path = f"{here.parent}/results/smefit_fcc_uv/{{}}_{{}}_UV_{{}}_{{}}_{{}}_NS/inv_posterior.json"


def nested_dict(num_levels):
    if num_levels == 1:
        return defaultdict(dict)
    else:
        return defaultdict(lambda: nested_dict(num_levels - 1))


def compute_bounds(collection, mod_nrs, label, caption):
    table = "\\begin{table}\n"
    table += "\\begin{center}\n"
    table += "\\scriptsize\n"
    table += "\\renewcommand{\\arraystretch}{1.5}"
    table += "\\begin{tabular}{c|c|c|c|c|c}\n"
    table += "Model & UV invariants & LO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ &LO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-2}\\right)$ & NLO $\\mathcal{O}\\left(\\Lambda^{-4}\\right)$ \\\\\n"
    table += "\\toprule\n"

    table_dict = nested_dict(4)

    n_params = 0
    for mod_nr in mod_nrs:
        print(mod_nr)

        model_row = True

        n_invariants = []
        for pQCD in ["LO", "NLO"]:
            for EFT in ["NHO", "HO"]:
                n_inv = 0
                # path to posterior with model number
                posterior_path_mod = Path(
                    posterior_path.format(collection, mod_nr, pQCD, EFT)
                )

                if posterior_path_mod.exists():
                    # Open the JSON file and load its contents
                    with open(posterior_path_mod) as f:
                        posterior = json.load(f)

                    for key, samples in posterior.items():
                        if not key.startswith("inv"):
                            continue
                        else:
                            if min(samples) > 0:
                                low = 0
                                up = np.nanpercentile(samples, 95.0)
                                up_str = (
                                    "{}".format(round(up, sigfigs=4, notation="sci"))
                                    .replace("E", "e")
                                    .replace("e0", "")
                                )
                                if "e" in up_str:
                                    base, exp = up_str.split("e")
                                    up_str = f"{base}$\\mathrm{{e}}{{{exp}}}$"
                                table_dict[mod_nr][key][pQCD][EFT] = up_str

                            else:
                                low, up = np.nanpercentile(
                                    samples, 2.5
                                ), np.nanpercentile(samples, 97.5)
                                low_str = (
                                    "{}".format(round(low, sigfigs=4, notation="sci"))
                                    .replace("E", "e")
                                    .replace("e0", "")
                                )
                                up_str = (
                                    "{}".format(round(up, sigfigs=4, notation="sci"))
                                    .replace("E", "e")
                                    .replace("e0", "")
                                )
                                if "e" in low_str:
                                    base, exp = low_str.split("e")
                                    low_str = "{}$\\mathrm{{e}}{{{}}}$".format(
                                        base, exp
                                    )
                                if "e" in up_str:
                                    base, exp = up_str.split("e")
                                    up_str = f"{base}$\\mathrm{{e}}{{{exp}}}$"
                                table_dict[mod_nr][key][pQCD][EFT] = "[{}, {}]".format(
                                    low_str, up_str
                                )

                            if up > 200 or low < -200:
                                table_dict[mod_nr][key][pQCD][EFT] = "$\\times$"
                            # hdi is two sides, so we cannot use it: |g| > 0 requires one sided test
                            # table_dict[mod_nr][key][pQCD][EFT] = az.hdi(np.array(samples), hdi_prob=.95)
                            # table_dict[mod_nr][key][pQCD][EFT] = low, up
                            n_inv += 1
                    n_invariants.append(n_inv)
        print("n_invariants:", n_invariants)
        if n_invariants:
            n_params += max(n_invariants)
        for parameter, param_dict in table_dict[mod_nr].items():
            LO_NHO = param_dict["LO"]["NHO"]
            LO_HO = param_dict["LO"]["HO"]
            NLO_NHO = param_dict["NLO"]["NHO"]
            NLO_HO = param_dict["NLO"]["HO"]
            if model_row:
                table += f"{mod_dict[mod_nr]} & {inv_param_dict[mod_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO} \\\\\n"
                model_row = False
            else:
                table += f" & {inv_param_dict[mod_nr][parameter]} & {LO_NHO} & {LO_HO} & {NLO_NHO} & {NLO_HO}  \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += f"\\caption{{{caption}}}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    return table, n_params


scalar_mdl_nrs = range(21)
vboson_mdl_nrs = range(21, 37)
vfermion_mdl_nrs = range(37, 50)
mp_mdl_idx = ["Q1_Q7_W"]
oneloop_mdl_idx = ["T1", "T2"]
#
# latex_table_scalar, n_scalars = compute_bounds("Granada", scalar_mdl_nrs, "cl-heavy-scalar",
#                                               "95\\% CL intervals of the heavy scalar fields UV couplings.")
#
# latex_table_vboson, n_vbosons = compute_bounds("Granada", vboson_mdl_nrs, "cl-heavy-vboson",
#                                              "95\\% CL intervals of the heavy vector boson fields UV couplings.")
#
# latex_table_vfermion, n_vfermions = compute_bounds("Granada", vfermion_mdl_nrs, "cl-heavy-vfermion",
#                                                   "95\\% CL intervals of the heavy vector fermion fields UV couplings.")
#
# latex_table_mp, n_mp = compute_bounds("MultiParticleCollection", mp_mdl_idx, "cl-mp-model-spl", "95\\% CL intervals of the UV couplings that enter in the multiparticle model.")
# latex_table_1l, n_scalars_1L = compute_bounds("OneLoop", oneloop_mdl_idx, "cl-1l-model", "95\\% CL intervals of the UV couplings in \\varphi at one-loop.")


# Save LaTeX code to a .tex file
# with open(result_dir / "table_scalar.tex", "w") as file:
#     file.write(latex_table_scalar)
# with open(result_dir / "table_vboson.tex", "w") as file:
#    file.write(latex_table_vboson)
# with open(result_dir / "table_vfermion.tex", "w") as file:
#     file.write(latex_table_vfermion)
# with open(result_dir / "table_mp.tex", "w") as file:
#    file.write(latex_table_mp)
# with open(result_dir / "table_1l.tex", "w") as file:
#     file.write(latex_table_1l)


def plot_uv_posterior_bar(n_params, collection, models, EFT=None, name=None, pQCD=None):
    fig, axes = plt.subplots(figsize=(18, 8 * 3), ncols=1, nrows=3)

    category = [
        r"${\rm Heavy\:Scalar}$",
        r"${\rm Heavy\:Fermion}$",
        r"${\rm Heavy\:Vector\:Boson}$",
    ]

    for i, mod_nrs in enumerate(models):
        lhc_bounds = []
        hllhc_bounds = []
        fcc_bounds = []
        x_labels = []
        for mod_nr in mod_nrs:
            if mod_nr == 5:
                continue
            print(mod_nr)

            posterior_path_mod_1 = Path(
                posterior_path.format(collection, "lhc", mod_nr, pQCD, "HO")
            )
            posterior_path_mod_2 = Path(
                posterior_path.format(collection, "hllhc", mod_nr, pQCD, "HO")
            )
            posterior_path_mod_3 = Path(
                posterior_path.format(collection, "fcc", mod_nr, pQCD, "HO")
            )

            if posterior_path_mod_1.exists():
                # Open the JSON file and load its contents
                try:
                    with open(posterior_path_mod_1) as f:
                        posterior_1 = json.load(f)

                    with open(posterior_path_mod_2) as f:
                        posterior_2 = json.load(f)

                    with open(posterior_path_mod_3) as f:
                        posterior_3 = json.load(f)
                except FileNotFoundError:
                    continue

                n_invariants = 0
                for key in posterior_1.keys():
                    if key.startswith("inv"):
                        n_invariants += 1

                if n_invariants > 1:
                    continue

                for (
                    (key, samples_1_list),
                    (_, samples_2_list),
                    (_, samples_3_list),
                ) in zip(posterior_1.items(), posterior_2.items(), posterior_3.items()):
                    if not key.startswith("inv"):
                        continue
                    else:
                        samples_1 = np.array(samples_1_list)
                        samples_2 = np.array(samples_2_list)
                        samples_3 = np.array(samples_3_list)

                        lhc_up = np.percentile(np.abs(samples_1), 95)
                        hllhc_up = np.percentile(np.abs(samples_2), 95)
                        fcc_up = np.percentile(np.abs(samples_3), 95)

                        lhc_bounds.append([1 / lhc_up, 4 * np.pi / lhc_up])
                        hllhc_bounds.append([1 / hllhc_up, 4 * np.pi / hllhc_up])
                        fcc_bounds.append([1 / fcc_up, 4 * np.pi / fcc_up])

                        x_labels.append(mod_dict[mod_nr])

        lhc_bounds = np.array(lhc_bounds)
        hllhc_bounds = np.array(hllhc_bounds)
        fcc_bounds = np.array(fcc_bounds)

        total_width = 8
        bar_width = total_width / 2
        x = np.linspace(0, 200, len(lhc_bounds))
        dx = x[1] - x[0]

        logo = plt.imread(
            "/data/theorie/jthoeve/smefit_release/src/smefit/analyze/logo.png"
        )

        axes[-1].imshow(
            logo,
            aspect="auto",
            transform=axes[-1].transAxes,
            extent=[0.88, 0.98, 0.87, 0.97],
            zorder=10,
        )

        # Plot the bars

        axes[i].bar(
            x - bar_width,
            lhc_bounds[:, 0],
            bar_width,
            align="center",
            label=r"$\rm{LEP+LHC_{Run-2}}, \:g_{{\rm UV}}=1$",
            color="C0",
        )

        axes[i].bar(
            x - bar_width,
            lhc_bounds[:, 1],
            bar_width,
            align="center",
            edgecolor="C0",
            linestyle="--",
            alpha=0.3,
            label=r"$\rm{LEP+LHC_{Run-2}},\:g_{{\rm UV}}=4\pi$",
        )

        axes[i].bar(
            x,
            hllhc_bounds[:, 0],
            bar_width,
            align="center",
            label=r"$\rm{+\:HL-LHC}, \:g_{{\rm UV}}=1$",
            color="C1",
        )

        axes[i].bar(
            x,
            hllhc_bounds[:, 1],
            bar_width,
            align="center",
            edgecolor="C1",
            linestyle="--",
            alpha=0.3,
            label=r"$\rm{+\:HL-LHC}, \:g_{{\rm UV}}=4\pi$",
        )

        axes[i].bar(
            x + bar_width,
            fcc_bounds[:, 0],
            bar_width,
            align="center",
            label=r"$\rm{+\:FCC-ee}, \:g_{{\rm UV}}=1$",
            color="C2",
        )

        axes[i].bar(
            x + bar_width,
            fcc_bounds[:, 1],
            bar_width,
            align="center",
            edgecolor="C2",
            linestyle="--",
            alpha=0.3,
            label=r"$\rm{+\:FCC-ee}, \:g_{{\rm UV}}=4\pi$",
        )

        axes[i].set_xticks(x, x_labels)
        axes[i].set_ylabel(r"$M_{\mathrm{UV}}\:{\rm [TeV]}$")
        axes[i].tick_params(axis="x", which="major", pad=15)
        # axes[i].legend(frameon=False, ncol=3)
        axes[i].set_yscale("log")

        if i == 2:
            axes[i].set_ylim(0.08, 5 * 10**3)

        axes[i].text(
            0.02,
            0.95,
            category[i],
            fontsize=24,
            ha="left",
            va="top",
            transform=axes[i].transAxes,
        )

    axes[0].legend(frameon=False, ncol=3, loc="upper right", fontsize=18)

    # axes[-1].set_xlim(x.min() - 2 * bar_width, x.max() + 2 * bar_width)

    fig.tight_layout()

    fig.savefig(result_dir / "fcc_uv_bounds.pdf")


def plot_uv_posterior(n_params, collection, mod_nrs, EFT=None, name=None, pQCD=None):
    plot_nr = 1

    # Calculate number of rows and columns
    n_cols = max(int(np.ceil(np.sqrt(n_params))), 4)
    if n_params < n_cols:
        n_cols = n_params

    n_rows = int(np.ceil(n_params / n_cols))

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for mod_nr in mod_nrs:
        if mod_nr == 5:
            continue
        print(mod_nr)

        # path to posterior with model number
        if pQCD is None:
            posterior_path_mod_1 = Path(
                posterior_path.format(collection, mod_nr, "LO", EFT)
            )
            posterior_path_mod_2 = Path(
                posterior_path.format(collection, mod_nr, "NLO", EFT)
            )
        elif EFT is None:
            posterior_path_mod_1 = Path(
                posterior_path.format(collection, "lhc", mod_nr, pQCD, "NHO")
            )
            posterior_path_mod_2 = Path(
                posterior_path.format(collection, "hllhc", mod_nr, pQCD, "NHO")
            )
            posterior_path_mod_3 = Path(
                posterior_path.format(collection, "fcc", mod_nr, pQCD, "NHO")
            )
        else:
            print("EFT and pQCD cannot both be None, aborting")
            sys.exit()

        if posterior_path_mod_1.exists():
            # Open the JSON file and load its contents
            try:
                with open(posterior_path_mod_1) as f:
                    posterior_1 = json.load(f)

                with open(posterior_path_mod_2) as f:
                    posterior_2 = json.load(f)

                with open(posterior_path_mod_3) as f:
                    posterior_3 = json.load(f)
            except FileNotFoundError:
                continue

            for (key, samples_1_list), (_, samples_2_list), (_, samples_3_list) in zip(
                posterior_1.items(), posterior_2.items(), posterior_3.items()
            ):
                if not key.startswith("inv"):
                    continue
                else:
                    samples_1 = np.array(samples_1_list)
                    samples_2 = np.array(samples_2_list)
                    samples_3 = np.array(samples_3_list)
                    ax = plt.subplot(n_rows, n_cols, plot_nr)
                    plot_nr += 1

                    x_low, x_high = find_xrange([samples_1, samples_2], 0.05)

                    ax.hist(
                        samples_1[(samples_1 > x_low) & (samples_1 < x_high)],
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

                    ax.hist(
                        samples_2[(samples_2 > x_low) & (samples_2 < x_high)],
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

                    ax.hist(
                        samples_3[(samples_3 > x_low) & (samples_3 < x_high)],
                        bins="fd",
                        density=True,
                        edgecolor="black",
                        alpha=0.4,
                    )

                    ax.set_xlim(x_low, x_high)
                    ax.set_ylim(0, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2)

                    ax.tick_params(which="both", direction="in", labelsize=22.5)
                    ax.tick_params(labelleft=False)

                    ax.text(
                        0.05,
                        0.95,
                        inv_param_dict[mod_nr][key],
                        fontsize=17,
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                    )

                    ax.text(
                        0.95,
                        0.95,
                        mod_dict[mod_nr],
                        fontsize=20,
                        ha="right",
                        va="top",
                        transform=ax.transAxes,
                    )

    if pQCD is None:
        order_EFT = -2 if EFT == "NHO" else -4
        fig.legend(
            [
                f"$\\mathrm{{LO}}\\;\\mathcal{{O}}\\left(\\Lambda^{{{order_EFT}}}\\right)$",
                f"$\\mathrm{{NLO}}\\;\\mathcal{{O}}\\left(\\Lambda^{{{order_EFT}}}\\right)$",
            ],
            loc="upper center",
            ncol=2,
            prop={"size": 25 * (n_cols * 4) / 20},
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )

        if n_rows > 1:
            plt.tight_layout(
                rect=[0, 0.05 * (5.0 / n_rows), 1, 1 - 0.05 * (5.0 / n_rows)]
            )  # make room for the legend
        else:
            plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend
        if name is not None:
            fig.savefig(
                result_dir / f"{collection}_posteriors_LO_vs_NLO_{EFT}_{name}.pdf"
            )
        else:
            fig.savefig(result_dir / f"{collection}_posteriors_LO_vs_NLO_{EFT}.pdf")
    elif EFT is None:
        fig.legend(
            [
                f"$\\mathrm{{LHC}}\\;\\mathcal{{O}}\\left(\\Lambda^{{-2}}\\right)$",
                f"$\\mathrm{{HL-LHC}}\\;\\mathcal{{O}}\\left(\\Lambda^{{-2}}\\right)$",
                f"$\\mathrm{{FCCee}}\\;\\mathcal{{O}}\\left(\\Lambda^{{-2}}\\right)$",
            ],
            ncol=3,
            prop={"size": 25 * (n_cols * 4) / 20},
            bbox_to_anchor=(0.5, 1.0),
            loc="upper center",
            frameon=False,
        )
        if n_rows > 1:
            plt.tight_layout(
                rect=[0, 0.05 * (5.0 / n_rows), 1, 1 - 0.05 * (5.0 / n_rows)]
            )  # make room for the legend
        else:
            plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # make room for the legend

        fig.savefig(result_dir / f"{collection}_posteriors_lhc_vs_hlhc_{name}.png")


# plot_uv_posterior_bar(10, "Granada", vboson_mdl_nrs, name="vbosons", pQCD="NLO")
# plot_uv_posterior(15, "Granada", vfermion_mdl_nrs, name="vfermions", pQCD="NLO")
plot_uv_posterior_bar(
    10,
    "Granada",
    [scalar_mdl_nrs, vfermion_mdl_nrs, vboson_mdl_nrs],
    name="scalars",
    pQCD="NLO",
)

# plot_uv_posterior(n_mp, "MultiParticleCollection", mp_mdl_idx, EFT="NHO")
# plot_uv_posterior(n_mp, "MultiParticleCollection", mp_mdl_idx, EFT="HO")
# plot_uv_posterior(n_mp, "MultiParticleCollection",mp_mdl_idx, pQCD="LO")
# plot_uv_posterior(n_mp, "MultiParticleCollection", mp_mdl_idx, pQCD="NLO")

# plot_uv_posterior(10, "Granada", vboson_mdl_nrs, name="vbosons", pQCD="NLO")
# plot_uv_posterior(15, "Granada", vfermion_mdl_nrs, name="vfermions", pQCD="NLO")
# plot_uv_posterior(10, "Granada", scalar_mdl_nrs, name="scalars", pQCD="NLO")

# plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, EFT="NHO")
# plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, EFT="HO")
# plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, pQCD="LO")
# plot_uv_posterior(n_scalars_1L, "OneLoop", oneloop_mdl_idx, pQCD="NLO")
#
#
# for file in result_dir.iterdir():
#     if file.suffix == ".pdf":
#         subprocess.run(["pdfcrop", str(file), str(file)])
