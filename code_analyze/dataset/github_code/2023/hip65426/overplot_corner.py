import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt

# read in the nested sampling posteriors & make some nice corner plots


def combine_teff_hilow(fitname="sinfoni", model="bt-settl-cifist"):
    """
    A couple of the runs are done by setting a low prior on Teff, then re-running
    with a higher prior on Teff. This method combines them, making sure the
    relative likelihoods are weighted correctly
    """

    if "nosphereGP" in fitname:
        names = ["Teff", "logg", "R", "plx"]
        labels = [
            "T$_{{\\mathrm{{eff}}}}$ [K]",
            "$\\log{{g}}$",
            "R [R$_J$]",
            "$\pi$ [mas]",
        ]
    else:
        names = ["Teff", "logg", "R", "plx", "gp_len", "gp_amp"]
        labels = [
            "T$_{{\\mathrm{{eff}}}}$ [K]",
            "$\\log{{g}}$",
            "R [R$_J$]",
            "$\pi$ [mas]",
            "$\\log{{l_{{SPHERE_IFU}}}}$",
            "$A_{{SPHERE_IFU}}$",
        ]
    if "sinfoni" in fitname:
        names += ["rv_sinfoni"]

    names += [""]
    chain_name = "results/{}_tefflo_{}/multinest/post_equal_weights.dat".format(
        fitname, model
    )
    print(chain_name)
    loteff_post = pd.read_csv(chain_name, delim_whitespace=True, names=names)

    chain_name = "results/{}_teffhi_{}/multinest/post_equal_weights.dat".format(
        fitname, model
    )
    print(chain_name)
    hiteff_post = pd.read_csv(chain_name, delim_whitespace=True, names=names)

    return pd.concat([loteff_post, hiteff_post]), labels


sphereGP = True
jwst = True

fitname = "sinfoni"
if jwst:
    fitname += "_jwst"
if not sphereGP:
    fitname += "_nosphereGP"
sinfoni_fit, labels = combine_teff_hilow(fitname=fitname)

fitname = "gravity"
if jwst:
    fitname += "_jwst"
if not sphereGP:
    fitname += "_nosphereGP"
gravity_fit, labels = combine_teff_hilow(fitname=fitname)

fig = corner.corner(
    sinfoni_fit[sinfoni_fit.keys()[:-2]],
    bins=25,
    labels=labels,
    plot_datapoints=False,
    color="hotpink",
    plot_density=False,
    weights=np.ones(len(sinfoni_fit)) / len(sinfoni_fit),
)
corner.corner(
    gravity_fit[gravity_fit.keys()[:-1]],
    fig=fig,
    bins=25,
    labels=labels,
    plot_datapoints=False,
    color="purple",
    plot_density=False,
    weights=np.ones(len(gravity_fit)) / len(gravity_fit),
)

for ax in fig.get_axes():
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.set_title(ax.get_title(), fontsize=14)
ax.set_xlabel(ax.get_xlabel(), fontsize=14)
plt.savefig(
    "results/plots/sinfoni_gravity_corner_sphereGP{}.png".format(sphereGP), dpi=250
)
