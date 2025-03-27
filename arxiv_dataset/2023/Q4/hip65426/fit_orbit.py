import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.transforms as transforms
from astropy.time import Time

from orbitize import system, read_input, priors, sampler
from orbitize import results

"""
Fits to run:
1. lit astrometry only (True, False, False, False, False)
2. lit astrometry + first GRAVITY epoch (True, True, False, False, False)
2.5. lit astrometry + second GRAVITY epoch (True, False, True, False, False)
3. lit astrometry + first 2 GRAVITY (True, True, True, False, False)
4. first 2 GRAVITY only (False, True, True, False, False)
5. first 2 GRAVITY with e fixed to 0 (True, True, True, True, False)
6. first 2 GRAVITY with linearly decreasing prior on e (True, True, True, False, True)
7. all astrometry
8. all astrometry with e fixed to 0
9. all astrometry with linearly decreasing e prior
10. all astrometry except NACO, linearly decreasing e prior
"""

np.random.seed(10)

"""
Begin keywords <<
"""
run_fit = False
make_corner = True

lit_astrom = True
first_grav = True
second_grav = True
third_grav = True
exclude_naco = False
fix_ecc = False
lin_ecc_prior = False

savedir = "results/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

if lit_astrom:
    savedir += "with_literature_astrom"
if first_grav:
    savedir += "with_first_vlti_point"
if second_grav:
    savedir += "with_second_vlti_point"
if third_grav:
    savedir += "with_third_vlti_point"
if fix_ecc:
    savedir += "_fixed_ecc"
if lin_ecc_prior:
    savedir += "_linear_ecc"
if exclude_naco:
    savedir += "_noNACO"
"""
>> End keywords
"""

if not os.path.exists(savedir):
    os.mkdir(savedir)

input_file = "HIP65426.csv"
plx = 9.303088053872793  # [mas] (Gaia eDR3)
plx_err = 0.034559656
m_st = 1.96  # [M_sol] (Bowler+ 2020)
mass_err = 0.04

num_secondary_bodies = 1
data_table = read_input.read_file(input_file)
insts = data_table["instrument"]

if not lit_astrom:
    data_table = data_table[data_table["instrument"] == "GRAVITY"]
if not first_grav:
    data_table = data_table[0 : -2 & -2 :]
if not second_grav:
    data_table = data_table[0 : -1 & -1]
if not third_grav:
    data_table = data_table[0:-1]
if exclude_naco:
    data_table = data_table[data_table["instrument"] != "NACO"]

print(data_table)

HIP654_system = system.System(
    num_secondary_bodies,
    data_table,
    m_st,
    plx,
    fit_secondary_mass=False,
    mass_err=mass_err,
    plx_err=plx_err,
)

# fix eccentricity to 0
if fix_ecc:
    HIP654_system.sys_priors[HIP654_system.param_idx["ecc1"]] = 0

# set a linearly decreasing prior on ecc
if lin_ecc_prior:
    HIP654_system.sys_priors[HIP654_system.param_idx["ecc1"]] = priors.LinearPrior(
        -2.18, 2.01
    )

# Check that orbitizie! initialized everything correctly.
# (I wrote the code, therefore I do not trust the code.)
assert not HIP654_system.fit_secondary_mass
assert not HIP654_system.track_planet_perturbs

# run MCMC
num_threads = 50
num_temps = 20
num_walkers = 1000
num_steps = 50_000_000  # 200_000_000  # n_walkers x n_steps_per_walker
burn_steps = 10_000  # 100_000
thin = 100

if run_fit:
    HIP654_sampler = sampler.MCMC(
        HIP654_system,
        num_threads=num_threads,
        num_temps=num_temps,
        num_walkers=num_walkers,
    )
    HIP654_sampler.run_sampler(num_steps, burn_steps=burn_steps, thin=thin)

    # save chains
    HIP654_sampler.results.save_results("{}/chains.hdf5".format(savedir))

HIP654_results = results.Results()  # create blank results object for loading
HIP654_results.load_results("{}/chains.hdf5".format(savedir))

# chop chains
# num_chop = 1000
# reshaped_post = HIP654_results.post.reshape(
#     (num_walkers, num_steps // num_walkers // thin, HIP654_results.post.shape[1])
# )
# HIP654_results.post = reshaped_post[:, -num_chop:, :].reshape(
#     (-1, HIP654_results.post.shape[1])
# )

# make corner plot

if make_corner:
    if fix_ecc:
        param_list = ["sma1", "inc1", "aop1", "pan1", "tau1", "mtot", "plx"]
    else:
        param_list = None

    corner_kwargs = {"show_titles": True}  # , "quantiles": [0.16, 0.84]}

    # median_values = np.median(HIP654_results.post, axis=0)
    # range_values = np.ones_like(median_values)*0.997 # Plot only 3-sigma range for each parameter
    fig = HIP654_results.plot_corner(
        param_list=param_list, range=[(40, 120), 1, 1, 1, 1, 1, 1, 1], **corner_kwargs
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.set_title(ax.get_title(), fontsize=14)
    plt.savefig("{}/corner.png".format(savedir), dpi=250)

# make orbit plot
fig = HIP654_results.plot_orbits(
    num_epochs_to_plot=500, start_mjd=56000, plot_astrometry=False
)
radec_ax, sep_ax, pa_ax, cbar_ax = fig.axes

sep, serr, pa, paerr = (
    data_table["quant1"],
    data_table["quant1_err"],
    data_table["quant2"],
    data_table["quant2_err"],
)
epoch = Time(data_table["epoch"], format="mjd").decimalyear

sphere_mask = np.where(insts == "SPHERE")[0]
naco_mask = np.where(insts == "NACO")[0]
grav_mask = np.where(insts == "GRAVITY")[0]

gravity_ra, gravity_dec = sep[grav_mask], pa[grav_mask]
sphere_ra, sphere_dec = system.seppa2radec(sep[sphere_mask], pa[sphere_mask])
naco_ra, naco_dec = system.seppa2radec(sep[naco_mask], pa[naco_mask])

radec_ax.scatter(sphere_ra, sphere_dec, marker="o", color="hotpink", zorder=20, s=3)
radec_ax.scatter(naco_ra, naco_dec, marker="o", color="hotpink", zorder=20, s=3)
radec_ax.scatter(gravity_ra, gravity_dec, marker="o", color="hotpink", zorder=20, s=3)

gravity_sep, gravity_pa = system.radec2seppa(sep[grav_mask], pa[grav_mask])

gravity_raerr, gravity_decerr, gravity_corr = (
    data_table["quant1_err"][grav_mask],
    data_table["quant2_err"][grav_mask],
    data_table["quant12_corr"][grav_mask],
)

sep_ax.errorbar(
    epoch[sphere_mask],
    sep[sphere_mask],
    serr[sphere_mask],
    marker="^",
    color="purple",
    markeredgecolor="purple",
    markerfacecolor="white",
    ls="",
    label="SPHERE",
)
sep_ax.errorbar(
    epoch[naco_mask],
    sep[naco_mask],
    serr[naco_mask],
    marker="s",
    ls="",
    color="purple",
    label="NACO",
)
sep_ax.scatter(
    epoch[grav_mask], gravity_sep, label="GRAVITY", zorder=20, color="hotpink"
)
sep_ax.legend()

pa_ax.errorbar(
    epoch[sphere_mask],
    pa[sphere_mask],
    paerr[sphere_mask],
    marker="^",
    color="purple",
    markeredgecolor="purple",
    markerfacecolor="white",
    ls="",
)
pa_ax.errorbar(
    epoch[naco_mask], pa[naco_mask], paerr[naco_mask], marker="s", ls="", color="purple"
)
pa_ax.scatter(epoch[grav_mask], gravity_pa, zorder=20, color="hotpink")

l, b, w, h = cbar_ax.get_position().bounds
# cbar_ax.set_position([l - 0.05, b, w, h])
cbar_ax.set_position([l - 0.17, b, w, h])

l, b, w, h = pa_ax.get_position().bounds


def confidence_ellipse(
    x, y, corr, x_unc, y_unc, ax, n_std=3.0, facecolor="hotpink", alpha=1
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    (shamelessly stolen from matplotlib docs)
    """

    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + corr)
    ell_radius_y = np.sqrt(1 - corr)
    ellipse = patches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        alpha=alpha,
        zorder=20,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = x_unc * n_std
    mean_x = x

    # calculating the standard deviation of y ...
    scale_y = y_unc * n_std
    mean_y = y

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


fig.subplots_adjust(right=0.6)
radec_ax.set_position([0.05, b, w - 0.03, 0.77])
pa_ax.set_position([l - 0.23, b, w - 0.06, h])
sep_ax.set_position([l - 0.23, b + 0.42, w - 0.06, h])

for i in np.arange(len(grav_mask)):
    if i > 1:
        grav_ax = fig.add_axes([0.82, b, 0.1, h])
    else:
        grav_ax = fig.add_axes([0.68, b + 0.42 * i, 0.1, h])
    for n_std in [1, 2]:
        ellipse = confidence_ellipse(
            gravity_ra[i],
            gravity_dec[i],
            gravity_corr[i],
            gravity_raerr[i],
            gravity_decerr[i],
            grav_ax,
            n_std=n_std,
            alpha=1 - n_std / 4,
        )

    grav_ax.set_xlim(gravity_ra[i] + 0.3, gravity_ra[i] - 0.3)
    grav_ax.set_ylim(gravity_dec[i] - 0.5, gravity_dec[i] + 0.5)
    grav_ax.set_aspect("equal")

    if i == 0 or i == 2:
        grav_ax.set_xlabel("$\Delta$RA [mas]")
    if i < 2:
        grav_ax.set_ylabel("$\Delta$Dec [mas]")

    for j in np.arange(len(sep_ax.lines) - 2):
        orbittracks_sep = sep_ax.lines[j].get_ydata()
        orbittracks_pa = pa_ax.lines[j].get_ydata()
        ra2plot, dec2plot = system.seppa2radec(orbittracks_sep, orbittracks_pa)
        grav_ax.plot(ra2plot, dec2plot, color="lightgrey")
        grav_ax.text(
            gravity_ra[i] + 0.27, gravity_dec[i] + 0.4, "Epoch {}".format(i + 1)
        )

pa_ax.set_xlabel("Epoch [year]")
radec_ax.set_aspect("equal")
plt.savefig("{}/orbit.png".format(savedir), dpi=250)
