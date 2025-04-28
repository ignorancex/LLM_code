import numpy as np
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.patches as matpatches
import matplotlib
matplotlib.rc('font', **{'size': 22})
from pathlib import Path
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import scipy.integrate as integrate
from scipy import stats

COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a", "#e6ab02", "#a6761d", "#816481", "#E08502", "#8E7368", "#666666"]

# ---
# WFIT Loading and prepping
# ---

"""
Load in wfit directories from pippin output
"""
def get_wfit_dirs(pippin_output):
    wfit_dir = Path(pippin_output) / "8_COSMOFIT" / "WFIT"
    wfit_dirs = list(wfit_dir.iterdir())
    return wfit_dirs

"""
Read in all pertinent files from a wfit directory
"""
def get_wfit_files(wfit_dir):
    out_dir = wfit_dir / "output" / "CHI2GRID"
    wfit_files = list(out_dir.iterdir())
    return wfit_files

"""
Collect and store important parameters from a wfit file
"""
def read_wfit(wfit_file):
    chi2 = np.loadtxt(wfit_file, dtype=str)
    Om = np.array([float(i[3]) for i in chi2[1:]])
    w0 = np.array([float(i[2]) for i in chi2[1:]])
    weight = np.array([float(i[5]) for i in chi2[1:]])
    weight = np.exp(-0.5 * weight)
    ind = np.argmax(weight)
    best_Om = Om[ind]
    best_w0 = w0[ind]
    best_weight = weight[ind]
    data = [Om, w0, weight]
    best = [best_Om, best_w0, best_weight]
    return wfit_file, data, best

"""
Essentially transpose from a dict of realisations, into lists of each parameter
"""
def unpack_dict(wfit_dict):
    Om_list = np.hstack([l[0] for l in [wfit_dict[k]["data"] for k in wfit_dict.keys()]])
    w0_list = np.hstack([l[1] for l in [wfit_dict[k]["data"] for k in wfit_dict.keys()]])
    weight_list = np.hstack([l[2] for l in [wfit_dict[k]["data"] for k in wfit_dict.keys()]])
    best_Om_list = np.hstack([l[0] for l in [wfit_dict[k]["best"] for k in wfit_dict.keys()]])
    best_w0_list = np.hstack([l[1] for l in [wfit_dict[k]["best"] for k in wfit_dict.keys()]])
    best_weight_list = np.hstack([l[2] for l in [wfit_dict[k]["best"] for k in wfit_dict.keys()]])
    return Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list

"""
Get the specific realisation with realisation closest to the mean Om and w0
"""
def get_closest(wfit_dict):
    Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
    Om = 0.3
    w0 = -1.0
    dist = np.sqrt(np.power(best_Om_list - Om, 2) + np.power(best_w0_list - w0, 2))
    ind = np.argmin(dist)
    print(f"Closest ind: {ind}")
    best_file = list(wfit_dict.keys())[ind]
    return read_wfit(best_file)

# ---
# Plotting
# ---

"""
Plot the pippin contour and best fit for the experiment cosmology
"""
def plot_contour(wfits, options, output, logger):
    logger.info("Plotting contour")
    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
    c = ChainConsumer()
    c.add_chain([data[0], data[1]], name="Nominal Cosmology", parameters=[r"$\Omega_{m}$", r"$w$"], weights=data[2], grid=True, shade_alpha=0.2)
    c.configure(label_font_size=22, tick_font_size=22, contour_label_font_size=22, colors=[COLOURS[-1]], shade=True, shade_alpha=0.2, bar_shade=True)
    extents = options.get("extents")
    fig = c.plotter.plot(figsize=(12, 10), extents=extents)
    ax = fig.get_axes()[2]
    ax.scatter([best[0]], [best[1]], label=r"$\Omega_{M}^{best}$, $w^{best}$", c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
    if "OMEGA_MATTER" in options.keys():
        Om_list = options["OMEGA_MATTER"]
        w0_list = options["W0_LAMBDA"]
        for i in range(len(Om_list)):
            n = (i % 4) + 1
            set_n = int(np.ceil((i + 0.5) / 4))
            if set_n == 1:
                marker = "X"
            else:
                marker = "P"
            ax.scatter([Om_list[i]], [w0_list[i]], label=r"$\Omega_{M}'$, $w'$" + f" {n}-{set_n}", c=COLOURS[i], marker=marker, s=100, edgecolor="black", zorder=5, linewidth=1)
    ax.legend(ncol=3, loc="lower left", columnspacing=0.8)
    plt.savefig(output / f"{options.get('name', 'Contour')}.svg")
    plt.close()

"""
Get a trained gaussian process for a distribution of best-fitting cosmologies
"""
def get_GP(wfit_dict):
    Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
    X = best_Om_list.reshape(-1, 1)
    Y = best_w0_list
    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, Y)
    return gp

"""
Calculate a GP for each distribution of best-fitting cosmologies, and subtract that from w.
Additionally, apply this offset to each (Om, w0) point in points.
"""
def apply_GP(wfits, points=None):
    rtn = {"FR": {}, "V": {}}
    k = list(wfits["FR"].keys())[0]
    wfit_dict = wfits["FR"][k]
    Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
    gp = get_GP(wfit_dict)
    w0_correction = gp.predict(best_Om_list.reshape(-1,1), return_std=False)
    best_w0_list -= w0_correction
    if points is not None:
        ps = []
        for point in points:
            p_corr = gp.predict(np.array(point[0]).reshape(-1, 1), return_std=False)[0]
            p = [point[0], point[1] - p_corr]
            ps.append(p)
        rtn["FR"][k] = (best_Om_list, best_w0_list, ps)
    else:
        rtn["FR"][k] = (best_Om_list, best_w0_list)
    for (i, k) in enumerate(wfits["V"].keys()):
        wfit_dict = wfits["V"][k]
        Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
        gp = get_GP(wfit_dict)
        w0_correction = gp.predict(best_Om_list.reshape(-1,1), return_std=False)
        best_w0_list -= w0_correction
        if points is not None:
            ps = []
            for point in points:
                p_corr = gp.predict(np.array(point[0]).reshape(-1, 1), return_std=False)[0]
                p = [point[0], point[1] - p_corr] 
                ps.append(p)
            rtn["V"][k] = (best_Om_list, best_w0_list, ps)
        else:
            rtn["V"][k] = (best_Om_list, best_w0_list)
    return rtn

def bootstrap_ellipse(x, y, ellipse, num_resamples = 1000):
    inds = range(len(x))
    resamples = np.random.choice(inds, (num_resamples, len(inds)), replace=True)
    r_x = x[resamples]
    r_y = y[resamples]
    ps = []
    def get_p(a, b):
        num_points = 0
        for i in range(len(a)):
            if ellipse.contains_point((a[i], b[i])):
                num_points += 1
        return round(100 * num_points / len(a))
    for i in range(len(r_x)):
        xx = r_x[i]
        yy = r_y[i]
        ps.append(get_p(xx, yy))
    return ps

"""
Iteratively scale ellipse to intersect point 
"""
def confidence_ellipse(x, y, point, Om_init, w0_init, logger):
    x = np.array(x)
    y = np.array(y)
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none')
    std_low = 0
    std_high = 10
    loop = True
    num_iter = 0
    while loop:
        std = 0.5 * (std_high + std_low)
        scale_x = np.sqrt(cov[0, 0]) * std
        mean_x = Om_init
        scale_y = np.sqrt(cov[1, 1]) * std
        mean_y = w0_init
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf)
        bbox = ellipse.get_extents().get_points()
        xmin = bbox[0][0]
        xmax = bbox[1][0]
        ymin = bbox[0][1]
        ymax = bbox[1][1]
        xpoint = point[0]
        ypoint = point[1]
        #contains_point = (xmin < xpoint < xmax) and (ymin < ypoint < ymax)
        contains_point = ellipse.contains_point((xpoint, ypoint))
        if contains_point:
            std_high = std
        else:
            std_low = std
        num_points_inside = 0
        for i in range(len(x)):
            if ellipse.contains_point((x[i], y[i])):
                num_points_inside += 1
        if std_high - std_low < 0.001:
            if contains_point:
                loop = False 
        num_iter += 1
        if num_iter > 100:
            loop = False
    scale_x = np.sqrt(cov[0, 0]) * std
    mean_x = Om_init
    scale_y = np.sqrt(cov[1, 1]) * std
    mean_y = w0_init
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    return ellipse, transf, round(100 * num_points_inside / len(x))

"""
Find the likelihood / confidence ellipse for each set of realisations
"""
def calculate_ellipse(wfits, points, logger):
    print(points)
    rtn = {"FR": {}, "V": {}}
    gp = apply_GP(wfits, points)
    k = list(wfits["FR"].keys())[0]
    rtn["FR"][k] = None 
    ks = sorted(list(wfits["V"].keys()))
    for (i, k) in enumerate(ks):
        Om, w0, (point, (Om_init, w0_init)) = gp["V"][k]
        rtn["V"][k] = []
        rtn["V"][k] = confidence_ellipse(Om, w0, point, Om_init, w0_init, logger)
    return rtn

"""
Plot an example of calculating the percentile contour at a given point in cosmology.
"""
def plot_percentile(wfits, options, output, logger):
    logger.info("Plotting percentile")
    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
    c = ChainConsumer()
    c.add_chain([data[0], data[1]], name="Nominal Cosmology", parameters=[r"$\Omega_{m}$", r"$w$"], weights=data[2], grid=True, shade_alpha=0.2)
    c.configure(label_font_size=22, tick_font_size=22, contour_label_font_size=22, colors=[COLOURS[-1]], shade=True, shade_alpha=0.2, bar_shade=True)
    extents = options.get("extents")
    fig = c.plotter.plot(figsize=(12, 10), extents=extents)
    ax = fig.get_axes()[2]
    ax.scatter([best[0]], [best[1]], label=r"$\Omega_{M}^{best}$, $w^{best}$", c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
    Om = options.get("OMEGA_MATTER", None)
    w0 = options.get("W0_LAMBDA", None)
    if not Om is None:
        ax.scatter([Om], [w0], label=r"68\% $\Omega_{M}'$, $w'$", c="black", marker="D", s=100, edgecolor="black", zorder=5, linewidth=1)

    ks = sorted(list(wfits["V"].keys()))
    validation = [i for i in options["validation"]]
    for i in validation:
        k = ks[i]
        n = (i % 4) + 1
        set_n = int(np.ceil((i + 0.5) / 4))
        if set_n == 1:
            marker = "o"
        elif set_n == 2:
            marker = "s"
        else:
            marker = "D"
        logger.info(f"i: {i}, k: {k}")
        _, _, point = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
        key = list(wfits["V"][k].keys())[0]
        Om_init = wfits["V"][k][key]["Om"]
        w0_init = wfits["V"][k][key]["w0"]
        if set_n == 1:
            marker = "X"
        else:
            marker = "P"
        ax.scatter([Om_init], [w0_init], label=r"$\Omega_{M}'$, $w'$" + f" {n}-{set_n}", c=COLOURS[i], marker=marker, s=100, edgecolor="black", zorder=5, linewidth=1)
        ell = calculate_ellipse(wfits, (point, (Om_init, w0_init)), logger)
        wfit_dict = wfits["V"][k]
        gp = get_GP(wfit_dict)
        Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
        ellipse, transf, percent = ell["V"][k]
        ellipse.set_transform(transf)
        path = ellipse.get_path()
        transform = ellipse.get_transform()
        path = transform.transform_path(path)
        verts = path.vertices
        x = verts[:,0]
        y = verts[:,1]
        w0_correction = gp.predict(x.reshape(-1,1), return_std=False)
        y += w0_correction
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        ax.plot(x, y, label=f"Coverage ellipse", c=COLOURS[i])
        ax.scatter(best_Om_list, best_w0_list, c=COLOURS[i], label=r"$\Omega_{M}'$, $w'$" f" {n}-{set_n} best fits", s=10, marker=marker, alpha=0.5)
    ax.legend()
    plt.savefig(output / f"{options.get('name', 'Percentile')}.svg")
    plt.close()

"""
Plot two panels, one showing the distribution of best-fitting cosmologies and another showing the GP-transformed elliptical fit
"""
def plot_GPE(wfits, options, output, logger):
    logger.info("Plotting GPE")
    fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained", sharex=True, figsize=(12, 8))
    edgecolor = 'firebrick'
    linestyle = '--'
    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])

    ks = sorted(list(wfits["V"].keys()))
    validation = [i for i in options["validation"]]
    for i in validation:
        k = ks[i]
        n = (i % 4) + 1
        set_n = int(np.ceil((i + 0.5) / 4))
        if set_n == 1:
            marker = "o"
        elif set_n == 2:
            marker = "s"
        else:
            marker = "D"
        logger.info(f"i: {i}, k: {k}")
        wfit_dict = wfits["V"][k]
        gp = get_GP(wfit_dict)
        Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
        x = np.linspace(best_Om_list.min(), best_Om_list.max(), 100)
        y, dy = gp.predict(x.reshape(-1, 1), return_std=True)
        ax1.scatter(best_Om_list, best_w0_list, c=COLOURS[i], label=r"$\Omega_{M}'$, $w'$" + f" best fits {n}-{set_n}", s=10, marker=marker, alpha=0.5)
        ax1.errorbar(x, y, yerr=dy, c=COLOURS[i], label="GP fit")
        ax1.scatter([best[0]], [best[1]], label=r"$\Omega_{M}^{best}$, $w^{best}$", c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
        ax1.set_ylabel(r"$w$")

        _, _, point = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
        key = list(wfits["V"][k].keys())[0]
        Om_init = wfits["V"][k][key]["Om"]
        w0_init = wfits["V"][k][key]["w0"]
        ax1.scatter([Om_init], [w0_init], label=r"$\Omega_{M}'$, $w'$" + f" {n}-{set_n}", c=COLOURS[i], marker="X", s=100, edgecolor="black", zorder=5, linewidth=1)
        gp = apply_GP(wfits, (point, (Om_init, w0_init)))
        ell = calculate_ellipse(wfits, (point, (Om_init, w0_init)), logger)
        Om, w0, (point, (Om_init, w0_init)) = gp["V"][k]
        ellipse = ell["V"][k]
        ax2.scatter(Om, w0, c=COLOURS[i], label=r"$\Omega_{M}'$, $w'$" + f" best fits", s=10, marker=marker, alpha=0.5)
        e, transf, p = ellipse
        e.set_transform(transf + ax2.transData)
        e.set_edgecolor(COLOURS[i])
        #e.set_linestyle(linestyle)
        e.set_label("Coverage ellipse")
        ax2.scatter([point[0]], [point[1]], label="$\Omega_{M}^{best}$, $w^{best}$", c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
        ax2.scatter([Om_init], [w0_init], label=r"$\Omega_{M}'$, $w'$" + f" {n}-{set_n}", c=COLOURS[i], marker="X", s=100, edgecolor="black", zorder=5, linewidth=1)
        ax2.add_patch(e)
        ax2.set_xlabel(r"$\Omega_{m}$")
        ax2.set_ylabel(r"$w - w^{*}(\Omega_{M})$")
    ax1.set_title("Gaussian Process Fit")
    ax2.set_title("Transformed Distribution")
    ax1.legend(loc="lower left")
    ax2.legend(loc="upper left")
    plt.savefig(output / f"{options.get('name', 'GPE')}.svg")
    plt.close()

def plot_ellipse(wfits, options, output, logger):
    logger.info("Plotting Ellipse")
    edgecolor = 'firebrick'
    linestyle = '--'
    validation = options["validation"]
    num_points = options.get("num", 4)
    num_rows = int(np.ceil(len(validation) / num_points))
    num_cols = num_points
    fig, axes = plt.subplots(num_rows, num_cols, layout="constrained", sharey=True, figsize=(num_rows * 8, num_cols * 4))
    Om_min = np.inf
    Om_max = -np.inf
    w0_min = np.inf
    w0_max = -np.inf
    ks = sorted(list(wfits["V"].keys()))
    for (i, v) in enumerate(validation):
        ax = axes.flatten()[i]
        k = ks[v]
        if v == "x":
            continue
        n = (i % 4) + 1
        set_n = int(np.ceil((i + 0.5) / 4))
        if set_n == 1:
            marker = "o"
        elif set_n == 2:
            marker = "s"
        else:
            marker = "D"
        _, _, point = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
        key = list(wfits["V"][k].keys())[0]
        Om_init = wfits["V"][k][key]["Om"]
        w0_init = wfits["V"][k][key]["w0"]
        ell = calculate_ellipse(wfits, (point, (Om_init, w0_init)), logger)
        gp = apply_GP(wfits, (point, (Om_init, w0_init)))
        Om, w0, (point, (Om_init, w0_init)) = gp["V"][k]
        ellipse = ell["V"][k]
        ax.scatter(Om, w0, s=10, c = COLOURS[i], marker=marker, alpha=0.5)
        ax.scatter([point[0]], [point[1]], c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
        if set_n == 1:
            marker = "X"
        else:
            marker = "P"
        ax.scatter([Om_init], [w0_init], c=COLOURS[i], marker=marker, s=100, edgecolor="black", zorder=5, linewidth=1)
        e, transf, p = ellipse
        bs = bootstrap_ellipse(Om, w0, e)
        ax.set_title(f"{n}-{set_n} ({round(np.mean(bs))}" + r"$\pm$" + f"{round(np.std(bs))}" + "%)")
        e.set_transform(transf + ax.transData)
        logger.info(f"Bootstrap mean {np.mean(bs)} and std {np.std(bs)}, true prob: {p}")
        e.set_edgecolor(COLOURS[i])
        ax.add_patch(e)
        scale = 0.1
        Om_min = min([Om_min, min(Om) - scale * abs(min(Om))])
        Om_max = max([Om_max, max(Om) + scale * abs(max(Om))])
        w0_min = min([w0_min, min(w0) - scale * abs(min(w0))])
        w0_max = max([w0_max, max(w0) + scale * abs(max(w0))])
    ax = axes.flatten()[0]
    ax.scatter([], [], c=COLOURS[-1], s=100, marker="s", label=r"$\Omega_{M}^{best}$, $w^{best}$", zorder=5)
    ax.scatter([], [], c="black", s=10, label=r"$\Omega_{M}'$, $w'$" + " best fits")
    ax.plot([], [], color="black", label="Coverage Ellipse")
    ax.legend(loc="upper left")
    for (i, ax) in enumerate(axes.flatten()):
        ax.set_ylim(w0_min, w0_max)
        ax.set_xlim(Om_min, Om_max)
        n = (i % 4) + 1
        set_n = int(np.ceil((i + 0.5) / 4))
        if n == 1:
            ax.set_ylabel(r"$w - w^{*}(\Omega_{M})$")
        ax.set_xlabel(r"$\Omega_{M}$")
    fig.suptitle("Coverage Ellipses")
    plt.savefig(output / f"{options.get('name', 'Ellipse')}.svg")
    plt.close()

"""
Plot the pippin contour and best fit for the experiment cosmology
"""
def plot_final(wfits, options, output, logger):
    logger.info("Plotting final")
    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
    c = ChainConsumer()
    c.add_chain([data[0], data[1]], name="Nominal Cosmology", parameters=[r"$\Omega_{m}$", r"$w$"], weights=data[2], grid=True, shade_alpha=0.2)
    c.configure(label_font_size=22, tick_font_size=22, contour_label_font_size=22, colors=[COLOURS[-1]], shade=True, shade_alpha=0.2, bar_shade=True)
    extents = options.get("extents")
    fig = c.plotter.plot(figsize=(12, 10), extents=extents)
    ax = fig.get_axes()[2]
    ax.scatter([best[0]], [best[1]], label=r"$\Omega_{M}^{best}$, $w^{best}$", c=COLOURS[-1], marker="s", s=100, edgecolor="black", zorder=5, linewidth=1)
    if "OMEGA_MATTER" in options.keys():
        Om_list = options["OMEGA_MATTER"]
        w0_list = options["W0_LAMBDA"]
        for i in range(len(Om_list)):
            n = (i % 4) + 1
            set_n = int(np.ceil((i + 0.5) / 4))
            marker = "D"
            ax.scatter([Om_list[i]], [w0_list[i]], label=r"$\Omega_{M}'$, $w'$" + f" {i + 1}", c=COLOURS[i], marker=marker, s=100, edgecolor="black", zorder=5, linewidth=1)
    ax.legend(ncol=3, loc="lower left", columnspacing=0.8)
    plt.savefig(output / f"{options.get('name', 'Final')}.svg")
    plt.close()

# ---
# Deprecated plotting
# ---
#def plot_nominal(wfits, options, output, logger):
#    logger.info("Plotting nominal")
#    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
#    c = ChainConsumer()
#    c.add_chain([data[0], data[1]], name="Nominal Cosmology", parameters=[r"$\Omega_{m}$", r"$w$"], weights=data[2], grid=True, shade_alpha=0.2)
#    c.configure(label_font_size=22, tick_font_size=22, contour_label_font_size=22, colors=[COLOURS[-1]], shade=True, shade_alpha=0.2, bar_shade=True)
#    extents = options.get("extents")
#    fig = c.plotter.plot(figsize=(12, 10), extents=extents)
#    ax = fig.get_axes()[2]
#    ks = sorted(list(wfits["V"].keys()))
#    for (i, k) in enumerate(ks):
#        if i > 0:
#            break
#        logger.info(f"i: {i}, k: {k}")
#        n = (i % 4) + 1
#        set_n = int(np.ceil((i + 0.5) / 4))
#        key = list(wfits["V"][k].keys())[0]
#        Om = wfits["V"][k][key]["Om"]
#        w0 = wfits["V"][k][key]["w0"]
#        logger.info(f"Om: {Om}, w0: {w0}")
#        if set_n == 1:
#            marker = "o"
#        else:
#            marker = "s"
#        ax.scatter([Om], [w0], label=f"{n}-{set_n}", c=COLOURS[i], marker=marker)
#    Om = options.get("Om", [])
#    w0 = options.get("w0", [])
#    for i in range(len(Om)):
#        ax.scatter([Om[i]], [w0[i]], label=f"Proposal Cosmology {i+1} Input", c=COLOURS[i])
#    ax.scatter([best[0]], [best[1]], label="Experiment", c=COLOURS[-1])
#    ax.legend(ncol=2, loc="lower left", columnspacing=0.8)
#    plt.savefig(output / "Nominal.svg")
#    fig.suptitle("Neyman Input Cosmologies")
#    plt.close()

#def plot_all(wfits, options, output, logger):
#    logger.info("Plotting all together")
#    f, data, best = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
#    c = ChainConsumer()
#    c.add_chain([data[0], data[1]], name="Nominal Cosmology", parameters=[r"$\Omega_{m}$", r"$w$"], weights=data[2], grid=True, shade_alpha=0.2)
#    c.configure(label_font_size=22, tick_font_size=22, contour_label_font_size=22, colors=[COLOURS[-1]], shade=True, shade_alpha=0.2, bar_shade=True)
#    extents = options.get("extents")
#    fig = c.plotter.plot(figsize=(12, 10), extents=extents)
#    ax = fig.get_axes()[2]
#    ax.scatter([best[0]], [best[1]], label="Nominal Cosmology Input", c=COLOURS[-1])
#    ax.scatter([], [], marker="x", label="Best Fit Cosmology", c="black", s=1)
#    Om = options.get("OMEGA_MATTER", [])
#    w0 = options.get("W0_LAMBDA", [])
#    ks = sorted(list(wfits["V"].keys()))
#    for (i, k) in enumerate(ks):
#        n = (i % 4) + 1
#        set_n = int(np.ceil((i + 0.5) / 4))
#        #if i > 3:
#        #    continue
#        logger.info(f"i: {i}, k: {k}")
#        _, _, point = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
#        key = list(wfits["V"][k].keys())[0]
#        Om_init = wfits["V"][k][key]["Om"]
#        w0_init = wfits["V"][k][key]["w0"]
#        ell = calculate_ellipse(wfits, (point, (Om_init, w0_init)), logger)
#        wfit_dict = wfits["V"][k]
#        gp = get_GP(wfit_dict)
#        Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
#        ellipse, transf, percent = ell["V"][k]
#        ellipse.set_transform(transf)
#        path = ellipse.get_path()
#        transform = ellipse.get_transform()
#        path = transform.transform_path(path)
#        verts = path.vertices
#        x = verts[:,0]
#        y = verts[:,1]
#        w0_correction = gp.predict(x.reshape(-1,1), return_std=False)
#        y += w0_correction
#        x = np.append(x, x[0])
#        y = np.append(y, y[0])
#        print(f"{n}-{set_n} {percent}\% ellipse")
#        #ax.plot(x, y, label=f"{n}-{set_n} {percent}\% ellipse", c=COLOURS[i])
#        #ax.scatter(best_Om_list, best_w0_list, marker="x", c=COLOURS[i], s=1)
#    ax.legend()
#    fig.suptitle("Coverage Ellipses")
#    plt.savefig(output / "All.svg")
#    plt.close()
#
#def plot_comparison(wfits, options, output, logger):
#    logger.info("Plotting Comparison")
#    fig, ax = plt.subplots(1, 1, figsize=(12, 8)) 
#    k = list(wfits["FR"].keys())[0]
#    wfit_dict = wfits["FR"][k]
#    Om_list, w0_list, weight_list, best_Om_list, best_w0_list, best_weight_list = unpack_dict(wfit_dict)
#    f, data, best = get_closest(wfit_dict)
#    x = best[0]
#    y = best[1]
#    xerr = np.std(best_Om_list)
#    yerr = np.std(best_w0_list)
#    ax.errorbar(x, y, xerr=xerr, yerr=yerr, label="Nominal Cosmology", c=COLOURS[-1])
#    ks = sorted(list(wfits["V"].keys()))
#    for (i, k) in enumerate(ks):
#        logger.info(f"i: {i}, k: {k}")
#        _, _, point = get_closest(wfits["FR"][list(wfits["FR"].keys())[0]])
#        key = list(wfits["V"][k].keys())[0]
#        Om_init = wfits["V"][k][key]["Om"]
#        w0_init = wfits["V"][k][key]["w0"]
#        ell = calculate_ellipse(wfits, (point, (Om_init, w0_init)), logger)
#        wfit_dict = wfits["V"][k]
#        gp = get_GP(wfit_dict)
#        ellipse, transf, percent = ell["V"][k]
#        ellipse.set_transform(transf)
#        path = ellipse.get_path()
#        transform = ellipse.get_transform()
#        path = transform.transform_path(path)
#        verts = path.vertices
#        x = verts[:,0]
#        y = verts[:,1]
#        w0_correction = gp.predict(x.reshape(-1,1), return_std=False)
#        y += w0_correction
#        x = np.append(x, x[0])
#        y = np.append(y, y[0])
#        label = "Validation " + str(i + 1) + " - " + str(percent) + "\% coverage"
#        ax.plot(x, y, label=label, c=COLOURS[i])
#    fig.legend()
#    fig.suptitle("Comparison")
#    plt.xlabel("Om")
#    plt.ylabel("w0")
#    plt.savefig(output / "Comparison.svg")
#    plt.close()
