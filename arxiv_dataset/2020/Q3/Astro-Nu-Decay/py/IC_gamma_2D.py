import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

fs = 15

# 1807.06209
# CMB+BAO
# 95%
summnumax = 0.12
# nu-fit v4.1, NO, w/o SK-atm
# http://www.nu-fit.org/sites/default/files/v41.tbl-parameters.pdf
Dmsq21 = 7.39e-5
Dmsq31 = +2.523e-3
s12sq = 0.31
s13sq = 0.02241

def f(m1, s):
	return m1 + np.sqrt(Dmsq21 + m1 ** 2) + np.sqrt(Dmsq31 + m1 ** 2) - s
m1max = fsolve(lambda m1: f(m1, summnumax), 0.1 * summnumax)[0]
print("Planck 95% current =", m1max)

c12sq = 1 - s12sq
c13sq = 1 - s13sq
Ue1sq = c12sq * c13sq
Ue2sq = s12sq * c13sq
Ue3sq = s13sq
# 1909.06048
# KATRIN
# 90%
mnumax = 1.1
m1sqmax = mnumax ** 2 - (Dmsq21 * Ue2sq + Dmsq31 * Ue3sq)
print("Katrin 90% current =", np.sqrt(m1sqmax))
# KATRIN future
mnumax = 0.2
m1sqmax = mnumax ** 2 - (Dmsq21 * Ue2sq + Dmsq31 * Ue3sq)
print("Katrin 90% future =", np.sqrt(m1sqmax))

# 1607.08006
# 1510.05223 page 59
# IC data
Delta_gamma_IC = 2.67 - 2.13
print("IC Delta gamma =", Delta_gamma_IC)

def p(vis):
	if vis:	name = "IC_gamma_2D_vis"
	else:	name = "IC_gamma_2D_inv"
	dataf = open("data/" + name + ".txt", "r")
	gamma, z = [float(x) for x in dataf.readline().split()]
	xs = [float(x) for x in dataf.readline().split()]
	ys = [float(x) for x in dataf.readline().split()]

	Delta_gammas = []
	for line in dataf.readlines():
		Delta_gammas.append([float(x) for x in line.split()])
	dataf.close()
	Delta_gammas = np.asarray(Delta_gammas).transpose()
	if vis:	print("Max delta gamma from vis =", Delta_gammas.max())
	else:	print("Max delta gamma from inv =", Delta_gammas.max())

	Xs, Ys = np.meshgrid(xs, ys)

	zmax = 0.5
	levels = np.arange(0, zmax + 1e-10, 1e-3)
	levels[0] = -1e-8
	c = plt.contourf(Xs, Ys, Delta_gammas, levels = levels, cmap = plt.get_cmap("Oranges"), vmin = 0, vmax = zmax, zorder = -10)
	plt.gca().set_rasterization_zorder(-5)

	step = 1e-1
	levels = np.arange(0, zmax + 1e-10, step)
	levels[0] = -1e-8
	c2 = plt.contour(Xs, Ys, Delta_gammas, levels = levels, cmap = plt.get_cmap("Oranges"), linewidths = 0.8, vmin = -step, vmax = zmax - step)
	if Delta_gamma_IC < Delta_gammas.max():
		c3 = plt.contour(Xs, Ys, Delta_gammas, levels = [Delta_gamma_IC], colors = "k", linewidths = 1)

	plt.xscale("log")
	plt.yscale("log")

	cbar = plt.colorbar(c, ticks = np.arange(0, 0.5 + 1e-10, 1e-1))
	cbar.ax.set_ylabel(r"$\Delta\gamma_f$")
	cbar.add_lines(c2)

	plt.xlabel(r"$g^{(\prime)}_{ij}$")
	plt.ylabel(r"$m_1{\rm\ [eV]}$")

	lw = 0.5
	zo = -1
	plt.plot([xs[0], xs[-1]], [m1max, m1max], ":", c = "purple", lw = lw, zorder = zo)
	plt.plot([xs[0], xs[-1]], [np.sqrt(m1sqmax), np.sqrt(m1sqmax)], ":", c = "gray", lw = lw, zorder = zo)

	v = list(plt.axis())
	plt.axis(v)

	if vis:	s = r"${\rm Visible}$"
	else:	s = r"${\rm Invisible}$"
	s += "\n"
	s += r"$z=%g$" % z
	s += "\n"
	s += r"$\gamma=%g$" % gamma
	plt.text(0.99, 0.99, s, ha = "right", va = "top", fontsize = fs, transform = plt.gca().transAxes)

	plt.savefig("fig/" + name + ".pdf", dpi = 200)
	plt.clf()

p(True)
p(False)

