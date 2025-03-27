import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

fs = 15

IC_chisq = 11.0138 # at gamma = 2.51

# plot for visible or invisible cases
def p(vis):
	if vis:	print("Visible")
	else:	print("Invisible")

	if vis:	name = "Chisq_vis"
	else:	name = "Chisq_inv"
	name += "_nom1min"
	dataf = open("data/" + name + ".txt", "r")
	z, = [float(x) for x in dataf.readline().split()]
	xs = [float(x) for x in dataf.readline().split()]
	ys = [float(x) for x in dataf.readline().split()]

	Chisqs = []
	for line in dataf.readlines():
		Chisqs.append([float(x) for x in line.split()])
	dataf.close()
	Chisqs = np.asarray(Chisqs).transpose()
	if vis:	print("Min chisq from vis =", Chisqs.min())
	else:	print("Min chisq from inv =", Chisqs.min())

	Chisqs = IC_chisq - Chisqs

	# label the best fit point
	indices = np.where(Chisqs == np.amax(Chisqs))
	plt.plot(xs[indices[1][0]], ys[indices[0][0]], "*", color = "peru")
	print("Best fit point is at: g =", xs[indices[1][0]], "and gamma =", ys[indices[0][0]], "with chisq =", np.amax(Chisqs))

	Xs, Ys = np.meshgrid(xs, ys)

	zmax = 11
	levels = np.arange(0, zmax + 1e-10, 1e-2)
	levels[0] = -1e-8
	cmap = plt.get_cmap("Blues")
	c = plt.contourf(Xs, Ys, Chisqs, levels = levels, cmap = cmap, vmin = 0, vmax = zmax, zorder = -10)
	plt.gca().set_rasterization_zorder(-5)

	step = 2
	levels = np.arange(0, zmax + 1e-10, step)
	levels[0] = -1e-8
#	c2 = plt.contour(Xs, Ys, Chisqs, levels = levels, cmap = cmap, linewidths = 0.8, vmin = -step, vmax = zmax - step)

	plt.xscale("log")

	cbar = plt.colorbar(c, ticks = np.arange(0, 11 + 1e-10, 1))
	cbar.ax.set_ylabel(r"$\chi^2_{\rm SM}-\chi^2_{\rm Decay}$")
#	cbar.add_lines(c2)

	plt.xlabel(r"$g^{(\prime)}_{ij}$")
	plt.ylabel(r"$\gamma$")

	v = list(plt.axis())
	plt.axis(v)

	plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.50))
	plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.10), minor = True)

	if vis:	s = r"${\rm Visible}$"
	else:	s = r"${\rm Invisible}$"
	s += "\n"
	s += r"$z=%g$" % z
	plt.text(0.99, 0.99, s, ha = "right", va = "top", fontsize = fs, transform = plt.gca().transAxes)

	plt.savefig("fig/" + name + ".pdf", dpi = 200)
	plt.clf()

p(True)
p(False)

