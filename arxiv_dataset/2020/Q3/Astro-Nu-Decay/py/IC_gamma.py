import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fs = 15

fname = "data/IC_gamma.txt"
dataf = open(fname, "r")
m1, gamma, z = [float(x) for x in dataf.readline().split()]
dataf.close()

dts = ["x", "track inv", "cascade inv", "track vis", "cascade vis"]
dt = [(d, "f") for d in dts]
data = np.loadtxt(fname, dtype = dt, skiprows = 1)

print("Max Delta gamma inv =", np.max(data["cascade inv"] - data["track inv"]))
print("Min Delta gamma inv =", np.min(data["cascade inv"] - data["track inv"]))
print("Max Delta gamma vis =", np.max(data["cascade vis"] - data["track vis"]))
print("Min Delta gamma vis =", np.min(data["cascade vis"] - data["track vis"]))

for i in range(1, len(dts)):
	d = dts[i]
	if "track" in d:	c = colors[0]
	else:				c = colors[1]
	if "inv" in d:		ls = "--"
	else:				ls = "-"
	plt.plot(data["x"], data[d], color = c, ls = ls)

plt.xscale("log")

v = list(plt.axis())
v[0] = 1e-8
v[1] = 1e-5
v[2] = 1
v[3] = 2.5
plt.axis(v)

plt.plot(v[:2], [gamma, gamma], color = colors[2], label = r"${\rm SM\ (Cascade\ \&\ Track)}$", zorder = -5)

plt.plot([v[0] - 1], [v[2] - 1], color = colors[1], label = r"${\rm Cascade}$")
plt.plot([v[0] - 1], [v[2] - 1], color = colors[0], label = r"${\rm Track}$")
plt.plot([v[0] - 1], [v[2] - 1], color = "k", ls = "-", label = r"${\rm Visible}$")
plt.plot([v[0] - 1], [v[2] - 1], color = "k", ls = "--", label = r"${\rm Invisible}$")

plt.gca().xaxis.set_minor_locator(LogLocator(subs = "all", numticks = 1e5))
plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.50), minor = False)
plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.10), minor = True)

plt.xlabel(r"$g^{(\prime)}_{ij}$")
plt.ylabel(r"$\gamma_f$")

s = ""
if m1 == 0:
	s += r"$m_1=0$"
else:
	s += r"$m_1=%g{\rm\ eV}$" % m1
s += "\n"
s += r"$z=%g$" % z
s += "\n"
s += r"$\gamma_i=%g$" % gamma
plt.text(0.99, 0.01, s, ha = "right", va = "bottom", fontsize = fs, transform = plt.gca().transAxes)

plt.legend(fontsize = fs, loc = 2, ncol = 2)

plt.savefig("fig/IC_gamma.pdf")

