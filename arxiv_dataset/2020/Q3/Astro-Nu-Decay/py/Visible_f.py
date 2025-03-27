import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

flavor_texs = ["e", r"\mu", r"\tau"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fs = 15

fname = "data/Visible_f.txt"
dataf = open(fname, "r")
l = [float(x) for x in dataf.readline().split()]
g = l[0]
z = l[1]
m1 = l[2]
gamma = l[3]
n_fi = int(l[4])
n_ff = int(l[5])
fis = []
ffs = []
for i in range(n_fi):
	fis.append(int(l[6 + i]))
for i in range(n_ff):
	ffs.append(int(l[6 + n_fi + i]))
dataf.close()

dts = ["Ef"]
for i in range(n_fi):
	for j in range(n_ff):
		dts.append("SM_%i_%i" % (fis[i], ffs[j]))
		dts.append(r"%s %s" % (flavor_texs[fis[i]], flavor_texs[ffs[j]]))
dt = [(d, "f") for d in dts]
data = np.loadtxt(fname, dtype = dt, skiprows = 1)

j = 0
for i in range(1, len(dts)):
	d = dts[i]
	print(i, d)
	if "SM" in d:
		ls = "--"
		lw = 1
		c = colors[j]
		label = None
		zo = -2
	else:
		ls = "-"
		lw = 2
		c = colors[j]
		j += 1
		label = r"$%s$" % d
		zo = +2
		if i in [2, 4, 6, 8, 10, 12]:
			ix = 93
		else:
			ix = 7
		if i in [2, 4, 6, 12]:
			va = "top"
		else:
			va = "bottom"
		plt.text(data["Ef"][ix], data[d][ix], label, color = c, fontsize = fs, ha = "center", va = va)
	plt.plot(data["Ef"], data[d], label = label, ls = ls, lw = lw, color = c, zorder = zo)

plt.xscale("log")

v = list(plt.axis())
v[0] = data["Ef"][0]
v[1] = data["Ef"][-1]
v[2] = 0
v[3] = 0.15
plt.axis(v)

plt.gca().xaxis.set_minor_locator(LogLocator(subs = "all", numticks = 1e5))
plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.005), minor = True)

plt.xlabel(r"$E_f{\rm\ [GeV]}$")
plt.ylabel(r"$\bar P_{\alpha\beta}^{\rm vis}$")
s = r"${\rm Visible}$"
plt.text(0.99, 0.99, s, ha = "right", va = "top", fontsize = fs, transform = plt.gca().transAxes)

# IceCube ROI
fc = "lightblue"
alpha = 0.4
plt.fill_between([1e4, 1e7], [v[2], v[2]], [v[3], v[3]], facecolor = fc, edgecolor = None, alpha = alpha, zorder = -10)
plt.fill_between(v[:2], [v[3] + 1, v[3] + 1], [v[3] + 2, v[3] + 2], facecolor = fc, edgecolor = "k", lw = 0.5, alpha = alpha)

if m1 == 0:
	s = r"$m_1=0$"
else:
	s = r"$m_1=%g{\rm\ eV}$" % m1
s += "$,$ "
s += r"$z=%g$" % z
s += "\n"
s += r"$\gamma=%g$" % gamma
s += "$,$ "
s += r"$g^{(\prime)}_{ij}="
p = np.log10(g)
if int(p) == p:
	s += r"10^{%i}" % p
else:
	s += r"%.1g\times10^{%i}" % (g / (10 ** p), p)
s += "$"
plt.text(8e8, 0.002, s, fontsize = fs, ha = "right", va = "bottom")

plt.savefig("fig/Visible_f.pdf")

