import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

flavor_texs = ["e", r"\mu", r"\tau"]
fs = 15

fname = "data/Analytic_Validate.txt"
dataf = open(fname, "r")
a, b, g31, m1, L, gamma = [float(x) for x in dataf.readline().split()]
dataf.close()
a = int(a)
b = int(b)

dts = ["Ef", "numerical", "analytic"]
dt = [(d, "f") for d in dts]
data = np.loadtxt(fname, dtype = dt, skiprows = 1)

lss = ["", "-", "--"]
for i in range(1, len(dts)):
	d = dts[i]
	label = r"${\rm %s}$" % d
	plt.plot(data["Ef"], data[d], label = label, ls = lss[i])

plt.xscale("log")

v = list(plt.axis())
v[0] = data["Ef"][0]
v[1] = data["Ef"][-1]
v[2] = 0
v[3] = 0.4
plt.axis(v)

plt.gca().xaxis.set_minor_locator(LogLocator(subs = "all", numticks = 1e5))
plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.05), minor = True)

plt.xlabel(r"$E_f{\rm\ [GeV]}$")
plt.ylabel(r"$\bar P_{%s %s}^{\rm reg}$" % (flavor_texs[a], flavor_texs[b]))

s = r"${\rm Visible}$"
s += "\n"
if m1 == 0:
	s += r"$m_1=0$"
else:
	s += r"$m_1=%g{\rm\ eV}$" % m1
s += "\n"
#s += "$,$ "
Lexp = np.floor(np.log10(L))
Lb = L / 10 ** Lexp
#Lstr = r"%.2g\times10^{%i}" % (Lb, Lexp)
Lstr = r"10^{%i}" % Lexp # round to order of magnitude, Lb=1.4 anyway
s += r"$L=%s{\rm\ km}$" % Lstr
s += "\n"
s += r"$\gamma=%g$" % gamma
s += "$,$ "
s += r"$g^{(\prime)}_{31}="
p = np.log10(g31)
if int(p) == p:
	s += r"10^{%i}" % p
else:
	s += r"%.1g\times10^{%i}" % (g / (10 ** p), p)
s += "$"
plt.text(0.99, 0.99, s, ha = "right", va = "top", fontsize = fs, transform = plt.gca().transAxes)

# IceCube ROI
fc = "lightblue"
alpha = 0.4
plt.fill_between([1e4, 1e7], [v[2], v[2]], [v[3], v[3]], facecolor = fc, edgecolor = None, alpha = alpha, zorder = -10)
plt.fill_between(v[:2], [v[3] + 1, v[3] + 1], [v[3] + 2, v[3] + 2], facecolor = fc, edgecolor = "k", lw = 0.5, alpha = alpha)

plt.legend(fontsize = fs, loc = 7)

plt.savefig("fig/Analytic_Validate.pdf")

