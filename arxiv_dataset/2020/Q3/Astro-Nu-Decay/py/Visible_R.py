import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

flavor_texs = ["e", r"\mu", r"\tau"]
fs = 15

plt.rc("text.latex", preamble = r"\usepackage{amsmath}")

fname = "data/Visible_R.txt"
dataf = open(fname, "r")
a, b, g, m1, gamma = [float(x) for x in dataf.readline().split()]
dataf.close()
a = int(a)
b = int(b)

dts = ["Ef", r"\rm SM", r"\delta(z-0.5)", r"\boldsymbol{\delta(z-1)}",  r"\delta(z-1.5)", r"\rm YKMH", r"\rm HERMES"]
dt = [(d, "f") for d in dts]
data = np.loadtxt(fname, dtype = dt, skiprows = 1)

def f(xs, A, B):
	return A * xs ** B
	return A + B * xs
	return A * np.exp(B * xs)

mask = (data["Ef"] > 1e5) & (data["Ef"] < 1e6)
pivot = 42
print("Pivot is at E =", data["Ef"][pivot])
for i in range(1, len(dts)):
	d = dts[i]
	label = r"$%s$" % d
	plt.plot(data["Ef"], data[d] / data[d][pivot], label = label)

plt.xscale("log")

v = list(plt.axis())
v[0] = data["Ef"][0]
v[1] = data["Ef"][-1]
v[2] = 0.0
v[3] = 2.00
plt.axis(v)

plt.gca().xaxis.set_minor_locator(LogLocator(subs = "all", numticks = 1e5))
plt.gca().set_yticks(np.arange(v[2], v[3] + 1e-5, 0.10), minor = True)

plt.xlabel(r"$E_f{\rm\ [GeV]}$")
plt.ylabel(r"$\bar P_{%s %s}^{\rm vis}(E_f){\rm\ [Rescaled]}$" % (flavor_texs[a], flavor_texs[b]))

s = r"${\rm Visible}$"
s += "\n"
if m1 == 0:
	s += r"$m_1=0$"
else:
	s += r"$m_1=%g{\rm\ eV}$" % m1
s += "$,$ "
s += r"$\gamma=%g$" % gamma
s += "\n"
s += r"$g^{(\prime)}_{ij}="
p = np.log10(g)
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

plt.legend(fontsize = fs, loc = 4)
plt.legend(fontsize = fs, loc = (0.67, -0.01))

plt.savefig("fig/Visible_R.pdf")

