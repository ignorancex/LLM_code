import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

taum_min = -12.
taum_max = 12.
def taum2x(taum):
	m = 2. / (taum_max - taum_min)
	return m * (taum - taum_min) - 1.

axis_color = "k"

v = [-1.1, 1.1, -0.16, 0.34]
plt.axis(v)
plt.xticks([],[])
plt.yticks([],[])
plt.axis("off")
plt.gca().set_aspect("equal")
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# draw arrows
hw = 0.020
hl = 0.02
c = axis_color
plt.arrow(0, 0, 1.05, 0, head_width = hw, head_length = hl, length_includes_head = True, ec = c, fc = c)
plt.arrow(0, 0, -1.05, 0, head_width = hw, head_length = hl, length_includes_head = True, ec = c, fc = c)

# draw ticks
fs = 12
tick_size = 0.015
taums = np.arange(taum_min, taum_max + 0.001, 1.)
taum_labels = np.arange(-12, taum_max + 0.001, 4.)
for taum in taums:
	x = taum2x(taum)
	plt.plot([x, x], [-tick_size, tick_size], c = c, ls = "-")
	if taum in taum_labels:
		plt.text(x, -0.025, r"$10^{%i}$" % taum, ha = "center", va = "top", fontsize = fs, color = c)

# axis label
plt.text(0, v[2], r"$\tau/m{\rm\ [s/eV]}$", ha = "center", va = "bottom", fontsize = fs, color = c)

count = 0

exp_fs = 9
dy = 0.05
def exclude(taum_, c, text, text_x = None):
	global count
	count += 1
	y = dy * count
	z = -1 - count

	x_min = -1.05 + hl
	x_max = taum2x(taum_)
	plt.fill_between([x_min, x_max], [y, y], edgecolor = "none", facecolor = c, zorder = z)

	ha = "left"
	if text_x == None:
		text_x = x_max - 0.01
		ha = "right"
	plt.text(text_x, y, text, ha = ha, va = "bottom", color = c, fontsize = exp_fs)

exclude(10.3, "m", r"${\rm CMB\ (}\nu_i\rm)$") # 95% 1907.05425 approximate
exclude(5., "r", r"${\rm SN1987A\ (}\bar\nu_e\rm)$", taum2x(2.19) + 0.01) # SK Hirata:1987hu
exclude(1., "g", r"${\rm IceCube\ (}\nu_i\rm)$") # 1610.02096
exclude(-3., "y", r"${\rm Solar\ (}\nu_2\rm)$") # SNO and others 99% 1812.01088
exclude(-10.1, "gray", r"${\rm Atm+LBL\ (}\nu_3\rm)$", -1.05 + hl) # 1506.02314 95%

x_min = taum2x(1.53)
x_max = taum2x(2.19)
c = "b"
y = dy * (count + 1.5)
plt.fill_between([x_min, x_max], [y, y], edgecolor = "none", facecolor = c, zorder = +1)
s = r"${\rm IceCube\ (}\nu_2,\nu_3\rm)$"
s += "\n"
s += r"${\rm Tracks\ \&\ Cascades}$"
plt.text(x_max + 0.01, y, s, ha = "left", va = "top", color = c, fontsize = exp_fs)

plt.axis(v)
plt.savefig("fig/Schematic.pdf", pad_inches = 0)

