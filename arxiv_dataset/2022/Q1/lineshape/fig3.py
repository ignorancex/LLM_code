"""Script to generate Figure 3."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lib

# Coordinates of the Metcalf Science Center of Boston University:
lambda_lab = np.deg2rad(42.3484) # Latitude
phi_lab = np.deg2rad(-71.1002) # Longitude

matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(9, 5))
ax = fig.add_axes([0, 0, 1, 1])
plt.subplots_adjust(wspace=0.3, hspace=0.35)


###########################################################
# Panel (a):

time = np.linspace(0, 365*24, 365*24*10) # 365-day period
v_lab, _ = lib.get_vlab(lambda_lab, phi_lab, time)

plt.subplot(2, 2, 1)
plt.plot(time/24., v_lab, '-k')
plt.xlim(0, 365)
plt.xticks([0, 30, 58, 89, 119, 150, 180, 211, 242, 272, 303, 333],
           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
           rotation=60, ha="left")
plt.text(0.05, 0.865, "(a)", transform=ax.transAxes, fontsize=12)
plt.ylabel(r"$v_{\mathrm{lab}}~[\mathrm{km}/\mathrm{s}]$", fontsize=11)


###########################################################
# Panel (b):

time = np.linspace(0, 24, 500) # 24-hour period
v_lab, alphas = lib.get_vlab(lambda_lab, phi_lab, time)

plt.subplot(2, 2, 2)
plt.plot([0, 24], [0, 0], '-k', linewidth=0.7, alpha=0.3)
plt.plot(time, np.cos(alphas[0]), '-r')
plt.plot(time, np.cos(alphas[1]), '--b')
plt.plot(time, np.cos(alphas[2]), '-.g')

plt.xlim(0, 24)
plt.ylim(-1, 1.04)
plt.xticks([0, 6, 12, 18, 24])
plt.text(0.485, 0.865, "(b)", transform=ax.transAxes, fontsize=12)
plt.xlabel(r"Time [hour]")
plt.ylabel(r"$\cos{\alpha}$", fontsize=11)


###########################################################
# Panel (c):

# Total power in the parallel gradient case:
P = lib.get_power(v_lab, alphas, case="par")

plt.subplot(2, 2, 3)
plt.plot(time, P[0], '-r')
plt.plot(time, P[1], '--b')
plt.plot(time, P[2], '-.g')

plt.xlim(0, 24)
plt.ylim(0, 1.02)
plt.xticks([0, 6, 12, 18, 24])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.text(0.05, 0.42, "(c)", transform=ax.transAxes, fontsize=12)
plt.xlabel(r"Time [hour]")
plt.ylabel(r"$P_{\parallel}~[\mathrm{a.\,u.}]$")


###########################################################
# Panel (d):

# Total power in the perpendicular gradient case:
P = lib.get_power(v_lab, alphas, case="perp")

plt.subplot(2, 2, 4)
plt.plot(time, P[0], '-r', label=r"North")
plt.plot(time, P[1], '--b', label=r"West")
plt.plot(time, P[2], '-.g', label=r"Zenith")

plt.xlim(0, 24)
plt.ylim(0, 1.02)
plt.xticks([0, 6, 12, 18, 24])
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.text(0.485, 0.42, "(d)", transform=ax.transAxes, fontsize=12)
plt.legend(loc="lower right")
plt.xlabel(r"Time [hour]")
plt.ylabel(r"$P_{\perp}~[\mathrm{a.\,u.}]$")


# Save the figure to a pdf file:
plt.savefig("fig3.pdf", bbox_inches="tight")

