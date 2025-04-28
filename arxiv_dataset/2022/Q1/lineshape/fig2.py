"""Script to generate Figure 2."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lib
from lib import pi, v_0, v_lab

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 14

fig = plt.figure(figsize=(12, 7))
plt.subplots_adjust(wspace=0.25, hspace=0.32)


###########################################################
# Panel (a):

nu_a = 1
nu_values = np.linspace(nu_a + 1e-15, nu_a + 4e-6, 200)
x = (nu_values - nu_a)/nu_a

curves = []
curves.append(lib.get_lineshape(v_0, v_lab, nu_a, nu_values, case="non-grad"))
curves.append(lib.get_lineshape(v_0, v_lab, nu_a, nu_values, case="grad_par", alpha=0))
curves.append(lib.get_lineshape(v_0, v_lab, nu_a, nu_values, case="grad_par", alpha=pi/2))
curves.append(lib.get_lineshape(v_0, v_lab, nu_a, nu_values, case="grad_perp", alpha=pi/2))

ax = fig.add_subplot(2, 2, 1)
line0, = plt.plot(x, curves[0], ":k", lw=2) # Non-gradient case
line1, = plt.plot(x, curves[1], "-r", lw=2) # Parallel gradient case, alpha = 0
line2, = plt.plot(x, curves[2], "--b", lw=2) # Parallel gradient case, alpha = pi/2
line3, = plt.plot(x, curves[3], "-.g", lw=2)  # Perpendicular gradient case, alpha = pi/2

plt.legend([line0, line1, line2, line2, line3], [r"$\lambda$",
                                          r"$\lambda_{\parallel}, \, \alpha = 0$",
                                          r"$\lambda_{\parallel}, \, \alpha = \frac{\pi}{2}$",
                                          r"$\lambda_{\perp}, \, \alpha = 0$",
                                          r"$\lambda_{\perp}, \, \alpha = \frac{\pi}{2}$"])
plt.xlim(-1e-8, 4e-6)
plt.yticks([0, 5e5, 1e6])
plt.text(-0.19, 1.02, "(a)", fontsize=16, transform=ax.transAxes)
plt.xlabel(r"$\nu / \nu_a - 1$")
plt.ylabel(r"$\nu_a \, \lambda(\nu)$")


###########################################################
# Panel (b):

# Increase the speeds by 1000 to increase the linewidth:
v_0 = 1000*v_0
v_lab = 1000*v_lab

# ALP Compton frequency in Hz:
nu_a = 1000

# Sampling frequency:
nu_s = 10*nu_a

# Interrogation time in seconds:
T = 0.05

# Total number of axionlike particles:
N = 1000

# Number of averages:
N_averages = 500

# Set a random seed:
np.random.seed(592)

# Number of samples and time array:
N_samp = int(T*nu_s)
t = np.linspace(0, T, N_samp)

# Calculate averaged power spectra:
psds = np.zeros((5, int(N_samp/2)))

for i in range(N_averages):
    v_x, v_y, v_z, nu, phi = lib.get_random_distributions(v_0, v_lab, nu_a, N)
    
    s = np.zeros((5, len(t)))
    
    # Non-gradient case:
    s[0] = lib.get_signal(nu, phi, t)
    
    # Gradient case with alpha = 0:
    v_par, v_perp = lib.get_v_components(v_x, v_y, v_z, alpha=0)
    s[1] = lib.get_signal(nu, phi, t, v_par) # Parallel
    s[3] = lib.get_signal(nu, phi, t, v_perp) # Perpendicular
    
    # Gradient case with alpha = pi/2:
    v_par, v_perp = lib.get_v_components(v_x, v_y, v_z, alpha=pi/2)
    s[2] = lib.get_signal(nu, phi, t, v_par) # Parallel
    s[4] = lib.get_signal(nu, phi, t, v_perp) # Perpendicular
     
    for i in range(5):
        psds[i] += lib.calculate_psd(s[i], T)
    
psds /= N_averages

freqs = np.linspace(0, nu_s/2, int(T*nu_s/2))
nu_values = np.linspace(1.0001*nu_a, nu_s/2, 200)

curves = []
curves.append(lib.get_psd(v_0, v_lab, nu_a, nu_values, case="non-grad"))
curves.append(lib.get_psd(v_0, v_lab, nu_a, nu_values, case="grad_par", alpha=0))
curves.append(lib.get_psd(v_0, v_lab, nu_a, nu_values, case="grad_par", alpha=pi/2))
curves.append(lib.get_psd(v_0, v_lab, nu_a, nu_values, case="grad_perp", alpha=0))
curves.append(lib.get_psd(v_0, v_lab, nu_a, nu_values, case="grad_perp", alpha=pi/2))

ax = fig.add_subplot(2, 2, 2)
plt.plot(freqs/nu_a - 1, 1e4*psds[0], "-k", alpha=0.3, label=r"MC")
plt.plot(nu_values/nu_a - 1, 1e4*curves[0], ":k", lw=2, label=r"Eq. (14)")
plt.xticks([0, 1, 2, 3, 4])
plt.xlim(-0.01, 4)
plt.legend()
plt.text(-0.19, 1.02, "(b)", fontsize=16, transform=ax.transAxes)
plt.text(0, 1.02, r"$\times 10^{-4}$", transform=ax.transAxes)
plt.xlabel(r"$\nu / \nu_a - 1$")
plt.ylabel(r"$|S(\nu)|^2 / (\kappa a_0)^2~\left[\mathrm{Hz}^{-1}\right]$")


###########################################################
# Panel (c):

ax = fig.add_subplot(2, 2, 3)
plt.plot(freqs/nu_a - 1, 1e4*psds[1], "-r", alpha=0.3, label=r"MC, $\alpha = 0$")
plt.plot(freqs/nu_a - 1, 1e4*psds[2], "-b", alpha=0.3, label=r"MC, $\alpha = \frac{\pi}{2}$")
plt.plot(nu_values/nu_a - 1, 1e4*curves[1], "--r", lw=2, label=r"Eq. (21), $\alpha = 0$")
plt.plot(nu_values/nu_a - 1, 1e4*curves[2], ":b", lw=2, label=r"Eq. (21), $\alpha = \frac{\pi}{2}$")
plt.xticks([0, 1, 2, 3, 4])
plt.xlim(-0.01, 4)
plt.legend()
plt.text(-0.19, 1.02, "(c)", fontsize=16, transform=ax.transAxes)
plt.text(0, 1.02, r"$\times 10^{-4}$", transform=ax.transAxes)
plt.xlabel(r"$\nu / \nu_a - 1$")
plt.ylabel(r"$|S_{\parallel}(\nu)|^2 / (\kappa_{\parallel}^2 \rho_{\mathrm{DM}})~\left[\mathrm{Hz}^{-1}\right]$")


###########################################################
# Panel (d):

ax = fig.add_subplot(2, 2, 4)
plt.plot(freqs/nu_a - 1, 1e4*psds[3], "-r", alpha=0.3, label=r"MC, $\alpha = 0$")
plt.plot(freqs/nu_a - 1, 1e4*psds[4], "-b", alpha=0.3, label=r"MC, $\alpha = \frac{\pi}{2}$")
plt.plot(nu_values/nu_a - 1, 1e4*curves[3], "--r", lw=2, label=r"Eq. (22), $\alpha = 0$")
plt.plot(nu_values/nu_a - 1, 1e4*curves[4], ":b", lw=2, label=r"Eq. (22), $\alpha = \frac{\pi}{2}$")
plt.xticks([0, 1, 2, 3, 4])
plt.xlim(-0.01, 4)
plt.legend()
plt.text(-0.19, 1.02, "(d)", fontsize=16, transform=ax.transAxes)
plt.text(0, 1.02, r"$\times 10^{-4}$", transform=ax.transAxes)
plt.xlabel(r"$\nu / \nu_a - 1$")
plt.ylabel(r"$|S_{\perp}(\nu)|^2 / (\kappa_{\perp}^2 \rho_{\mathrm{DM}})~\left[\mathrm{Hz}^{-1}\right]$")


# Save the figure to a pdf file:
plt.savefig("fig2.pdf", bbox_inches="tight")

