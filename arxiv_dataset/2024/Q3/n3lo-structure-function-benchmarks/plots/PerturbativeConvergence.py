import numpy as np
import matplotlib.pyplot as plt
import MatplotlibSettings

pdfs0 = np.loadtxt("StructureFunctions_N0LO.dat")
pdfs1 = np.loadtxt("StructureFunctions_N1LO.dat")
pdfs2 = np.loadtxt("StructureFunctions_N2LO.dat")
pdfs3 = np.loadtxt("StructureFunctions_N3LO.dat")

###############################################################
nd = 501
Q = [2, 5, 10, 50, 100]
nsf = 4 # F2NC (4)

###############################################################
for n in range(4):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
    plt.subplots_adjust(wspace = 0, hspace = 0)

    ax1.set_title(r"\textbf{$Q = " + str(Q[n]) + "$ GeV}", fontsize = 20)
    ax1.set(ylabel = r"$F_2^{\rm NC}(x_{\rm B},Q^2)$")
    ax1.set_xlim([0.00001, 1])
    ax1.set_ylim([0.0002, 1.2 * np.amax(pdfs3[n*nd:(n+1)*nd:,nsf])])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(pdfs0[n*nd:(n+1)*nd:,0], pdfs0[n*nd:(n+1)*nd:,nsf], color = "red",    lw = 1.5, ls = "-", label = r"\textbf{LO}")
    ax1.plot(pdfs1[n*nd:(n+1)*nd:,0], pdfs1[n*nd:(n+1)*nd:,nsf], color = "blue",   lw = 1.5, ls = "-", label = r"\textbf{NLO}")
    ax1.plot(pdfs2[n*nd:(n+1)*nd:,0], pdfs2[n*nd:(n+1)*nd:,nsf], color = "orange", lw = 1.5, ls = "-", label = r"\textbf{NNLO}")
    ax1.plot(pdfs3[n*nd:(n+1)*nd:,0], pdfs3[n*nd:(n+1)*nd:,nsf], color = "green",  lw = 1.5, ls = "-", label = r"\textbf{N$^3$LO}")
    ax1.legend(fontsize = 20, loc = "lower left")

    ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
    ax2.set_ylabel(r"\textbf{Ration to N$^3$LO}", fontsize = 16)
    ax2.set_xlim([0.00001, 1])
    ax2.set_ylim([0.85, 1.15])
    ax2.set_xscale("log")
    ax2.plot(pdfs3[0:nd:,0], pdfs3[0:nd:,nsf] / pdfs3[0:nd:,nsf], color = "black", lw = 1, ls = "--")
    ax2.plot(pdfs3[n*nd:(n+1)*nd:,0], pdfs0[n*nd:(n+1)*nd:,nsf] / pdfs3[n*nd:(n+1)*nd:,nsf], color = "red",    lw = 1.5, ls = "--")
    ax2.plot(pdfs3[n*nd:(n+1)*nd:,0], pdfs1[n*nd:(n+1)*nd:,nsf] / pdfs3[n*nd:(n+1)*nd:,nsf], color = "blue",   lw = 1.5, ls = "--")
    ax2.plot(pdfs3[n*nd:(n+1)*nd:,0], pdfs2[n*nd:(n+1)*nd:,nsf] / pdfs3[n*nd:(n+1)*nd:,nsf], color = "orange", lw = 1.5, ls = "--")

    plt.savefig("PerturbativeConvergence_Q_" + str(Q[n]) + "_GeV.pdf")
    plt.close()
