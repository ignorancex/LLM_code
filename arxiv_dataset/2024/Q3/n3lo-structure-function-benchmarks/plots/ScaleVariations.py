import numpy as np
import matplotlib.pyplot as plt
import MatplotlibSettings

pdfs0 = np.loadtxt("F2NC_Scale_Variations_N0LO.dat")
pdfs1 = np.loadtxt("F2NC_Scale_Variations_N1LO.dat")
pdfs2 = np.loadtxt("F2NC_Scale_Variations_N2LO.dat")
pdfs3 = np.loadtxt("F2NC_Scale_Variations_N3LO.dat")

###############################################################
nd = 101
Q = [2, 5, 10, 50, 100]

###############################################################
for n in range(4):
    #f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [1, 1, 1, 1]))
    f, (ax2, ax3, ax4) = plt.subplots(3, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [1, 1, 1]))
    plt.subplots_adjust(wspace = 0, hspace = 0)

    #ax1.set_title(r"\textbf{$F_2^{\rm NC}(x_{\rm B},Q^2)$, ratio to $\mu_{\rm R} = \mu_{\rm F} = Q = " + str(Q[n]) + "$ GeV}", fontsize = 20, x = 0.5, y = 1.05)

    #ax1.set_ylabel(r"\textbf{LO}", fontsize = 20)
    #ax1.set_xlim([0.00001, 1])
    #ax1.set_ylim([0.96, 1.04])
    #ax1.set_xscale("log")
    #ax1.plot(pdfs0[n*nd:(n+1)*nd:,0], pdfs0[n*nd:(n+1)*nd:,2] / pdfs0[n*nd:(n+1)*nd:,2], color = "black", lw = 1.5, ls = "--")
    #ax1.fill_between(pdfs0[n*nd:(n+1)*nd:,0], pdfs0[n*nd:(n+1)*nd:,4] / pdfs0[n*nd:(n+1)*nd:,2], pdfs0[n*nd:(n+1)*nd:,6] / pdfs0[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm R}/Q\in[0.5:2]$}", color = "red", alpha = 0.4)
    #ax1.fill_between(pdfs0[n*nd:(n+1)*nd:,0], pdfs0[n*nd:(n+1)*nd:,8] / pdfs0[n*nd:(n+1)*nd:,2], pdfs0[n*nd:(n+1)*nd:,10] / pdfs0[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm F}/Q\in[0.5:2]$}", color = "blue", alpha = 0.4)

    ax2.set_title(r"\textbf{$F_2^{\rm NC}(x_{\rm B},Q^2)$, ratio to $\mu_{\rm R} = \mu_{\rm F} = Q = " + str(Q[n]) + "$ GeV}", fontsize = 20, x = 0.5, y = 1.05)

    ax2.set_ylabel(r"\textbf{NLO}", fontsize = 20)
    ax2.set_xlim([0.00001, 1])
    ax2.set_ylim([0.96, 1.04])
    ax2.set_xscale("log")
    ax2.plot(pdfs1[n*nd:(n+1)*nd:,0], pdfs1[n*nd:(n+1)*nd:,2] / pdfs1[n*nd:(n+1)*nd:,2], color = "black", lw = 1.5, ls = "--")
    ax2.fill_between(pdfs1[n*nd:(n+1)*nd:,0], pdfs1[n*nd:(n+1)*nd:,4] / pdfs1[n*nd:(n+1)*nd:,2], pdfs1[n*nd:(n+1)*nd:,6] / pdfs1[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm R}/Q\in[0.5:2]$}", color = "red", alpha = 0.4)
    ax2.fill_between(pdfs1[n*nd:(n+1)*nd:,0], pdfs1[n*nd:(n+1)*nd:,8] / pdfs1[n*nd:(n+1)*nd:,2], pdfs1[n*nd:(n+1)*nd:,10] / pdfs1[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm F}/Q\in[0.5:2]$}", color = "blue", alpha = 0.4)

    ax3.set_ylabel(r"\textbf{NNLO}", fontsize = 20)
    ax3.set_xlim([0.00001, 1])
    ax3.set_ylim([0.96, 1.04])
    ax3.set_xscale("log")
    ax3.plot(pdfs2[n*nd:(n+1)*nd:,0], pdfs2[n*nd:(n+1)*nd:,2] / pdfs2[n*nd:(n+1)*nd:,2], color = "black", lw = 1.5, ls = "--")
    ax3.fill_between(pdfs2[n*nd:(n+1)*nd:,0], pdfs2[n*nd:(n+1)*nd:,4] / pdfs2[n*nd:(n+1)*nd:,2], pdfs2[n*nd:(n+1)*nd:,6] / pdfs2[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm R}/Q\in[0.5:2]$}", color = "red", alpha = 0.4)
    ax3.fill_between(pdfs2[n*nd:(n+1)*nd:,0], pdfs2[n*nd:(n+1)*nd:,8] / pdfs2[n*nd:(n+1)*nd:,2], pdfs2[n*nd:(n+1)*nd:,10] / pdfs2[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm F}/Q\in[0.5:2]$}", color = "blue", alpha = 0.4)

    ax4.set_ylabel(r"\textbf{N$^3$LO}", fontsize = 20)
    ax4.set_xlabel(r"\textbf{$x_{\rm B}$}", fontsize = 20)
    ax4.set_xlim([0.00001, 1])
    ax4.set_ylim([0.96, 1.04])
    ax4.set_xscale("log")
    ax4.plot(pdfs3[n*nd:(n+1)*nd:,0], pdfs3[n*nd:(n+1)*nd:,2] / pdfs3[n*nd:(n+1)*nd:,2], color = "black", lw = 1.5, ls = "--")
    ax4.fill_between(pdfs3[n*nd:(n+1)*nd:,0], pdfs3[n*nd:(n+1)*nd:,4] / pdfs3[n*nd:(n+1)*nd:,2], pdfs3[n*nd:(n+1)*nd:,6] / pdfs3[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm R}/Q\in[0.5:2]$}", color = "red", alpha = 0.4)
    ax4.fill_between(pdfs3[n*nd:(n+1)*nd:,0], pdfs3[n*nd:(n+1)*nd:,8] / pdfs3[n*nd:(n+1)*nd:,2], pdfs3[n*nd:(n+1)*nd:,10] / pdfs3[n*nd:(n+1)*nd:,2], label = r"\textbf{$\mu_{\rm F}/Q\in[0.5:2]$}", color = "blue", alpha = 0.4)
    ax4.legend(fontsize = 18, loc = "lower left", ncols = 2)

    plt.savefig("F2NC_Scale_Variations_Q_" + str(Q[n]) + "_GeV.pdf")
    plt.close()

