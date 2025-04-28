import numpy as np
import matplotlib.pyplot as plt
import MatplotlibSettings

pdfs = np.loadtxt("StructureFunctions_N2LO.dat")

###############################################################
nd = 501
Q = [2, 5, 10, 50, 100]
c = ["red", "blue", "orange", "green", "magenta"]

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_1^{\rm NC}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 1000000])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,2], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,2] / pdfs[0:nd:,2], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,2] / pdfs[n*nd:(n+1)*nd:,3], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F1NC_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_2^{\rm NC}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 100])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,4], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,4] / pdfs[0:nd:,4], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,4] / pdfs[n*nd:(n+1)*nd:,5], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F2NC_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_3^{\rm NC}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00000005, 1000])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,6], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,6] / pdfs[0:nd:,6], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,6] / pdfs[n*nd:(n+1)*nd:,7], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F3NC_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_1^{W^+}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 1000000])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,8], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,8] / pdfs[0:nd:,8], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,8] / pdfs[n*nd:(n+1)*nd:,9], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F1CCp_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_2^{W^+}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 1000000])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,10], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,10] / pdfs[0:nd:,10], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,10] / pdfs[n*nd:(n+1)*nd:,11], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F2CCp_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_3^{W^+}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 100])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,12], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,12] / pdfs[0:nd:,12], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,12] / pdfs[n*nd:(n+1)*nd:,13], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F3CCp_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_1^{W^-}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 50])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,14], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,14] / pdfs[0:nd:,14], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,14] / pdfs[n*nd:(n+1)*nd:,15], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F1CCm_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_2^{W^-}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 1000000])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,16], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,16] / pdfs[0:nd:,16], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,16] / pdfs[n*nd:(n+1)*nd:,17], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F2CCm_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{NNLO {(NNLL evolution)}}", fontsize = 20)
ax1.set(ylabel = r"$F_3^{W^-}(x_{\rm B},Q^2)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([-5600, 1000])
ax1.set_xscale("log")
#ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,18], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower right", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,18] / pdfs[0:nd:,18], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,18] / pdfs[n*nd:(n+1)*nd:,19], color = c[n], lw = 1.5, ls = "--")

plt.savefig("F3CCm_N2LO.pdf")
plt.close()

Q = [2, 5, 10, 50, 100]

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{N$^2$LO (NNLL evolution), $\sqrt{s}=320$ GeV}", fontsize = 20)
ax1.set(ylabel = r"$\sigma_{\rm NC, red}^{+}(x_{\rm B},Q^2,s)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.0002, 2])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,20], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,20] / pdfs[0:nd:,20], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,20] / pdfs[n*nd:(n+1)*nd:,21], color = c[n], lw = 1.5, ls = "--")

plt.savefig("RedXSecNCp_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{N$^2$LO (NNLL evolution), $\sqrt{s}=320$ GeV}", fontsize = 20)
ax1.set(ylabel = r"$\sigma_{\rm NC, red}^{-}(x_{\rm B},Q^2,s)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.0002, 2])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,22], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,22] / pdfs[0:nd:,22], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,22] / pdfs[n*nd:(n+1)*nd:,23], color = c[n], lw = 1.5, ls = "--")

plt.savefig("RedXSecNCm_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{N$^2$LO (NNLL evolution), $\sqrt{s}=320$ GeV}", fontsize = 20)
ax1.set(ylabel = r"$\sigma_{\rm CC, red}^{+}(x_{\rm B},Q^2,s)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 10])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,24], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,24] / pdfs[0:nd:,24], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,24] / pdfs[n*nd:(n+1)*nd:,25], color = c[n], lw = 1.5, ls = "--")

plt.savefig("RedXSecCCp_N2LO.pdf")
plt.close()

###############################################################
f, (ax1, ax2) = plt.subplots(2, 1, sharex = "all", gridspec_kw = dict(width_ratios = [1], height_ratios = [3, 1]))
plt.subplots_adjust(wspace = 0, hspace = 0)

ax1.set_title(r"\textbf{N$^2$LO (NNLL evolution), $\sqrt{s}=320$ GeV}", fontsize = 20)
ax1.set(ylabel = r"$\sigma_{\rm CC, red}^{-}(x_{\rm B},Q^2,s)$")
ax1.set_xlim([0.00001, 1])
ax1.set_ylim([0.00005, 10])
ax1.set_xscale("log")
ax1.set_yscale("log")
for n in range(len(Q)):
    ax1.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,26], color = c[n], lw = 1.5, ls = "-", label = r"\textbf{$Q = " + str(Q[n]) + "$ GeV}")
ax1.legend(fontsize = 20, loc = "lower left", ncol = 2)

ax2.set(xlabel = r"\textbf{$x_{\rm B}$}")
ax2.set_ylabel(r"\textbf{HOPPET / APFEL++}", fontsize = 16)
ax2.set_xlim([0.00001, 1])
ax2.set_ylim([0.99995, 1.00005])
ax2.set_xscale("log")
ax2.plot(pdfs[0:nd:,0], pdfs[0:nd:,26] / pdfs[0:nd:,26], color = "black", lw = 1, ls = "--")
for n in range(len(Q)):
    ax2.plot(pdfs[n*nd:(n+1)*nd:,0], pdfs[n*nd:(n+1)*nd:,26] / pdfs[n*nd:(n+1)*nd:,27], color = c[n], lw = 1.5, ls = "--")

plt.savefig("RedXSecCCm_N2LO.pdf")
plt.close()
