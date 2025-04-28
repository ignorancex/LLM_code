#!/usr/bin/env python
from matplotlib.pyplot import xlabel, ylabel
from plotter_style import *
from os import system
from optparse import OptionParser
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import math


def main():

    # initialise latex
    initializetex()

    # color and line styles
    # # Get "Set2" and "Set1" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)
    set1 = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    set2 = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
    set3 = brewer2mpl.get_map('Set3', 'qualitative', 8).mpl_colors
    setBlue = brewer2mpl.get_map('Blues', 'sequential', 3).mpl_colors
    # setBlueExt = brewer2mpl.get_map('Blues', 'sequential', 5).mpl_colors
    # setOrange = brewer2mpl.get_map('Oranges', 'sequential', 3).mpl_colors
    # setOrangeExt = brewer2mpl.get_map('Oranges', 'sequential', 5).mpl_colors
    setRed = brewer2mpl.get_map('Reds', 'sequential', 3).mpl_colors
    # setRedExt = brewer2mpl.get_map('Reds', 'sequential', 5).mpl_colors
    setGreen = brewer2mpl.get_map('Greens', 'sequential', 3).mpl_colors
    # setGreenExt = brewer2mpl.get_map('Greens', 'sequential', 5).mpl_colors
    # set1light = palettable.tableau.TableauMedium_10.mpl_colors
    # set1dark = palettable.tableau.Tableau_10.mpl_colors
    # set1 = palettable.tableau.TableauMedium_10.mpl_colors
    hatch = ('/////////', '\\\\\\\\\\\\\\\\\\\\\\\\\\',
             '/////', '\\\\\\\\', None, None, None, None)
    facecolor = ('none', 'none', 'none', 'none',
                 set1[0], set1[1], set1[2], set1[3])
    facecolor2 = ('none', 'none', 'none', 'none',
                  set2[0], set2[1], set2[2], set2[3])
    alpha = (1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7)
    # alphaleg = (None,None,0.7,0.2)
    # color = '#FF8937'
    # alphas = (0.2, 0.5, 0.8)
    almost_black = '#262626'
    # light_grey = np.array([float(248)/float(255)]*3)
    dashedlines = [[1, 1], [4, 2, 4, 2], [
        100000, 1], [4, 1, 1, 1], [4, 2, 1, 2, 1, 2]]

    # dists = ["total", "y-LQ", "m-LQ", "pt-lep1", "pt-jet1", "eta-lep1", "eta-jet1"]
    # xlabels = ["", r"$y_{\rm LQ}$", r"$m_{\rm LQ}$", r"$p_{T}^{\ell_1}$", r"$p_{T}^{j_1}$", r"$\eta_{T}^{\ell_1}$" ,r"$\eta_{T}^{j_1}$"  ]
    # ylabels = [ r"$\sigma$", r"$d \sigma / d y_{\rm LQ}$" , r"$d \sigma / d m_{\rm LQ}$",r"$d \sigma / d p_{T}^{\ell_1} $",r"$d \sigma / d p_{T}^{j_1}$",r"$d \sigma / d \eta_{T}^{\ell_1}$",r"$d \sigma / d \eta_{T}^{j_1}$"  ]

   # dists = [ "y-LQ", "m-LQ", "pt-lep1", "pt-jet1", "eta-lep1", "eta-jet1"]
   # xlabels = [ r"$m_{\rm LQ}$", r"$p_{T}^{\ell_1}$", r"$p_{T}^{j_1}$", r"$\eta^{\ell_1}$" ,r"$\eta^{j_1}$"  ]
   # ylabels = [ r"$d \sigma / d m_{\rm LQ}$",r"$d \sigma / d p_{T}^{\ell_1} $",r"$d \sigma / d p_{T}^{j_1}$",r"$d \sigma / d \eta^{\ell_1}$",r"$d \sigma / d \eta^{j_1}$"  ]
   # dists = ["y-LQ","m-LQ", "pt-lep1", "pt-jet1", "eta-lep1", "eta-jet1","dphi","pt-sys","dr","y-LQrec","m-LQrec","pt-lep1rec","eta-lep1rec","pt-sysrec","pt-LQ"]
    dists = ["y-LQ","m-LQ", "pt-lep1", "pt-jet1", "eta-lep1", "eta-jet1","dphi","pt-sys","dr","pt-LQ","ptmiss","mcoll","pt-taujet"]
    xlabels = [ r"$y_{\ell j}$", 
          r"$m_{\ell j} \; [{\rm GeV}]$", 
          r"$p_{T}^{\ell} \; [{\rm GeV}]$", 
          r"$p_{T}^{j} \; [{\rm GeV}]$", 
          r"$\eta _{\ell} $",
          r"$\eta _{j} $",
          r"$\Delta\phi$",
          r"$p_{T}^{\rm sys} \; [\rm GeV]$",
          r"$dR$",
          r"$p_T ^{\ell j} \; [\rm GeV]$",
          r"$p_T ^{miss} \; [\rm GeV]$",
          r"$m_{coll} \; [{\rm GeV}]$",
          r"$p_T ^{\tau} \; [\rm GeV]$"]
    ylabels = [r"$d \sigma / d y_{\ell j} \; [{\rm pb}]$", 
          r"$d \sigma / d m_{\ell j} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d p_{T}^{\ell} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d p_{T}^{j} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d \eta _{\ell} \; [{\rm pb}]$",
          r"$d \sigma / d \eta _{j} \; [{\rm pb}]$",
          r"$d \sigma / d \Delta \phi \; [{\rm pb}]$",
          r"$d \sigma / d p_{T} ^{\rm sys} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d dR \; [{\rm pb}]$",
          r"$d \sigma / d p_T ^{\ell j} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d p_T ^{miss} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d m_{coll} \; [{\rm pb}/{\rm GeV}]$",
          r"$d \sigma / d p_T ^{\tau} \; [{\rm pb}/{\rm GeV}]$"]
    RUNDIRLO = "rundirnameLO"
    RUNDIRNLO = "rundirnameNLO"

    for (dist,xlabel,ylabel) in zip(dists,xlabels,ylabels):
    
        lowedge, highedge, LOLHE_c, LOLHE_c_err = np.loadtxt(
            f"../{RUNDIRLO}/pwgLHEFanalysis-output-11-{dist}.dat", unpack=True)
        dummy, dummy, LOLHE_max, LOLHE_max_err = np.loadtxt(
        f"../{RUNDIRLO}/pwgLHEFanalysis-output-max-{dist}.dat", unpack=True)
        dummy, dummy, LOLHE_min, LOLHE_min_err = np.loadtxt(
            f"../{RUNDIRLO}/pwgLHEFanalysis-output-min-{dist}.dat", unpack=True)
        mid = (lowedge + highedge)/2.
        if dist == "total":
            lowedge = np.insert(lowedge, 1
                , highedge, axis=0)
            LOLHE_max = np.insert(LOLHE_max, 1,
                LOLHE_max, axis=0)
            LOLHE_min = np.insert(LOLHE_min, 1,
                LOLHE_min, axis=0)
            LOLHE_c = np.insert(LOLHE_c, 1, LOLHE_c, axis=0)

        else:
            lowedge = np.insert(lowedge, len(
                lowedge), highedge[-1], axis=0)
            LOLHE_max = np.insert(LOLHE_max, len(
                LOLHE_max), LOLHE_max[-1], axis=0)
            LOLHE_min = np.insert(LOLHE_min, len(
                LOLHE_min), LOLHE_min[-1], axis=0)
            LOLHE_c = np.insert(LOLHE_c, len(LOLHE_c), LOLHE_c[-1], axis=0)


        lowedge, highedge, NLOLHE_c, NLOLHE_c_err = np.loadtxt(
            f"../{RUNDIRNLO}/pwgLHEFanalysis-output-11-{dist}.dat", unpack=True)
        dummy, dummy, NLOLHE_max, NLOLHE_max_err = np.loadtxt(
        f"../{RUNDIRNLO}/pwgLHEFanalysis-output-max-{dist}.dat", unpack=True)
        dummy, dummy, NLOLHE_min, NLOLHE_min_err = np.loadtxt(
            f"../{RUNDIRNLO}/pwgLHEFanalysis-output-min-{dist}.dat", unpack=True)
        mid = (lowedge + highedge)/2.
        if dist == "total":
            lowedge = np.insert(lowedge, 1
                , highedge, axis=0)
            NLOLHE_max = np.insert(NLOLHE_max, 1,
                NLOLHE_max, axis=0)
            NLOLHE_min = np.insert(NLOLHE_min, 1,
                NLOLHE_min, axis=0)
            NLOLHE_c = np.insert(NLOLHE_c, 1, NLOLHE_c, axis=0)

        else:
            lowedge = np.insert(lowedge, len(
                lowedge), highedge[-1], axis=0)
            NLOLHE_max = np.insert(NLOLHE_max, len(
                NLOLHE_max), NLOLHE_max[-1], axis=0)
            NLOLHE_min = np.insert(NLOLHE_min, len(
                NLOLHE_min), NLOLHE_min[-1], axis=0)
            NLOLHE_c = np.insert(NLOLHE_c, len(NLOLHE_c), NLOLHE_c[-1], axis=0)

            
        lowedge, highedge, LOPS_c, LOPS_c_err = np.loadtxt(
            f"../{RUNDIRLO}/pwg-POWHEG+HERWIG7-output-11-{dist}.dat", unpack=True)
        dum:250
        my, dummy, LOPS_max, LOPS_max_err = np.loadtxt(
            f"../{RUNDIRLO}/pwg-POWHEG+HERWIG7-output-max-{dist}.dat", unpack=True)
        dummy, dummy, LOPS_min, LOPS_min_err = np.loadtxt(
            f"../{RUNDIRLO}/pwg-POWHEG+HERWIG7-output-min-{dist}.dat", unpack=True)
#        mid = (lowedge_minlo + highedge_minlo)/2.
        if dist == "total":
            LOPS_max = np.insert(LOPS_max, 1,
                LOPS_max, axis=0)
            LOPS_min = np.insert(LOPS_min, 1,
                LOPS_min, axis=0)
            LOPS_c = np.insert(LOPS_c, 1, LOPS_c, axis=0)
        else:
            lowedge = np.insert(lowedge, len(
                lowedge), highedge[-1], axis=0)
            LOPS_max = np.insert(LOPS_max, len(
                LOPS_max), LOPS_max[-1], axis=0)
            LOPS_min = np.insert(LOPS_min, len(
                LOPS_min), LOPS_min[-1], axis=0)
            LOPS_c = np.insert(LOPS_c, len(LOPS_c), LOPS_c[-1], axis=0)


        lowedge, highedge, NLOPS_c, NLOPS_c_err = np.loadtxt(
            f"../{RUNDIRNLO}/pwg-POWHEG+HERWIG7-output-11-{dist}.dat", unpack=True)
        dummy, dummy, NLOPS_max, NLOPS_max_err = np.loadtxt(
            f"../{RUNDIRNLO}/pwg-POWHEG+HERWIG7-output-max-{dist}.dat", unpack=True)
        dummy, dummy, NLOPS_min, NLOPS_min_err = np.loadtxt(
            f"../{RUNDIRNLO}/pwg-POWHEG+HERWIG7-output-min-{dist}.dat", unpack=True)
#        mid = (lowedge_minlo + highedge_minlo)/2.
        if dist == "total":
            NLOPS_max = np.insert(NLOPS_max, 1,
                NLOPS_max, axis=0)
            NLOPS_min = np.insert(NLOPS_min, 1,
                NLOPS_min, axis=0)
            NLOPS_c = np.insert(NLOPS_c, 1, NLOPS_c, axis=0)
        else:
            lowedge = np.insert(lowedge, len(
                lowedge), highedge[-1], axis=0)
            NLOPS_max = np.insert(NLOPS_max, len(
                NLOPS_max), NLOPS_max[-1], axis=0)
            NLOPS_min = np.insert(NLOPS_min, len(
                NLOPS_min), NLOPS_min[-1], axis=0)
            NLOPS_c = np.insert(NLOPS_c, len(NLOPS_c), NLOPS_c[-1], axis=0)

            
        #LOLHE_c_err = np.insert(LOLHE_c_err , len(LOLHE_c_err),LOLHE_c_err[-1], axis=0)
        #LOPS_c_err = np.insert(LOPS_c_err , len(LOPS_c_err),LOPS_c_err[-1], axis=0)

        # set size
        fig_dims = []
        fig_dims = figsize(fig_dims)
        fig_dims = [4, 4]
        fig = plt.figure(figsize=fig_dims)

        # set outputfile and initialize pdfpages
        output = "LHE_vs_HERWIG_{}.pdf".format(dist)
        print("Plotting "+output+"...")
        pdf_pages = PdfPages(output)

        # upper and lower panel
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
        if dist in ["ptmiss","dphi","pt-LQ"] : gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
        gs.update(hspace=0.12)
        ax1 = fig.add_subplot(gs[0])

        # upper panel
        mystyle(ax1)
        mygrid(ax1)

        plt.ylabel(ylabel, labelpad=1.5)

        lwf = 0.4
        lwc=0.5
        norm = 1
        lines = []
        labels = [r"LO (LHE)", r"LO+PS (HW7)", r"NLO (LHE)", r"NLO+PS (HW7)"]
        if dist == "pt-LQ" : labels=[r"LO+PS (HW7)", r"NLO (LHE)", r"NLO+PS (HW7)"]
        if dist == "ptmiss" : labels=[r"LO+PS (HW7)", r"NLO+PS (HW7)"]
        if dist == "pt-taujet" : labels=[r"LO+PS (HW7)", r"NLO+PS (HW7)"]
        
        # Add process and mass of the LQ to the plot
        ax1.text(0,1.11,r"$b + \tau \rightarrow U_1$",
              horizontalalignment="left",
              verticalalignment="bottom",
              transform=ax1.transAxes)
        ax1.text(1,1.11,r"$m_{U_1} = 2 \, {\rm TeV}$",
              horizontalalignment="right",
              verticalalignment="bottom",
              transform=ax1.transAxes)
        ax1.text(1,1.19,r"$(\beta_L)_{b\tau} = (\beta_R)_{b\tau} = 1$",
              horizontalalignment="right",
              verticalalignment="bottom",
              transform=ax1.transAxes)
        ax1.text(1,1.03,r"$\sqrt{s} = 13 \, {\rm TeV}$",
              horizontalalignment="right",
              verticalalignment="bottom",
              transform=ax1.transAxes)

        if dist == "ptmiss" or dist == "pt-taujet": 
           LOLHE_c=np.zeros(len(lowedge))
           LOLHE_min=np.zeros(len(lowedge))
           LOLHE_max=np.zeros(len(lowedge))
           NLOLHE_c=np.zeros(len(lowedge))
           NLOLHE_min=np.zeros(len(lowedge))
           NLOLHE_max=np.zeros(len(lowedge))

        p1, = ax1.plot(lowedge, LOLHE_c, color=set1[1],
                       alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
        ax1.fill_between(lowedge, LOLHE_min, LOLHE_max, facecolor=facecolor[3],
                         linewidth=lwf, edgecolor=set1[1], hatch=hatch[0], alpha=alpha[1], zorder=99, step="post")
        p2, = [plt.Rectangle((0, 0), 0, 0, facecolor=facecolor[1],
                             linewidth=lwf, edgecolor=set1[1], hatch=hatch[0], alpha=None)]
        if (not dist == "pt-LQ") and (not dist == "ptmiss") :
           lines += ((p1, p2),)

        p1, = ax1.plot(lowedge, LOPS_c, color=set2[1],
                       alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
        ax1.fill_between(lowedge, LOPS_min, LOPS_max, facecolor=facecolor2[1],
                         linewidth=lwf, edgecolor=set2[1], hatch=hatch[1], alpha=alpha[0], zorder=99, step="post")
        p2, = [plt.Rectangle((0, 0), 0, 0, facecolor=facecolor[3],
                             linewidth=lwf, edgecolor=set2[1], hatch=hatch[1], alpha=None)]
        lines += ((p1, p2),)

        ax1.fill_between(lowedge, NLOLHE_min, NLOLHE_max, facecolor=set2[2],
                         linewidth=lwc, edgecolor=set2[2], hatch=hatch[7], alpha=alpha[5], zorder=99, step="post")
        p1, = ax1.plot(lowedge, NLOLHE_c, color=set2[2],
                       alpha=alpha[5], zorder=99, drawstyle="steps-post",linewidth=lwc)
        p2, = [plt.Rectangle((0, 0), 0, 0, facecolor=set2[2],
                             linewidth=lwc, edgecolor=set2[2], hatch=hatch[7], alpha=alpha[5])]
        if not dist == "ptmiss" :
           lines += ((p2, p1),)

        ax1.fill_between(lowedge, NLOPS_min, NLOPS_max, facecolor=set2[4],
                         linewidth=lwf, edgecolor=set2[0], hatch=hatch[2], alpha=alpha[2], zorder=99, step="post")
        p2, = [plt.Rectangle((0, 0), 0, 0, facecolor=set2[4],
                             linewidth=lwf, edgecolor=set2[0], hatch=hatch[2], alpha=None)]
        p1, = ax1.plot(lowedge, NLOPS_c, color=set2[0],
                       alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
        lines += ((p2, p1),)

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which='both', left=True, right=True,
                        bottom=True, top=True, direction='in', zorder=200)
        ax1.set_xlim(lowedge[0], highedge[-1])

        # Set limits for x and y scale in specific cases:

        if (dist == "eta-lep1rec" or dist == "eta-jet1rec" or dist == "y-LQrec"):
            ax1.set_xlim(-2,2)
            ax1.set_ylim(10**(math.floor(np.log10(LOLHE_min[np.where(lowedge == 2)[0]]))),10**(math.ceil(np.log10(np.max(LOLHE_c))))*1.2)
        if (dist == "eta-lep1" or dist == "eta-jet1" or dist == "y-LQ"):
            ax1.set_xlim(-2,2)
            ax1.set_ylim(10**(math.floor(np.log10(LOLHE_min[np.where(lowedge == 2)[0]]))),10**(math.ceil(np.log10(np.max(LOLHE_c))))*1.2)
        if (dist == "m-LQ" or dist == "m-LQrec" or dist == "mcoll"):
           ax1.set_xlim(900,3000)
        if (dist == "pt-jet1" or dist == "pt-lep1rec" or dist == "pt-lep1"):
           ax1.set_xlim(450,1500)
        if (dist == "pt-LQ"):
           ax1.set_xlim(0,1500)
        if (dist == "ptmiss"):
           ax1.set_xlim(100,800)
           ax1.set_ylim(10e-11,10e-8)
        if (dist == "pt-sys" or dist == "pt-sysrec"):
           ax1.set_xlim(900,2500)
        leg = ax1.legend(lines, labels, loc='best', numpoints=1)

        mylegstyle(leg)

        ax1.set_yscale('log')
        # second plot
        if not  dist in ["ptmiss","dphi","pt-LQ","pt-taujet"]:
           ax2 = fig.add_subplot(gs[1])
           mystyle(ax2)
           mygrid(ax2)

           norm = LOLHE_c

           ax2.fill_between(lowedge, NLOLHE_min/norm, NLOLHE_max/norm, facecolor=set2[2],
                            linewidth=lwf, edgecolor=set2[2], hatch=hatch[7], alpha=alpha[5], zorder=99, step="post")
           p1, = ax2.plot(lowedge, NLOLHE_c/norm, color=set2[2],
                          alpha=alpha[5], zorder=99, drawstyle="steps-post",linewidth=lwc)
           norm = LOLHE_c

           ax2.fill_between(lowedge, LOLHE_min/norm, LOLHE_max/norm, facecolor=facecolor[3],
                            linewidth=lwf, edgecolor=set1[1], hatch=hatch[0], alpha=alpha[1], zorder=99, step="post")
           p1, = ax2.plot(lowedge, LOLHE_c/norm, color=set1[1],
                          alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
           ax2.yaxis.set_minor_locator(AutoMinorLocator())
           ax2.tick_params(which='both', left=True, right=True,
                           bottom=True, top=True, direction='in', zorder=100)

           plt.ylim(0.5, 1.5)
           plt.ylabel(r"$\frac{\mathrm{NLO}}{\mathrm{LO}}$")
           #plt.xlabel(xlabel)
           plt.setp(ax2.get_xticklabels(), visible=False)
           ax2.set_xlim(lowedge[0], highedge[-1])
           # plt.xlabel(r"$p_\perp^{\rm \ell \ell}$ [GeV]", labelpad=+0.5)

           if (dist == "eta-lep1rec" or dist == "eta-jet1rec" or dist == "y-LQrec"):
               ax2.set_xlim(-2,2)
           if (dist == "y-LQ" or dist == "eta-lep1" or dist == "eta-jet1"):
               ax2.set_xlim(-2,2)
           if (dist == "m-LQ" or dist == "m-LQrec" or  dist == "mcoll"):
              ax2.set_xlim(900,3000)
           if (dist == "pt-jet1" or dist == "pt-lep1rec" or dist == "pt-lep1"):
              ax2.set_xlim(450,1500)
           if (dist == "pt-LQ"):
              ax2.set_xlim(0,1500)
           if (dist == "pt-sys" or dist == "pt-sysrec"):
              ax2.set_xlim(900,2500)
        
        # Third plot
        if dist in ["ptmiss","dphi","pt-LQ","pt-taujet"]:
           ax3 =fig.add_subplot(gs[1])
        else:
           ax3 = fig.add_subplot(gs[2])
        mystyle(ax3)
        mygrid(ax3)


        norm = LOPS_c

        p1, = ax3.plot(lowedge, LOPS_c/norm, color=set2[1],
                       alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
        ax3.fill_between(lowedge, LOPS_min/norm, LOPS_max/norm, facecolor=facecolor2[1],
                         linewidth=lwf, edgecolor=set2[1], hatch=hatch[1], alpha=alpha[0], zorder=99, step="post")


        norm = LOPS_c
        
        ax3.fill_between(lowedge, NLOPS_min/norm, NLOPS_max/norm, facecolor=set2[4],
                         linewidth=lwf, edgecolor=set2[0], hatch=hatch[2], alpha=alpha[2], zorder=99, step="post")
        p1, = ax3.plot(lowedge, NLOPS_c/norm, color=set2[0],
                       alpha=alpha[1], zorder=99, drawstyle="steps-post",linewidth=lwc)
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.tick_params(which='both', left=True, right=True,
                        bottom=True, top=True, direction='in', zorder=100)

        plt.ylim(0.5, 1.5)
        if dist == "dphi": plt.ylim(0,2.0)
        if dist == "pt-LQ": plt.ylim(0,2.0)
        plt.ylabel(r"$\frac{\mathrm{NLO+PS}}{\mathrm{LO+PS}}$")
        plt.xlabel(xlabel)
        ax3.set_xlim(lowedge[0], highedge[-1])
        # plt.xlabel(r"$p_\perp^{\rm \ell \ell}$ [GeV]", labelpad=+0.5)

        if (dist == "eta-lep1rec" or dist == "eta-jet1rec" or dist == "y-LQrec"):
            ax3.set_xlim(-2,2)
        if (dist == "y-LQ" or dist == "eta-lep1" or dist == "eta-jet1"):
            ax3.set_xlim(-2,2)
        if (dist == "m-LQ" or dist == "m-LQrec" or dist == "mcoll" ):
           ax3.set_xlim(900,3000)
        if (dist == "pt-jet1" or dist == "pt-lep1rec" or dist == "pt-lep1" or dist == "pt-taujet"):
           ax3.set_xlim(450,1500)
        if (dist == "pt-LQ"):
           ax3.set_xlim(0,1500)
        if (dist == "pt-sys" or dist == "pt-sysrec"):
           ax3.set_xlim(900,2500)
        if (dist == "ptmiss"):
           ax3.set_xlim(100,800)
        # save figure
        plt.subplots_adjust(left=0.13,right=0.95,top=0.88,bottom=0.1)
        pdf_pages.savefig(fig)
        pdf_pages.close()

if __name__ == "__main__":
    main()
