#!/bin/python
################################################################################
# HEAT MAP VISUALIZER
################################################################################
# author: Andreas Hessenthaler, University of Stuttgart
################################################################################
import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import code

def main():
    interactiveMode = False
    exportFigure    = True
    silentMode      = False
    exportFormat    = "png" # png, svg
    exportDPI       = 300
    # get Matplotlib version
    mplVersion          = mpl.__version__
    mplVersionMajor, mplVersionMinor, mplVersionPatch = [int(x, 10) for x in mplVersion.split(".")]
    # check SciPy version for compatibility
    scipyVersion        = sp.version.version
    scipyVersionMajor, scipyVersionMinor, scipyVersionPatch = [int(x, 10) for x in scipyVersion.split(".")]
    if ((scipyVersionMajor != 0) or (scipyVersionMinor <= 13)):
        print(">>>ERROR: Requires SciPy version >= 0.14. You have "+scipyVersion+"!")
        sys.exit()
    # print help
    if (len(sys.argv) < 9):
        print("missing or invalid commandline arguments:")
        print("    python heatmap.py arg1 arg2 arg3")
        print("         arg1 : name of file that contains 2D axis data")
        print("         arg2 : name of file that contains data")
        print("         arg3 : minimum of x-range (NAN will auto-scale x-axis)")
        print("         arg4 : maximum of x-range (NAN will auto-scale x-axis)")
        print("         arg5 : minimum of y-range (NAN will auto-scale y-axis)")
        print("         arg6 : maximum of y-range (NAN will auto-scale y-axis)")
        print("         arg7 : minimum of data range (NAN will auto-scale data range)")
        print("         arg8 : maximum of data range (NAN will auto-scale data range)")
        print("         arg9 : allow user interaction (optional; default: False)")
        return
    # get command line arguments
    axisfile        = str(sys.argv[1])
    datafile        = str(sys.argv[2])
    xmin            = float(sys.argv[3])
    xmax            = float(sys.argv[4])
    ymin            = float(sys.argv[5])
    ymax            = float(sys.argv[6])
    dmin            = float(sys.argv[7])
    dmax            = float(sys.argv[8])
    if(len(sys.argv) > 9):
        interactiveMode = bool(sys.argv[9])
    # load data
    xy      = np.loadtxt(axisfile)
    d       = np.loadtxt(datafile)
    x, y    = xy.T
    nanMask = np.isnan(d)

    tri     = Delaunay(xy, incremental=False)
    # get data range
    if (np.isnan(xmin) or np.isnan(xmax)):
        xmin    = np.min(x)
        xmax    = np.max(x)
    if (np.isnan(ymin) or np.isnan(ymax)):
        ymin    = np.min(y)
        ymax    = np.max(y)
    if (np.isnan(dmin) or np.isnan(dmax)):
        dmin    = np.nanmin(d)
        dmax    = np.nanmax(d)
    # plot data
    fig = plt.figure()
    plt.rc("text", usetex=True)
    plt.rc("font", size=24)
    # plt.rc("ps", usedistiller=xpdf)
    mng = plt.get_current_fig_manager()
    # colormap    = plt.get_cmap("coolwarm")
    colormap    = plt.get_cmap("viridis")
    heatmap     = mpl.pyplot.tripcolor(mpl.tri.Triangulation(x, y), d, vmin=dmin, vmax=dmax, shading="gouraud", cmap=colormap)
    heatmap.cmap.set_under(color="white")
    plt.gca().set_aspect("equal")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel(r"$\delta_t \Re (\xi)$")
    plt.ylabel(r"$\delta_t \Im (\xi)$")
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    fig.canvas.set_window_title(os.path.abspath(datafile))
    try:
        # tkAgg
        mng.frame.Maximize(True)
    except AttributeError:
        # Qt
        try:
            mng.window.showMaximized()
        except AttributeError:
            try:
                mng.resize(*mng.window.maxsize())
            except AttributeError:
                print("Encountered unknown backend "+mpl.get_backend()+". Please check with developer.")
    if exportFigure:
        if ((mplVersionMajor < 2) and (mplVersionMinor < 6) and (mplVersionPatch < 2) and (exportFormat == "svg")):
            print(">>>ERROR: EPS/SVG export requires Matplotlib version >= 1.5.2. You have "+mplVersion+"!")
            sys.exit()
        plt.savefig("heatmap."+exportFormat, format=exportFormat, dpi=exportDPI)
    if not(silentMode):
        plt.show()
    # allow user to interact with data in workspace
    if interactiveMode:
        code.interact(local=locals())

if __name__ == "__main__":
    main()
