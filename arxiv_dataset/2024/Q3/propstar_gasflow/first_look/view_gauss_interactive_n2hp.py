import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
sys.path.append('../')
from setup import *
import argparse

def main(fitfile, specfile, ngauss, xstart=232, ystart=247, vminplot=-3, vmaxplot=18):

    # File in K and in km/s
    # x0, y0, x1, y1 = 83, 131, 133, 180
    # where the fit started
    # xmax, ymax = (141, 139)
    # vmin_plot = 9.2
    # vmax_plot = 11.2

    cube = pyspeckit.Cube(specfile+'.fits')
    cube.xarr.convert_to_unit('km/s')
    cube.xarr.velocity_convention = 'radio'
    cube.xarr.xtype = 'velocity'
    cube.Registry.add_fitter('n2hp_vtau', pyspeckit.spectrum.models.n2hp.n2hp_vtau_fitter, 4)
    cube.load_model_fit(fitfile+'.fits', npars=4, npeaks=ngauss, fittype='n2hp_vtau')

    #open interactive panel
    cube.mapplot()
    cube.plot_spectrum(xstart, ystart, plot_fit=True)
    cube.mapplot.plane = fits.getdata(fitfile+'.fits')[2] #cube.parcube[4, :, :] - 10.2
    cube.mapplot(estimator=None, vmin=vminplot, 
                 vmax=vmaxplot, cmap='RdYlBu_r')
    plt.draw()
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive view of the fit results')
    parser.add_argument('fitfile', type=str, help="Filename of the .fits that has the parameters")
    parser.add_argument('specfile', type=str, help="Filename of the .fits that has the spectra")
    parser.add_argument('ngauss', type=int, help="Number of Gaussian components in the fit.")
    parser.add_argument('-x', '--xstart', type=int, default=141, help="X position of pixel initially shown.")
    parser.add_argument('-y', '--ystart', type=int, default=141, help="Y position of pixel initially shown.")
    parser.add_argument('--vmin', type=float, default=-3.0, help="Minimum velocity plotted in colorbar")
    parser.add_argument('--vmax', type=float, default=18.0, help="Minimum velocity plotted in colorbar")
    args = parser.parse_args()

    #
    # dir_list = os.getcwd().split('/')
    # fitfile = dir_list[-3] #'HH211'
    # specfile = dir_list[-2] # 'CD'
    # ngauss = dir_list[-1] 
    #dir_list[-2] 
    main(args.fitfile, args.specfile, args.ngauss, xstart=args.xstart, ystart=args.ystart, vminplot=args.vmin, vmaxplot=args.vmax)
