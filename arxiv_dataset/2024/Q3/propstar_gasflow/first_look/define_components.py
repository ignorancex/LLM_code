import sys
sys.path.append('../')
from setup import *
from AICfunc import *
import argparse
from astropy.wcs import WCS
from astropy.io import fits
import os
import pyspeckit
from spectral_cube import SpectralCube

'''
Note that this code only works for Gaussian models!
'''


def main(ngausslist, overwrite=False, verbose=False):
    
    diagnosticfolder = fitdir + 'diagnosticAIC/'
    
    ncomponentsfile = diagnosticfolder + 'ncomponents_AIC.fits'
    ncomponentsflagfile = diagnosticfolder + 'ncomponents_AIC_flag.fits'
    chisquarefile = diagnosticfolder + 'chisquare_values_{}G.fits'
    aicfile = diagnosticfolder + 'AIC_values_models_{}G.fits'
    deltaaicfile = diagnosticfolder+ 'deltaAIC_values_models.fits'
    probaicfile = diagnosticfolder + 'deltaAICprob_values_models.fits'
    
    paramfiles = []
    for n in ngausslist:
        paramfiles.append(fitfilebase.format(n) + '_2_filtered.fits')
        
    paramsaicnamelist = [fitfilebase.format(n) + '_aic.fits' for n in ngausslist]
    
    if not os.path.exists(ncomponentsfile) or overwrite:
        # Prepare the list of files to compare

        #load the cube
        cubenormal = SpectralCube.read(hc3n_10_9_cube+'.fits')
        mask, header2D = fits.getdata(maskfile+'.fits', header=True)
        rmsmap = fits.getdata(rmsfile)
        nparamsfittedmodels = np.array(ngausslist) * 3 #Models have 3 parameters per number of Gaussians
        print(nparamsfittedmodels)

        # Here we obtained the modeled cubes for each model
        modelcubelist = []
        for paramfile, n in zip(paramfiles, ngausslist):
            print('Loading: ', paramfile)
            spcaux = pyspeckit.Cube(cube=cubenormal, mask=mask)
            spcaux.load_model_fit(paramfile, npars=3, npeaks=n, fittype='gaussian')
            fittedmodel = spcaux.get_modelcube()
            modelcubelist.append(fittedmodel)
        print('Calculating AIC') 
        # (cube, rmsmap, mask, fittedmodels, nfittedmodels=2, nparamsfittedmodels=[3, 6], prob_tolerance=0.05):
        results_aic = get_AIC_for_cube(cubenormal, rmsmap, mask, 
                                       modelcubelist, nfittedmodels=len(ngausslist), 
                                       nparamsfittedmodels=nparamsfittedmodels)
        print('Saving diagnostic plots in ', diagnosticfolder)
        fits.writeto(ncomponentsfile, results_aic[0], header2D, overwrite=True)
        fits.writeto(ncomponentsflagfile, results_aic[1], header2D, overwrite=True)
        fits.writeto(deltaaicfile, results_aic[3], header2D, overwrite=True)
        fits.writeto(probaicfile, results_aic[4], header2D, overwrite=True)
        for pos, n in enumerate(ngausslist):
            fits.writeto(chisquarefile.format(n), results_aic[2][pos], header2D, overwrite=True)
            fits.writeto(aicfile.format(n), results_aic[5][pos], header2D, overwrite=True)
        
    else:
        results_aic = (fits.getdata(ncomponentsfile), fits.getdata(ncomponentsflagfile))
        print(ncomponentsfile + ' and ' + ncomponentsflagfile + ' already exists. Loaded.')
        
    filesexist = [os.path.exists(paramsaicname) for paramsaicname in paramsaicnamelist]
    
    if not np.all(filesexist) or overwrite:
        parameters = [fits.getdata(paramfile) for paramfile in paramfiles] # this is an [nmodel, 6, y, x] dimension array 
        for i, n in enumerate(ngausslist):
            headeraic = fits.getheader(paramfiles[i])
            paramsaic = get_aic_filtered_params(parameters[i], n, results_aic[0])
            fits.writeto(paramsaicnamelist[i], paramsaic, headeraic, overwrite=True)
    else:
        print('Parameter files already exist.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Determines the number of Gaussian components in each spectrum (either 1 or 2) according to the Akaike Information Criterion')
    parser.add_argument('g', type=int, nargs='+', help="Gaussian models to compare, e.g. '1 2' compares models of 1 and 2 Gaussian fits.")
    parser.add_argument('-o', '--overwrite', action="store_true", help="Overwrite existing files.")
    parser.add_argument('-v', '--verbose', action="store_true", help="Activate verbose output.")

    args = parser.parse_args()
    ngausslist = args.g
    print(ngausslist)
    
    main(ngausslist, overwrite=args.overwrite, verbose=args.verbose)


