import pylab as pl
import healpy as hp
from astropy import units as u
import glob
import h5py as h5
import os
import  ps4c
from  ps4c.IO  import read_h5_file, write_h5_file
from ps4c.experiment   import Experiment
from ps4c.catalogforecaster import   CatalogForecaster
import warnings
warnings.filterwarnings("ignore")
import argparse


def main(args):
    fsky=args.fsky
    dir_ps=args.ps4c_dir+'/'
    sigmadet = args.sigma_detection

    debug =args.verbose
    Exp  =Experiment( ID='SO_LAT',sensitivity= args.sensitivities, frequency=(args.frequencies )  ,
                        fwhm=args.fwhms  , fsky=fsky,
                         units_sensitivity=args.unit_sensitivity,
                      units_beam='arcmin')
    fc =CatalogForecaster(Exp ,  ps4c_dir=dir_ps,sigmadetection=5 )
    if args.polarization :
        fc.forecast_pi2scaling(verbose=False    )
    fc(model='lagache',verbose=debug  )
    if debug :
        print(fc.print_info()  )
    data_dic = read_h5_file(args.input_catalog   , ['phi', 'theta','flux'])

    coords= [data_dic['phi'],data_dic['theta']   ]
    lonlat =args.lonlat_coordinates
    ids = pl.argsort(data_dic['flux'])

    fc. forecast_radio_source_fluxes ( coords,
                            verbose = debug  ,
                            rescale_fluxes=True ,
                            nuref= args. reference_freq,
                            simulate_polarization = args.polarization   ,
                            lonlat =lonlat,
                            sort_fluxes=False  ,
                            store_fluxes=False  ,
                            all_sources=True
                              )

    nuref = args.reference_freq *u.GHz
    imin = np.argmin(np.fabs(fc.nu - nuref))
    idx = pl.argsort(fc._fluxes[fc.nu[imin].value]) 
    for freq in args.frequencies   :
        massaged =  pl.zeros_like (   data_dic['flux'] )
        pmassaged =  pl.zeros_like (   data_dic['flux'] )
        massaged[ids] = fc._fluxes[freq][idx]
        pmassaged[ids] = fc._Pfluxes[freq][idx]


        try :
            os.makedirs ( args.output_dir)
        except FileExistsError :
            print(f"Overwriting in {args.output_dir}" )
        write_h5_file(f'{args.output_dir}catalog_{freq}.h5' ,  {'phi':data_dic['phi'],
                                                'theta':data_dic['theta'],
                                                'flux': massaged,
                                                'polarized flux':pmassaged } )
    pass



if __name__=="__main__":
	parser = argparse.ArgumentParser( description="Matching the Websky catalogs with SED and number counts   " )

	parser.add_argument('--frequencies', nargs='+', type=float,help='list of frequencies to simulate catalogs [GHz]')
	parser.add_argument('--fwhms', nargs='+', type=float,
                            help='list of fwhms to simulate catalogs [arcmin ]')
	parser.add_argument('--sensitivities', nargs='+', type=float,
                                  help='list of sensitivities [default = uK arcmin ]')
	parser.add_argument("--input-catalog" , help='path to the  input template catalog'  )

	parser.add_argument("--ps4c-dir", help='path to PS4C (needed to look up the number counts data)', required=True )

	parser.add_argument("--output-dir", help='path to the output matched catalogs', default='./')
	parser.add_argument("--reference-freq", default=100., help='reference frequency to rescale the fluxes with observed SED')

	parser.add_argument("--polarization", action="store_true" , default=False )
	parser.add_argument("--verbose", action="store_true" , default=False )
	parser.add_argument("--lonlat-coordinates", action="store_true" , default=False,
	                        help='assuming coordinates in degrees, (default assumed in radians)  ' )

	parser.add_argument("--unit-sensitivity", help='string chosen between :[uK arcmin, Jy, mJy ]', default='uKarcmin')
	parser.add_argument("--fsky",   default=1 )
	parser.add_argument("--sigma-detection",   default=5 )

	args = parser.parse_args()
	main( args)
