import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pickle
import pymaster as nmt
import argparse



parser = argparse.ArgumentParser(description='Compute cross power spectrum and store as pickle.')
parser.add_argument('directory', type=str, help='directory which contains data')
parser.add_argument('-b', type=int, help='bandpower width (ells per bandpower); default is 50')
parser.add_argument('--dla', help='compute power spectrum for DLAs (default is QSOs)',action="store_true")
parser.add_argument('--kappa', help='compute auto power spectrum for kappa (default is QSOs)',action="store_true")
args = parser.parse_args()

#dir = '/Users/Joseph/Brookhaven/Research'
dir = args.directory

if args.dla:
    obj = 'dla'
else:
    obj = 'qso'

nside = 2048
if args.b == None:
    n = 50
else:
    n = args.b #bandpower width
b = nmt.NmtBin(nside,nlb=n)
    
mask_kappa = pickle.load(open(dir+'/data/mask_celestial.pkl','rb')) #load unapodized mask
map_kappa = [pickle.load(open(dir+'/data/dat_klm_celestial.pkl','rb'))*mask_kappa] #load map and mask
mask_kappa_apo = pickle.load(open(dir+'/pickled/masks/celestial/mask_sim_celestial_apo.pkl','rb')) #load apodized mask
print('initializing kappa field')
f_kappa=nmt.NmtField(mask_kappa_apo,map_kappa)

# initialize fields
if args.kappa:
    print('computing auto power spectrum')
    cls = nmt.compute_full_master(f_kappa,f_kappa,b)
    print('pickling')
    pickle.dump(cls,open(dir+'/pickled/power_spectra/ps_kk_'+str(n)+'_new.pkl','wb'))
    print('cps stored as:',dir+'/pickled/power_spectra/ps_kk_'+str(n)+'_new.pkl')
else: 
    map_obj = [pickle.load(open(dir+'/pickled/maps/celestial/map_'+obj+'_celestial.pkl','rb'))]
    mask_obj =  pickle.load(open(dir+'/pickled/masks/celestial/mask_'+obj+'_celestial_apo.pkl','rb')) 
    f_obj=nmt.NmtField(mask_obj,map_obj)
    print('computing power spectrum')
    cls = nmt.compute_full_master(f_kappa,f_obj,b)
    print('pickling')
    pickle.dump(cls,open(dir+'/pickled/cps/cps_'+obj+'_'+str(n)+'_new.pkl','wb'))
    print('cps stored as:',dir+'/pickled/cps/cps_'+obj+'_'+str(n)+'_new.pkl')