from astropy.io import fits
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import os
import sys
import common as cmm

if len(sys.argv)!=6 :
    print "Usage : run_correlations.py thmax nbins nside_cmbl use_wiener[0,1] nsims"
    exit(1)

#Maximum scale (deg) in the computation of the 2PCF
thmax=float(sys.argv[1])
#Number of bins in theta
nth=int(sys.argv[2])
#Resolution of the CMB lensing map
nside_cmbl=int(sys.argv[3])
#Do we implement a wiener filter?
do_wiener=int(sys.argv[4])
#How many simulations do we run for the errors?
nsims=int(sys.argv[5])
use_wiener=False
if do_wiener>0 :
    use_wiener=True

#Create output directory
outdir="outputs_thm%.1lf_ns%d_nb%d"%(thmax,nside_cmbl,nth)
if use_wiener :
    outdir+="_wiener"
outdir+="/"
os.system("mkdir -p "+outdir)

#Filenames
fname_cmbl=cmm.fname_kappa_cmbl_prefix+"_%d"%nside_cmbl
fname_mask_cmbl=cmm.fname_mask_cmbl_prefix+"_%d"%nside_cmbl
if use_wiener :
    fname_cmbl+="_wiener"
    fname_mask_cmbl+="_wiener"
fname_cmbl+=".fits"
fname_mask_cmbl+=".fits"
fname_alldata=outdir+"wth_qxk_all"

if os.path.isfile(fname_alldata+".npz") :
    print " File already exists"
    exit(0)

#print "  Reading kappa map and mask"
mask =hp.read_map(fname_mask_cmbl,verbose=False)
#field=hp.read_map(fname_cmbl,verbose=False)

print " Computing the DLA (N12) 2PCF"
th_dla_n12,wth_dla_n12,hf_dla_n12,hm_dla_n12=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_dla_n12,thmax,nth,
                                                                 fname_out=outdir+"corr_c_dla_n12.txt")
print " Computing the QSO (N12) 2PCF"
th_qso_n12,wth_qso_n12,hf_qso_n12,hm_qso_n12=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,thmax,nth,
                                                                 fname_out=outdir+"corr_c_qso_n12.txt",weight_name='W_N12')
wth_dlo_n12=wth_dla_n12-wth_qso_n12

print " Computing the DLA (N12B) 2PCF"
th_dla_n12b,wth_dla_n12b,hf_dla_n12b,hm_dla_n12b=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_dla_n12b,thmax,nth,
                                                                     fname_out=outdir+"corr_c_dla_n12b.txt")
print " Computing the QSO (N12B) 2PCF"
th_qso_n12b,wth_qso_n12b,hf_qso_n12b,hm_qso_n12b=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,thmax,nth,
                                                                     fname_out=outdir+"corr_c_qso_n12b.txt",weight_name='W_N12B')
wth_dlo_n12b=wth_dla_n12b-wth_qso_n12b

print " Computing the DLA (G16) 2PCF"
th_dla_g16,wth_dla_g16,hf_dla_g16,hm_dla_g16=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_dla_g16,thmax,nth,
                                                                 fname_out=outdir+"corr_c_dla_g16.txt")
print " Computing the QSO (G16) 2PCF"
th_qso_g16,wth_qso_g16,hf_qso_g16,hm_qso_g16=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,thmax,nth,
                                                                 fname_out=outdir+"corr_c_qso_g16.txt",weight_name='W_G16')
wth_dlo_g16=wth_dla_g16-wth_qso_g16

print " Computing the QSO-UNIFORM 2PCF"
th_qsu,wth_qsu,hf_qsu,hm_qsu=cmm.compute_xcorr_c(fname_cmbl,fname_mask_cmbl,cmm.fname_qso,
                                                 thmax,nth,fname_out=outdir+"corr_c_qsu.txt",
                                                 cut_name='UNIHI',weight_name='W_DUM')

def get_random_corr(isim) :
    cleanup=False
    fname_dla_n12 =outdir+'corr_c_dla_n12_random%d.txt'%isim
    fname_qso_n12 =outdir+'corr_c_qso_n12_random%d.txt'%isim
    fname_dla_n12b=outdir+'corr_c_dla_n12b_random%d.txt'%isim
    fname_qso_n12b=outdir+'corr_c_qso_n12b_random%d.txt'%isim
    fname_dla_g16 =outdir+'corr_c_dla_g16_random%d.txt'%isim
    fname_qso_g16 =outdir+'corr_c_qso_g16_random%d.txt'%isim
    fname_qsu=outdir+'corr_c_qsu_random%d.txt'%isim

    if (not ((os.path.isfile(fname_dla_n12)) and (os.path.isfile(fname_qso_n12)) and
             (os.path.isfile(fname_dla_n12b)) and (os.path.isfile(fname_qso_n12b)) and
             (os.path.isfile(fname_dla_g16)) and (os.path.isfile(fname_qso_g16)) and (os.path.isfile(fname_qsu)))) :
        print "  %d"%isim
        cleanup=True
        cmm.random_map(1000+isim,mask,cmm.fname_kappa_cl,fname_out=outdir+'map_random_%d.fits'%isim,
                       use_wiener=use_wiener)

    t_dla_n12,w_dla_n12,f_dla_n12,m_dla_n12=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_dla_n12,thmax,nth,fname_out=fname_dla_n12)
    t_qso_n12,w_qso_n12,f_qso_n12,m_qso_n12=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_qso,thmax,nth,fname_out=fname_qso_n12,weight_name='W_N12')

    t_dla_n12b,w_dla_n12b,f_dla_n12b,m_dla_n12b=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_dla_n12b,thmax,nth,fname_out=fname_dla_n12b)
    t_qso_n12b,w_qso_n12b,f_qso_n12b,m_qso_n12b=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_qso,thmax,nth,fname_out=fname_qso_n12b,weight_name='W_N12B')

    t_dla_g16,w_dla_g16,f_dla_g16,m_dla_g16=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_dla_g16,thmax,nth,fname_out=fname_dla_g16)
    t_qso_g16,w_qso_g16,f_qso_g16,m_qso_g16=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                                cmm.fname_qso,thmax,nth,fname_out=fname_qso_g16,weight_name='W_G16')

    t_qsu,w_qsu,f_qsu,m_qsu=cmm.compute_xcorr_c(outdir+'map_random_%d.fits'%isim,fname_mask_cmbl,
                                                cmm.fname_qso,thmax,nth,cut_name='UNIHI',weight_name='NO_WEIGHT',fname_out=fname_qsu)
        
    if cleanup :
        os.system('rm '+outdir+'map_random_%d.fits '%isim)

    return t_dla_n12,w_dla_n12,t_dla_n12b,w_dla_n12b,t_dla_g16,w_dla_g16,t_qso_n12,w_qso_n12,t_qso_n12b,w_qso_n12b,t_qso_g16,w_qso_g16,t_qsu,w_qsu

print " Generating %d random measurements"%nsims
data_randoms=np.zeros([nsims,10,2,nth])
data_randoms_2=np.zeros([nsims,3,2,2*nth])
for i in np.arange(nsims) :
    tdn12,wdn12,tdn12b,wdn12b,tdg16,wdg16,tqn12,wqn12,tqn12b,wqn12b,tqg16,wqg16,tu,wu=get_random_corr(i)
    data_randoms[i,0,:,:]=np.array([tdn12 ,wdn12 ])
    data_randoms[i,1,:,:]=np.array([tdn12b,wdn12b])
    data_randoms[i,2,:,:]=np.array([tdg16 ,wdg16 ])

    data_randoms[i,3,:,:]=np.array([tqn12 ,wqn12 ])
    data_randoms[i,4,:,:]=np.array([tqn12b,wqn12b])
    data_randoms[i,5,:,:]=np.array([tqg16 ,wqg16 ])
    data_randoms[i,6,:,:]=np.array([tu    ,wu    ])

    data_randoms[i,7,:,:]=data_randoms[i,0,:,:]-data_randoms[i,3,:,:]
    data_randoms[i,8,:,:]=data_randoms[i,1,:,:]-data_randoms[i,4,:,:]
    data_randoms[i,9,:,:]=data_randoms[i,2,:,:]-data_randoms[i,5,:,:]
    data_randoms_2[i,0,:,:nth]=np.array([tdn12 ,wdn12 ])
    data_randoms_2[i,0,:,nth:]=np.array([tqn12 ,wqn12 ])
    data_randoms_2[i,1,:,:nth]=np.array([tdn12b,wdn12b])
    data_randoms_2[i,1,:,nth:]=np.array([tqn12b,wqn12b])
    data_randoms_2[i,2,:,:nth]=np.array([tdg16 ,wdg16 ])
    data_randoms_2[i,2,:,nth:]=np.array([tqg16 ,wqg16 ])
tharr=np.mean(data_randoms[:,0,0,:],axis=0)

np.savez(fname_alldata,th=tharr,
         wth_dla_n12=wth_dla_n12,wth_dla_n12b=wth_dla_n12b,wth_dla_g16=wth_dla_g16,
         wth_qso_n12=wth_qso_n12,wth_qso_n12b=wth_qso_n12b,wth_qso_g16=wth_qso_g16,
         wth_qsu=wth_qsu,randoms=data_randoms,randoms_2=data_randoms_2)
