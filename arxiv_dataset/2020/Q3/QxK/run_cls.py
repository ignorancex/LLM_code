from astropy.io import fits
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import os
import sys
import common as cmm
import pymaster as nmt

if len(sys.argv)!=7 :
    print "Usage : run_correlations.py lmin lmax nlb nside nsims aposcale"
    exit(1)

#ell_min
lmin=int(sys.argv[1])
#ell_max
lmax=int(sys.argv[2])
#nlb
nlb=int(sys.argv[3])
#Resolution of the CMB lensing map
nside=int(sys.argv[4])
#How many simulations do we run for the errors?
nsims=int(sys.argv[5])
#Apodization scale in degrees
aposcale=float(sys.argv[6])
#Use high S/N sample?

#Create output directory
outdir="outputs_ell%d_%d_ns%d_nlb%d_apo%.3lf"%(lmin,lmax,nside,nlb,aposcale)
outdir+="/"
os.system("mkdir -p "+outdir)

#Filenames
fname_cmbl=cmm.fname_kappa_cmbl_prefix+"_%d.fits"%nside
fname_mask_cmbl=cmm.fname_mask_cmbl_prefix+"_%d.fits"%nside
fname_alldata=outdir+"cl_qxk_all"

if os.path.isfile(fname_alldata+".npz") :
    print " File already exists"
    exit(0)

print " Reading QSOs and DLAs"
data_dla_n12=(fits.open(cmm.fname_dla_n12))[1].data
data_dla_n12b=(fits.open(cmm.fname_dla_n12b))[1].data
data_dla_g16=(fits.open(cmm.fname_dla_g16))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data
data_qsu=(fits.open(cmm.fname_qso))[1].data
data_qsu=data_qsu[np.where(data_qso['UNIHI'])[0]];

print " Reading QSO mask"
#msk_qso=hp.ud_grade(hp.read_map(cmm.fname_mask_cmass,verbose=False),nside_out=nside)
msk_qso=hp.ud_grade(hp.read_map(cmm.fname_mask_qso,verbose=False),nside_out=nside)

print " Computing DLA and QSO overdensity maps"
mpn_dla_n12 =np.bincount(hp.ang2pix(nside,np.pi*(90-data_dla_n12['B'])/180,np.pi*data_dla_n12['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_dla_n12['W'])+0.
mpn_qso_n12 =np.bincount(hp.ang2pix(nside,np.pi*(90-data_qso['B'])/180,np.pi*data_qso['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_qso['W_N12'])+0.
mpn_dla_n12b=np.bincount(hp.ang2pix(nside,np.pi*(90-data_dla_n12b['B'])/180,np.pi*data_dla_n12b['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_dla_n12b['W'])+0.
mpn_qso_n12b=np.bincount(hp.ang2pix(nside,np.pi*(90-data_qso['B'])/180,np.pi*data_qso['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_qso['W_N12B'])+0.
mpn_dla_g16 =np.bincount(hp.ang2pix(nside,np.pi*(90-data_dla_g16['B'])/180,np.pi*data_dla_g16['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_dla_g16['W'])+0.
mpn_qso_g16 =np.bincount(hp.ang2pix(nside,np.pi*(90-data_qso['B'])/180,np.pi*data_qso['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_qso['W_G16'])+0.
mpn_qsu     =np.bincount(hp.ang2pix(nside,np.pi*(90-data_qsu['B'])/180,np.pi*data_qsu['L']/180),
                         minlength=hp.nside2npix(nside),weights=data_qsu['W_DUM'])+0.
mpd_dla_n12 =msk_qso*(msk_qso*mpn_dla_n12*np.sum(msk_qso)/np.sum(mpn_dla_n12*msk_qso)-1.)
mpd_qso_n12 =msk_qso*(msk_qso*mpn_qso_n12*np.sum(msk_qso)/np.sum(mpn_qso_n12*msk_qso)-1.)
mpd_dla_n12b=msk_qso*(msk_qso*mpn_dla_n12b*np.sum(msk_qso)/np.sum(mpn_dla_n12b*msk_qso)-1.)
mpd_qso_n12b=msk_qso*(msk_qso*mpn_qso_n12b*np.sum(msk_qso)/np.sum(mpn_qso_n12b*msk_qso)-1.)
mpd_dla_g16 =msk_qso*(msk_qso*mpn_dla_g16*np.sum(msk_qso)/np.sum(mpn_dla_g16*msk_qso)-1.)
mpd_qso_g16 =msk_qso*(msk_qso*mpn_qso_g16*np.sum(msk_qso)/np.sum(mpn_qso_g16*msk_qso)-1.)
mpd_qsu     =msk_qso*(msk_qso*mpn_qsu*np.sum(msk_qso)/np.sum(mpn_qsu*msk_qso)-1.)

print " Reading kappa map and mask"
msk_kappa=hp.read_map(fname_mask_cmbl,verbose=False)
map_kappa=hp.read_map(fname_cmbl,verbose=False)

print " Preparing bandpowers"
bpw=nmt.NmtBin(nside,nlb=nlb)
ell_list_full=bpw.get_effective_ells()
ibin_use=np.where((ell_list_full<=lmax) & (ell_list_full>=lmin))[0]
ell_list=ell_list_full[ibin_use]
nbins=len(ell_list)

print " Apodising masks"
if aposcale>0 :
    msk_kappa_apo=nmt.mask_apodization(msk_kappa,aposcale,apotype='C1')
    msk_qso_apo=nmt.mask_apodization(msk_qso,aposcale,apotype='C1')
else :
    msk_kappa_apo=msk_kappa
    msk_qso_apo=msk_qso

print " Generating initial fields"
fld_kappa   =nmt.NmtField(msk_kappa_apo,[map_kappa])
fld_dla_n12 =nmt.NmtField(msk_qso_apo,[mpd_dla_n12 ])
fld_qso_n12 =nmt.NmtField(msk_qso_apo,[mpd_qso_n12 ])
fld_dla_n12b=nmt.NmtField(msk_qso_apo,[mpd_dla_n12b])
fld_qso_n12b=nmt.NmtField(msk_qso_apo,[mpd_qso_n12b])
fld_dla_g16 =nmt.NmtField(msk_qso_apo,[mpd_dla_g16 ])
fld_qso_g16 =nmt.NmtField(msk_qso_apo,[mpd_qso_g16 ])
fld_qsu     =nmt.NmtField(msk_qso_apo,[mpd_qsu     ])

print " Computing workspace"
wsp=nmt.NmtWorkspace()
if os.path.isfile(outdir+"wsp.dat") :
    wsp.read_from(outdir+"wsp.dat")
else :
    wsp.compute_coupling_matrix(fld_kappa,fld_qsu,bpw)
    wsp.write_to(outdir+"wsp.dat")

def compute_cell(fld1,fld2) :
    cl_coupled=nmt.compute_coupled_cell(fld1,fld2)
    cl_decoupled=wsp.decouple_cell(cl_coupled)
    
    return (cl_decoupled[0])[ibin_use]

print " Computing data power spectra"
cell_dla_n12 =compute_cell(fld_kappa,fld_dla_n12 ); np.savetxt(outdir+"cell_c_dla_n12.txt" ,np.transpose([ell_list,cell_dla_n12 ]))
cell_qso_n12 =compute_cell(fld_kappa,fld_qso_n12 ); np.savetxt(outdir+"cell_c_qso_n12.txt" ,np.transpose([ell_list,cell_qso_n12 ]))
cell_dlo_n12 =cell_dla_n12-cell_qso_n12
cell_dla_n12b=compute_cell(fld_kappa,fld_dla_n12b); np.savetxt(outdir+"cell_c_dla_n12b.txt",np.transpose([ell_list,cell_dla_n12b]))
cell_qso_n12b=compute_cell(fld_kappa,fld_qso_n12b); np.savetxt(outdir+"cell_c_qso_n12b.txt",np.transpose([ell_list,cell_qso_n12b]))
cell_dlo_n12b=cell_dla_n12b-cell_qso_n12b
cell_dla_g16 =compute_cell(fld_kappa,fld_dla_g16 ); np.savetxt(outdir+"cell_c_dla_g16.txt" ,np.transpose([ell_list,cell_dla_g16 ]))
cell_qso_g16 =compute_cell(fld_kappa,fld_qso_g16 ); np.savetxt(outdir+"cell_c_qso_g16.txt" ,np.transpose([ell_list,cell_qso_g16 ]))
cell_dlo_g16 =cell_dla_g16-cell_qso_g16
cell_qsu     =compute_cell(fld_kappa,fld_qsu     ); np.savetxt(outdir+"cell_c_qsu.txt"     ,np.transpose([ell_list,cell_qsu     ]))

def get_random_cell(isim) :
    fname_dla_n12 =outdir+'cell_c_dla_n12_random%d.txt'%isim
    fname_qso_n12 =outdir+'cell_c_qso_n12_random%d.txt'%isim
    fname_dla_n12b=outdir+'cell_c_dla_n12b_random%d.txt'%isim
    fname_qso_n12b=outdir+'cell_c_qso_n12b_random%d.txt'%isim
    fname_dla_g16 =outdir+'cell_c_dla_g16_random%d.txt'%isim
    fname_qso_g16 =outdir+'cell_c_qso_g16_random%d.txt'%isim
    fname_qsu     =outdir+'cell_c_qsu_random%d.txt'%isim
    print "  %d"%isim
    if (not ((os.path.isfile(fname_dla_n12)) and (os.path.isfile(fname_qso_n12)) and
             (os.path.isfile(fname_dla_n12b)) and (os.path.isfile(fname_qso_n12b)) and
             (os.path.isfile(fname_dla_g16)) and (os.path.isfile(fname_qso_g16)) and
             (os.path.isfile(fname_qsu)))) :
        cleanup=True
        mpk=cmm.random_map(1000+isim,msk_kappa,cmm.fname_kappa_cl)
        fldk=nmt.NmtField(msk_kappa_apo,[mpk])
        cl_dla_n12=compute_cell(fldk,fld_dla_n12); np.savetxt(fname_dla_n12,np.transpose([ell_list,cl_dla_n12]))
        cl_qso_n12=compute_cell(fldk,fld_qso_n12); np.savetxt(fname_qso_n12,np.transpose([ell_list,cl_qso_n12]))
        cl_dla_n12b=compute_cell(fldk,fld_dla_n12b); np.savetxt(fname_dla_n12b,np.transpose([ell_list,cl_dla_n12b]))
        cl_qso_n12b=compute_cell(fldk,fld_qso_n12b); np.savetxt(fname_qso_n12b,np.transpose([ell_list,cl_qso_n12b]))
        cl_dla_g16=compute_cell(fldk,fld_dla_g16); np.savetxt(fname_dla_g16,np.transpose([ell_list,cl_dla_g16]))
        cl_qso_g16=compute_cell(fldk,fld_qso_g16); np.savetxt(fname_qso_g16,np.transpose([ell_list,cl_qso_g16]))
        cl_qsu=compute_cell(fldk,fld_qsu); np.savetxt(fname_qsu,np.transpose([ell_list,cl_qsu]))
    else :
        ll,cl_dla_n12 =np.loadtxt(fname_dla_n12 ,unpack=True)
        ll,cl_qso_n12 =np.loadtxt(fname_qso_n12 ,unpack=True)
        ll,cl_dla_n12b=np.loadtxt(fname_dla_n12b,unpack=True)
        ll,cl_qso_n12b=np.loadtxt(fname_qso_n12b,unpack=True)
        ll,cl_dla_g16 =np.loadtxt(fname_dla_g16 ,unpack=True)
        ll,cl_qso_g16 =np.loadtxt(fname_qso_g16 ,unpack=True)
        ll,cl_qsu     =np.loadtxt(fname_qsu     ,unpack=True)

    return ell_list,cl_dla_n12,cl_qso_n12,cl_dla_n12b,cl_qso_n12b,cl_dla_g16,cl_qso_g16,cl_qsu

print " Generating %d random measurements"%nsims
data_randoms=np.zeros([nsims,10,2,nbins])
data_randoms_2=np.zeros([nsims,3,2,2*nbins])
for i in np.arange(nsims) :
    ll,cdn12,cqn12,cdn12b,cqn12b,cdg16,cqg16,cu=get_random_cell(i)
    data_randoms[i,0,:,:]=np.array([ll,cdn12])
    data_randoms[i,1,:,:]=np.array([ll,cdn12b])
    data_randoms[i,2,:,:]=np.array([ll,cdg16])
    data_randoms[i,3,:,:]=np.array([ll,cqn12])
    data_randoms[i,4,:,:]=np.array([ll,cqn12b])
    data_randoms[i,5,:,:]=np.array([ll,cqg16])
    data_randoms[i,6,:,:]=np.array([ll,cu])
    data_randoms[i,7,:,:]=data_randoms[i,0,:,:]-data_randoms[i,3,:,:]
    data_randoms[i,8,:,:]=data_randoms[i,1,:,:]-data_randoms[i,4,:,:]
    data_randoms[i,9,:,:]=data_randoms[i,2,:,:]-data_randoms[i,5,:,:]
    data_randoms_2[i,0,:,:nbins]=np.array([ll,cdn12])
    data_randoms_2[i,0,:,nbins:]=np.array([ll,cqn12])
    data_randoms_2[i,1,:,:nbins]=np.array([ll,cdn12b])
    data_randoms_2[i,1,:,nbins:]=np.array([ll,cqn12b])
    data_randoms_2[i,2,:,:nbins]=np.array([ll,cdg16])
    data_randoms_2[i,2,:,nbins:]=np.array([ll,cqg16])
larr=np.mean(data_randoms[:,0,0,:],axis=0)

np.savez(fname_alldata,ll=larr,
         cell_dla_n12=cell_dla_n12,cell_dla_n12b=cell_dla_n12b,cell_dla_g16=cell_dla_g16,
         cell_qso_n12=cell_qso_n12,cell_qso_n12b=cell_qso_n12b,cell_qso_g16=cell_qso_g16,
         cell_qsu=cell_qsu,randoms=data_randoms,randoms_2=data_randoms_2)
