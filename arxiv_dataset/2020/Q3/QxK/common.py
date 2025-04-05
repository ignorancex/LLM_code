import numpy as np
import healpy as hp
import os
import pyfits as pf
import healpy as hp
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

nside_qso=64

fname_mask_cmbl_orig='data/mask.fits'
fname_mask_cmbl_prefix='data/mask'
fname_kappa_cmbl_prefix='data/kappa'
fname_alm_cmbl='data/dat_klm.fits'
fname_dla_n12_orig='data/DLA_DR12_v2.dat'
fname_dla_g16_orig='data/table3.dat'
fname_qso_orig='data/DR12Q.fits'
fname_dla_n12='data/DLA_DR12_n12.fits'
fname_dla_n12b='data/DLA_DR12_n12b.fits'
fname_dla_g16='data/DLA_DR12_g16.fits'
fname_qso='data/QSO_DR12.fits'
fname_mask_qso='data/mask_QSO_ns%d.fits'%nside_qso
fname_mask_cmass='data/msk_cmass.fits'
fname_kappa_cl='data/nlkk.dat'

def compute_xcorr_c(fname_field,fname_mask,fname_catalog,thmax_deg,nbins,logbin=False,
                    thmin_deg=0,fname_out='none',weight_name='W',cut_name='NO_CUT') :
    """Computes 2PCF between a pixelized field and a set of points. Uses fast C implementation.
    fname_field : path to healpix map containing the field
    fname_mask : path to healpix map containing the field's mask
    fname_catalog : path to FITS file containing the points
    thmax_deg : maximum angular separation in degrees
    nbins : number of bins in angular separation
    logbin : should I use logarithmic bins?
    thmin_deg : minimum angular separation in degrees
    fname_out : path to output filename
    cut_name : column name corresponding to a cuts flag. Only objects with that flag!=0 will be used.
    weight_name : column name corresponding to the weights.
    """

    if (fname_out!='none') and (os.path.isfile(fname_out)) :
        th,wth,hf,hm=np.loadtxt(fname_out,unpack=True)
    else :
        do_log=0
        if logbin : do_log=1
        command ="./FieldXCorr "
        command+=fname_field+" "
        command+=fname_mask+" "
        command+=fname_catalog+" "
        command+=fname_out+" "
        command+="%lf %d %d %lf "%(thmax_deg,nbins,do_log,thmin_deg)
        command+=weight_name+" "
        command+=cut_name+" "
        command+=" > "+fname_out+"_log"
        os.system(command)
        th,wth,hf,hm=np.loadtxt(fname_out,unpack=True)

    return th,wth,hf,hm

def compute_xcorr_py(pos1,w1,map2,mask2,thmax_deg,nbins,nside,logbin=False,thmin_deg=0,fname_out='none') :
    """Computes 2PCF between a pixelized field and a set of points. Uses slower python implementation.
    pos1 : 3D positions of the points (unit vectors)
    w1 : weights for each point
    map2 : healpix map of the field
    mask2 : healpix map of the field's mask
    nside : healpix resolution parameter
    thmax_deg : maximum angular separation in degrees
    nbins : number of bins in angular separation
    logbin : should I use logarithmic bins?
    thmin_deg : minimum angular separation in degrees
    fname_out : path to output filename
    """

    if (fname_out!='none') and (os.path.isfile(fname_out)) :
        th,wth,hf_sum,hm_sum=np.loadtxt(fname_out,unpack=True)
    else :
        hf_sum=np.zeros(nbins)
        hn_sum=np.zeros(nbins)
        thmax=thmax_deg*np.pi/180
        thmin=thmin_deg*np.pi/180
        thmax_extend=thmax*1.2
        
        ipix=np.arange(hp.nside2npix(nside))
        pospix=np.array(hp.pix2vec(nside,ipix))

        if logbin :
            logthmax=np.log10(thmax)
            logthmin=np.log10(thmin)
            for i in np.arange(len(pos1)) :
                pos=pos1[i,:]
                w=w1[i]
                disc=hp.query_disc(nside,pos,thmax_extend)
                o_m_cth=1-np.minimum(np.dot(pos,pospix[:,disc]),1)
                lth=0.5*np.log10(2*o_m_cth+o_m_cth*o_m_cth*0.33333+o_m_cth*o_m_cth*o_m_cth*0.0888888)
                
                n=mask2[disc]
                f=map2[disc]
                hn,bins=np.histogram(lth,range=[logthmin,logthmax],bins=nbins,weights=n)
                hf,bins=np.histogram(lth,range=[logthmin,logthmax],bins=nbins,weights=f)
                hn_sum+=hn*w
                hf_sum+=hf*w
                
                th=10.**(0.5*(bins[1:]+bins[:-1]))*180/np.pi
        else :
            for i in np.arange(len(pos1)) :
                pos=pos1[i,:]
                w=w1[i]
                disc=hp.query_disc(nside,pos,thmax_extend)
                o_m_cth=1-np.minimum(np.dot(pos,pospix[:,disc]),1)
                th=np.sqrt(2*o_m_cth+o_m_cth*o_m_cth*0.33333+o_m_cth*o_m_cth*o_m_cth*0.0888888)
                
                n=mask2[disc]
                f=map2[disc]
                hn,bins=np.histogram(th,range=[0,thmax],bins=nbins,weights=n)
                hf,bins=np.histogram(th,range=[0,thmax],bins=nbins,weights=f)
                hn_sum+=hn*w
                hf_sum+=hf*w
        
            th=0.5*(bins[1:]+bins[:-1])*180/np.pi

        igood=np.where(hn_sum>0)
        wth=np.zeros(nbins)
        wth[igood]=hf_sum[igood]/hn_sum[igood]

        if fname_out!='none' :
            np.savetxt(fname_out,np.transpose([th,wth,hf_sum,hn_sum]))

    return th,wth,hf_sum,hn_sum

def random_map(seed,mask,fname_cl,fname_out='none',use_wiener=False) :
    """Generates gaussian random field with correct power spectrum
    mask : healpix map containing a binary mask
    fname_cl : path to ascii file containing the field's power spectrum
    fname_out : path to output FITS file
    """
    np.random.seed(seed)
    nside=hp.npix2nside(len(mask))
    ll,nll,cll=np.loadtxt(fname_cl,unpack=True)
    cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
    nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
    if use_wiener :
        alm=hp.synalm(cl,lmax=2048,new=True,verbose=False) 
        alm=hp.almxfl(alm,(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10))
        mp=hp.alm2map(alm,nside,lmax=2048)
    else :
        mp=hp.synfast(cl,nside,lmax=2048,new=True,verbose=False)
    mp*=mask

    if fname_out!='none' :
        hp.write_map(fname_out,mp)

    return mp

def random_points(seed,mask,npart,fname_out='none',weights=None) :
    """Generates set of points randomly sampled on the sphere within a given mask.
    mask : healpix map containing the mask
    npart : number of points
    fname_out : path to output file
    weights : array to draw random weights from
    """

    data_dla=(fits.open(fname_dla))[1].data
    data_qso=(fits.open(fname_qso))[1].data
    pz_qso,bins=np.histogram(data_qso['Z_PIPE'],range=[-0.1,7.5],bins=50,normed=True)
    pz_par,bins=np.histogram(data_dla['zqso'  ],range=[-0.1,7.5],bins=50,normed=True)
    pz_dla,bins=np.histogram(data_dla['z_abs' ],range=[-0.1,7.5],bins=50,normed=True)
    zarr=0.5*(bins[1:]+bins[:-1])
    w_weight=np.zeros_like(pz_qso);
    igood=np.where(pz_qso>0)[0];
    w_weight[igood]=pz_par[igood]/pz_qso[igood]; w_weight[zarr>5.5]=0
    wfunc=interp1d(zarr,w_weight,bounds_error=False,fill_value=False,kind='nearest')

    fsky=np.mean(mask)
    nside=hp.npix2nside(len(mask))
    nfull=int(npart/fsky)
    np.random.seed(seed)
    rand_nums=np.random.rand(2*nfull)
    th=np.arccos(-1+2*rand_nums[::2])
    phi=2*np.pi*rand_nums[1::2]
    ipx=hp.ang2pix(nside,th,phi)
    isgood=np.where(mask[ipx]>0)[0]
    print len(isgood)
    
    b=90-180*th[isgood]/np.pi
    l=180*phi[isgood]/np.pi
    
    if weights!=None :
        w=np.random.choice(weights,size=len(isgood))
    else :
        w=np.ones_like(b)

    if fname_out!='none' :
        tbhdu=pf.new_table([pf.Column(name='B',format='D',array=b),
                            pf.Column(name='L',format='D',array=l),
                            pf.Column(name='W',format='D',array=w)])
        tbhdu.writeto(fname_out)

def bias_qso(z) :
    """QSO bias according to 1705.04718
    """
    return 1.*np.ones_like(z)

def bias_dla(z) :
    """Default DLA bias
    """
    return 1.*np.ones_like(z)
