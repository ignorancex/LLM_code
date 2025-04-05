import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import common as cmm
from scipy.interpolate import interp1d
from astropy.io import fits

class Exper(object) :
    def __init__(self,name,ndla,nqso,fname_noise_cmbl,fsky,lmin=8,lmax=3000) :
        self.name=name
        self.lmin=lmin
        self.lmax=lmax
        self.ndla=ndla
        self.nqso=nqso
        data=np.loadtxt(fname_noise_cmbl,unpack=True)
        self.nlcmb=interp1d(data[0],data[1],bounds_error=False,fill_value=data[1][-1])
        self.fsky=fsky

    def cl_noise_dla(self,larr) :
        return 4*np.pi*self.fsky*np.ones_like(larr)/(self.ndla+0.)

    def cl_noise_qso(self,larr) :
        return 4*np.pi*self.fsky*np.ones_like(larr)/(self.nqso+0.)

    def cl_noise_cmb(self,larr) :
        return self.nlcmb(larr)
    
bqso=2.5
bdla=2.
data_dla=(fits.open(cmm.fname_dla_n12))[1].data
def get_nz_oversample(bins,nzar,nbins) :
    zsub=(bins[1:]+bins[:-1])*0.5
    nsub=(nzar+0.0)/((np.sum(nzar)+0.)*(bins[1]-bins[0]))
    nf=interp1d(zsub,nsub,bounds_error=False,fill_value=0)
    zarr=bins[0]+(bins[-1]-bins[0])*np.arange(nbins)/(nbins-1.)
    pzarr=nf(zarr)
    return zarr,pzarr
nz,bn=np.histogram(data_dla['z_abs'],range=[0,7],bins=50); zarr_dlo,nzarr_dlo=get_nz_oversample(bn,nz,256)
bzarr_dlo=bdla*np.ones_like(zarr_dlo)
nz,bins=np.histogram(data_dla['zqso'],range=[0,7],bins=50); zarr_qso,nzarr_qso=get_nz_oversample(bn,nz,256)
bzarr_qso=bqso*np.ones_like(zarr_qso)

cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
clt_dlo=ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo,nzarr_dlo),(zarr_dlo,bzarr_dlo))
clt_qso=ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso,nzarr_qso),(zarr_qso,bzarr_qso))
clt_cmbl=ccl.ClTracerCMBLensing(cosmo,z_source=1100.)

def get_fisher(xp) :
    ll=np.arange(xp.lmin,xp.lmax)
    cl_dd=ccl.angular_cl(cosmo,clt_dlo ,clt_dlo ,ll)#,l_limber=-1)
    cl_dq=ccl.angular_cl(cosmo,clt_dlo ,clt_qso ,ll)#,l_limber=-1)
    cl_dc=ccl.angular_cl(cosmo,clt_dlo ,clt_cmbl,ll)#,l_limber=-1)
    cl_qq=ccl.angular_cl(cosmo,clt_qso ,clt_qso ,ll)#,l_limber=-1)
    cl_qc=ccl.angular_cl(cosmo,clt_qso ,clt_cmbl,ll)#,l_limber=-1)
    cl_cc=ccl.angular_cl(cosmo,clt_cmbl,clt_cmbl,ll)#,l_limber=-1)
    cl_aa=cl_dd+2*cl_dq+cl_qq
    cl_aq=cl_dq+cl_qq
    cl_ac=cl_dc+cl_qc
    nl_dd=cl_dd+xp.cl_noise_dla(ll) 
    nl_dc=cl_dc
    nl_aa=cl_aa+xp.cl_noise_dla(ll)
    nl_aq=cl_aq
    nl_ac=cl_ac
    nl_qq=cl_qq+xp.cl_noise_qso(ll)
    nl_qc=cl_qc
    nl_cc=cl_cc+xp.cl_noise_cmb(ll)

    dvec  =np.transpose(np.array([[cl_dc/bdla,np.zeros_like(cl_dc)],[cl_qc/bqso,cl_qc/bqso]]),axes=[2,0,1])
    covar =np.array([[nl_aa*nl_cc+nl_ac*nl_ac,nl_aq*nl_cc+nl_ac*nl_qc],
                     [nl_aq*nl_cc+nl_ac*nl_qc,nl_qq*nl_cc+nl_qc*nl_qc]])
    covar=np.transpose(covar*(1./(xp.fsky*(2*ll+1.)))[None,None,:],axes=[2,0,1])
    icovar=np.linalg.inv(covar)
    fisher=np.sum(np.sum(dvec[:,:,:,None]*np.sum(icovar[:,:,:,None]*np.transpose(dvec,axes=[0,2,1])[:,None,:,:],axis=2)[:,None,:,:],axis=2),axis=0)
    sigb1=np.sqrt(np.diag(np.linalg.inv(fisher)))[0]
    sigb2=1./np.sqrt(np.sum((cl_dc/bdla)**2*xp.fsky*(2*ll+1.)/(nl_dd*nl_cc+nl_dc**2)))
    sigbdla=np.sqrt(np.diag(np.linalg.inv(fisher)))[0]
    print "   "+xp.name+": sigma(b_DLA)=%.3lf"%(sigbdla)
    #print "   "+xp.name+": sigma(b_DLA)=%.3lE. Best-case: sigma(b_DLA)=%.3lE"%(np.sqrt(np.diag(np.linalg.inv(fisher)))[0],sigb2)
    return sigbdla

fsky_full_boss=0.25
factor_full_boss=1.
fsky_overlap_boss=5000./(4*np.pi*(180/np.pi)**2)
factor_overlap_boss=factor_full_boss*fsky_overlap_boss/fsky_full_boss
xp_pl_boss=Exper("SDSS x Planck",
                 3.4E4*factor_full_boss,1.5E5*factor_full_boss,
                 cmm.fname_kappa_cl,fsky_full_boss,lmin=8)
xp_s3_boss=Exper("SDSS x S3",
                 3.4E4*factor_overlap_boss,1.5E5*factor_overlap_boss,
                 "data/noise_s3.txt",fsky_overlap_boss,lmin=50)
xp_s4_boss=Exper("SDSS x S4",
                 3.4E4*factor_overlap_boss,1.5E5*factor_overlap_boss,
                 "data/noise_s4.txt",fsky_overlap_boss,lmin=50)
fsky_full_desi=14000./(4*np.pi*(180/np.pi)**2)
factor_full_desi=3.*fsky_full_desi/fsky_full_boss
fsky_overlap_desi=5000./(4*np.pi*(180/np.pi)**2)
factor_overlap_desi=factor_full_desi*fsky_overlap_desi/fsky_full_desi
xp_pl_desi=Exper("DESI x Planck",
                 3.4E4*factor_full_desi,1.5E5*factor_full_desi,
                 cmm.fname_kappa_cl,fsky_full_desi,lmin=2)
xp_s3_desi=Exper("DESI x S3",
                 3.4E4*factor_overlap_desi,1.5E5*factor_overlap_desi,
                 "data/noise_s3.txt",fsky_overlap_desi,lmin=50)
xp_s4_desi=Exper("DESI x S4",
                 3.4E4*factor_overlap_desi,1.5E5*factor_overlap_desi,
                 "data/noise_s4.txt",fsky_overlap_desi,lmin=50)
xp_pl_4most=Exper("4MOST x Planck",
                  3.4E4*factor_full_desi,1.5E5*factor_full_desi,
                  cmm.fname_kappa_cl,fsky_full_desi,lmin=2)
xp_s3_4most=Exper("4MOST x S3",
                  3.4E4*factor_full_desi,1.5E5*factor_full_desi,
                 "data/noise_s3.txt",fsky_full_desi,lmin=50)
xp_s4_4most=Exper("4MOST x S4",
                  3.4E4*factor_full_desi,1.5E5*factor_full_desi,
                  "data/noise_s4.txt",fsky_full_desi,lmin=50)

print "\n\n\n"
results={}
results['sdss']={}
results['sdss']['planck']=get_fisher(xp_pl_boss)
results['sdss']['s3']=get_fisher(xp_s3_boss)
results['sdss']['s4']=get_fisher(xp_s4_boss)
results['desi']={}
results['desi']['planck']=get_fisher(xp_pl_desi)
results['desi']['s3']=get_fisher(xp_s3_desi)
results['desi']['s4']=get_fisher(xp_s4_desi)
results['4most']={}
results['4most']['planck']=get_fisher(xp_pl_4most)
results['4most']['s3']=get_fisher(xp_s3_4most)
results['4most']['s4']=get_fisher(xp_s4_4most)

np.save('4casts',results)

print "\n\n\n"
