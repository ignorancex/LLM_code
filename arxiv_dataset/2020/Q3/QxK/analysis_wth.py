from astropy.io import fits
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pylab
from scipy.interpolate import interp1d
import os
import sys
import common as cmm
import pyccl as ccl
import cl2wth as c2w
from scipy.integrate import quad
import scipy.stats as st

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


if len(sys.argv)!=6 :
    print "Usage : analysis_wth.py thmax nbins nside use_wiener[0,1] plot_stuff[0,1]"
    exit(1)

verbose=False
#Maximum scale (deg) in the computation of the 2PCF
thmax=float(sys.argv[1])
#Number of bins in theta
nth=int(sys.argv[2])
#Resolution of the CMB lensing map
nside=int(sys.argv[3])
#Used Wiener-filtered map?
do_wiener=int(sys.argv[4])
#Plot stuff
do_plot_stuff=int(sys.argv[5])

use_wiener=False
if do_wiener>0 :
    use_wiener=True
plot_stuff=False
if do_plot_stuff>0 :
    plot_stuff=True

#Create output directory
outdir="outputs_thm%.1lf_ns%d_nb%d"%(thmax,nside,nth)
if use_wiener :
    outdir+="_wiener"
outdir+="/"

line_out="wth_Ns%d_thm%.1lf_nb%d "%(nside,thmax,nth)

fname_alldata=outdir+"wth_qxk_all"

data_dla_n12=(fits.open(cmm.fname_dla_n12))[1].data
data_dla_n12b=(fits.open(cmm.fname_dla_n12b))[1].data
data_dla_g16=(fits.open(cmm.fname_dla_g16))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data

d=np.load(fname_alldata+'.npz')

tharr=d['th']
wth_dla_n12=d['wth_dla_n12']; wth_qso_n12=d['wth_qso_n12']; wth_dlo_n12=wth_dla_n12-wth_qso_n12
wth_dla_n12b=d['wth_dla_n12b']; wth_qso_n12b=d['wth_qso_n12b']; wth_dlo_n12b=wth_dla_n12b-wth_qso_n12b
wth_dla_g16=d['wth_dla_g16']; wth_qso_g16=d['wth_qso_g16']; wth_dlo_g16=wth_dla_g16-wth_qso_g16
wth_qsu=d['wth_qsu']


if verbose :
    print " Computing covariance matrices"
mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])
mean_all_n12b=np.mean(d['randoms_2'][:,0,1,:],axis=0)
covar_all_n12b=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12b[:,None]*mean_all_n12b[None,:]
corr_all_n12b=covar_all_n12b/np.sqrt(np.diag(covar_all_n12b)[None,:]*np.diag(covar_all_n12b)[:,None])
mean_all_g16=np.mean(d['randoms_2'][:,0,1,:],axis=0)
covar_all_g16=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_g16[:,None]*mean_all_g16[None,:]
corr_all_g16=covar_all_g16/np.sqrt(np.diag(covar_all_g16)[None,:]*np.diag(covar_all_g16)[:,None])

mean_dla_n12=np.mean(d['randoms'][:,0,1,:],axis=0)
covar_dla_n12=np.mean(d['randoms'][:,0,1,:,None]*d['randoms'][:,0,1,None,:],axis=0)-mean_dla_n12[:,None]*mean_dla_n12[None,:]
corr_dla_n12=covar_dla_n12/np.sqrt(np.diag(covar_dla_n12)[None,:]*np.diag(covar_dla_n12)[:,None])
mean_dla_n12b=np.mean(d['randoms'][:,1,1,:],axis=0)
covar_dla_n12b=np.mean(d['randoms'][:,1,1,:,None]*d['randoms'][:,1,1,None,:],axis=0)-mean_dla_n12b[:,None]*mean_dla_n12b[None,:]
corr_dla_n12b=covar_dla_n12b/np.sqrt(np.diag(covar_dla_n12b)[None,:]*np.diag(covar_dla_n12b)[:,None])
mean_dla_g16=np.mean(d['randoms'][:,2,1,:],axis=0)
covar_dla_g16=np.mean(d['randoms'][:,2,1,:,None]*d['randoms'][:,2,1,None,:],axis=0)-mean_dla_g16[:,None]*mean_dla_g16[None,:]
corr_dla_g16=covar_dla_g16/np.sqrt(np.diag(covar_dla_g16)[None,:]*np.diag(covar_dla_g16)[:,None])

mean_qso_n12=np.mean(d['randoms'][:,3,1,:],axis=0)
covar_qso_n12=np.mean(d['randoms'][:,3,1,:,None]*d['randoms'][:,3,1,None,:],axis=0)-mean_qso_n12[:,None]*mean_qso_n12[None,:]
corr_qso_n12=covar_qso_n12/np.sqrt(np.diag(covar_qso_n12)[None,:]*np.diag(covar_qso_n12)[:,None])
mean_qso_n12b=np.mean(d['randoms'][:,4,1,:],axis=0)
covar_qso_n12b=np.mean(d['randoms'][:,4,1,:,None]*d['randoms'][:,4,1,None,:],axis=0)-mean_qso_n12b[:,None]*mean_qso_n12b[None,:]
corr_qso_n12b=covar_qso_n12b/np.sqrt(np.diag(covar_qso_n12b)[None,:]*np.diag(covar_qso_n12b)[:,None])
mean_qso_g16=np.mean(d['randoms'][:,5,1,:],axis=0)
covar_qso_g16=np.mean(d['randoms'][:,5,1,:,None]*d['randoms'][:,5,1,None,:],axis=0)-mean_qso_g16[:,None]*mean_qso_g16[None,:]
corr_qso_g16=covar_qso_g16/np.sqrt(np.diag(covar_qso_g16)[None,:]*np.diag(covar_qso_g16)[:,None])

mean_qsu=np.mean(d['randoms'][:,6,1,:],axis=0)
covar_qsu=np.mean(d['randoms'][:,6,1,:,None]*d['randoms'][:,6,1,None,:],axis=0)-mean_qsu[:,None]*mean_qsu[None,:]
corr_qsu=covar_qsu/np.sqrt(np.diag(covar_qsu)[None,:]*np.diag(covar_qsu)[:,None])

mean_dlo_n12=np.mean(d['randoms'][:,7,1,:],axis=0)
covar_dlo_n12=np.mean(d['randoms'][:,7,1,:,None]*d['randoms'][:,7,1,None,:],axis=0)-mean_dlo_n12[:,None]*mean_dlo_n12[None,:]
corr_dlo_n12=covar_dlo_n12/np.sqrt(np.diag(covar_dlo_n12)[None,:]*np.diag(covar_dlo_n12)[:,None])
mean_dlo_n12b=np.mean(d['randoms'][:,8,1,:],axis=0)
covar_dlo_n12b=np.mean(d['randoms'][:,8,1,:,None]*d['randoms'][:,8,1,None,:],axis=0)-mean_dlo_n12b[:,None]*mean_dlo_n12b[None,:]
corr_dlo_n12b=covar_dlo_n12b/np.sqrt(np.diag(covar_dlo_n12b)[None,:]*np.diag(covar_dlo_n12b)[:,None])
mean_dlo_g16=np.mean(d['randoms'][:,9,1,:],axis=0)
covar_dlo_g16=np.mean(d['randoms'][:,9,1,:,None]*d['randoms'][:,9,1,None,:],axis=0)-mean_dlo_g16[:,None]*mean_dlo_g16[None,:]
corr_dlo_g16=covar_dlo_g16/np.sqrt(np.diag(covar_dlo_g16)[None,:]*np.diag(covar_dlo_g16)[:,None])

if plot_stuff :
    def plot_corr(mat,name,fname='none') :
        plt.figure()
        ax=plt.gca()
        im=ax.imshow(corr_all,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
        cb=plt.colorbar(im,ax=ax)
        ax.text(0.27,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes)
        ax.text(0.27,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes)
        ax.text(0.77,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes)
        ax.text(0.77,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes)
        ax.set_xlabel('${\\rm bin}\\,i$',fontsize=14)
        ax.set_ylabel('${\\rm bin}\\,j$',fontsize=14)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        if fname!='none' :
            plt.savefig(fname,bbox_inches='tight')
        plot_corr(corr_all_n12,"N12")
        plot_corr(corr_all_n12b,"N12B")
        plot_corr(corr_all_g16,"G16")

if verbose :
    print " Computing theory prediction"
def get_nz_oversample(bins,nzar,nbins) :
    zsub=(bins[1:]+bins[:-1])*0.5
    nsub=(nzar+0.0)/((np.sum(nzar)+0.)*(bins[1]-bins[0]))
    nf=interp1d(zsub,nsub,bounds_error=False,fill_value=0)
    zarr=bins[0]+(bins[-1]-bins[0])*np.arange(nbins)/(nbins-1.)
    pzarr=nf(zarr)
    return zarr,pzarr
nz,bn=np.histogram(data_dla_n12['z_abs'],range=[0,7],bins=50); zarr_dlo_n12,nzarr_dlo_n12=get_nz_oversample(bn,nz,256)
bzarr_dlo_n12=cmm.bias_dla(zarr_dlo_n12)
nz,bins=np.histogram(data_dla_n12['zqso'],range=[0,7],bins=50); zarr_qso_n12,nzarr_qso_n12=get_nz_oversample(bn,nz,256)
bzarr_qso_n12=cmm.bias_qso(zarr_qso_n12)
nz,bn=np.histogram(data_dla_n12b['z_abs'],range=[0,7],bins=50); zarr_dlo_n12b,nzarr_dlo_n12b=get_nz_oversample(bn,nz,256)
bzarr_dlo_n12b=cmm.bias_dla(zarr_dlo_n12b)
nz,bins=np.histogram(data_dla_n12b['zqso'],range=[0,7],bins=50); zarr_qso_n12b,nzarr_qso_n12b=get_nz_oversample(bn,nz,256)
bzarr_qso_n12b=cmm.bias_qso(zarr_qso_n12b)
nz,bn=np.histogram(data_dla_g16['z_abs'],range=[0,7],bins=50); zarr_dlo_g16,nzarr_dlo_g16=get_nz_oversample(bn,nz,256)
bzarr_dlo_g16=cmm.bias_dla(zarr_dlo_g16)
nz,bins=np.histogram(data_dla_g16['zqso'],range=[0,7],bins=50); zarr_qso_g16,nzarr_qso_g16=get_nz_oversample(bn,nz,256)
bzarr_qso_g16=cmm.bias_qso(zarr_qso_g16)
nz,bins=np.histogram(data_qso['Z_PIPE'][np.where(data_qso['UNIHI'])[0]],range=[0,7],bins=50)
zarr_qsu,nzarr_qsu=get_nz_oversample(bn,nz,256)
bzarr_qsu=cmm.bias_qso(zarr_qsu)

if plot_stuff :
    plt.figure()
    plt.plot(zarr_dlo_n12 ,nzarr_dlo_n12 ,'r-')
    plt.plot(zarr_dlo_n12b,nzarr_dlo_n12b,'r--')
    plt.plot(zarr_dlo_g16 ,nzarr_dlo_g16 ,'r-.')
    plt.plot(zarr_qso_n12 ,nzarr_qso_n12 ,'b-')
    plt.plot(zarr_qso_n12b,nzarr_qso_n12b,'b--')
    plt.plot(zarr_qso_g16 ,nzarr_qso_g16 ,'b-.')
    plt.plot(zarr_qsu,nzarr_qsu,'g-')

if verbose :
    print "   Cls"
ll,nll,cll=np.loadtxt(cmm.fname_kappa_cl,unpack=True)
cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
wl=(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10)
lmx_wl=ll[-1]
def get_wiener(ell) :
    ret=np.zeros(len(ell))
    ids_good=np.where(ell<=lmx_wl)[0]
    ret[ids_good]=wl[(ell.astype(int))[ids_good]]
    return ret

if not os.path.isfile(outdir+"cls_th.txt") :
    cosmo=ccl.Cosmology(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
    clt_dlo_n12 =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo_n12,nzarr_dlo_n12),(zarr_dlo_n12,bzarr_dlo_n12))
    clt_qso_n12 =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso_n12,nzarr_qso_n12),(zarr_qso_n12,bzarr_qso_n12))
    clt_dlo_n12b =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo_n12b,nzarr_dlo_n12b),(zarr_dlo_n12b,bzarr_dlo_n12b))
    clt_qso_n12b =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso_n12b,nzarr_qso_n12b),(zarr_qso_n12b,bzarr_qso_n12b))
    clt_dlo_g16 =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_dlo_g16,nzarr_dlo_g16),(zarr_dlo_g16,bzarr_dlo_g16))
    clt_qso_g16 =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qso_g16,nzarr_qso_g16),(zarr_qso_g16,bzarr_qso_g16))
    clt_qsu =ccl.ClTracerNumberCounts(cosmo,False,False,(zarr_qsu,nzarr_qsu),(zarr_qsu,bzarr_qsu))
    clt_cmbl=ccl.ClTracerCMBLensing(cosmo)
    larr_b     =np.concatenate((1.*np.arange(500),500+10*np.arange(950)))
    if use_wiener :
        wil=get_wiener(larr_b)
    else :
        wil=np.ones(len(larr_b))
    cl_dc_n12=ccl.angular_cl(cosmo,clt_dlo_n12,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_qc_n12=ccl.angular_cl(cosmo,clt_qso_n12,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_dc_n12b=ccl.angular_cl(cosmo,clt_dlo_n12b,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_qc_n12b=ccl.angular_cl(cosmo,clt_qso_n12b,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_dc_g16=ccl.angular_cl(cosmo,clt_dlo_g16,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_qc_g16=ccl.angular_cl(cosmo,clt_qso_g16,clt_cmbl,larr_b,l_limber=-1)*wil
    cl_uc=ccl.angular_cl(cosmo,clt_qsu,clt_cmbl,larr_b,l_limber=-1)*wil
    np.savetxt(outdir+"cls_th.txt",np.transpose([larr_b,
                                                 cl_dc_n12,cl_qc_n12,
                                                 cl_dc_n12b,cl_qc_n12b,
                                                 cl_dc_g16,cl_qc_g16,
                                                 cl_uc]))
larr,cl_dc_n12,cl_qc_n12,cl_dc_n12b,cl_qc_n12b,cl_dc_g16,cl_qc_g16,cl_uc=np.loadtxt(outdir+"cls_th.txt",unpack=True)

if verbose :
    print "   w(theta)"
if not os.path.isfile(outdir+"wth_th.txt") :
    tharr_th=thmax*np.arange(256)/255.
    wth_th_dlo_n12=c2w.compute_wth(larr,cl_dc_n12,tharr_th)
    wth_th_qso_n12=c2w.compute_wth(larr,cl_qc_n12,tharr_th)
    wth_th_dlo_n12b=c2w.compute_wth(larr,cl_dc_n12b,tharr_th)
    wth_th_qso_n12b=c2w.compute_wth(larr,cl_qc_n12b,tharr_th)
    wth_th_dlo_g16=c2w.compute_wth(larr,cl_dc_g16,tharr_th)
    wth_th_qso_g16=c2w.compute_wth(larr,cl_qc_g16,tharr_th)
    wth_th_qsu=c2w.compute_wth(larr,cl_uc,tharr_th)
    np.savetxt(outdir+"wth_th.txt",np.transpose([tharr_th,
                                                 wth_th_dlo_n12,wth_th_qso_n12,
                                                 wth_th_dlo_n12b,wth_th_qso_n12b,
                                                 wth_th_dlo_g16,wth_th_qso_g16,
                                                 wth_th_qsu]))
tharr_th,wth_th_dlo_n12,wth_th_qso_n12,wth_th_dlo_n12b,wth_th_qso_n12b,wth_th_dlo_g16,wth_th_qso_g16,wth_th_qsu=np.loadtxt(outdir+"wth_th.txt",unpack=True)
    
#Binning theory
dth=tharr[1]-tharr[0]
wth_f_dlo_n12=interp1d(tharr_th,tharr_th*wth_th_dlo_n12,bounds_error=False,fill_value=0)
wth_pr_dlo_n12=np.array([quad(wth_f_dlo_n12,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_qso_n12=interp1d(tharr_th,tharr_th*wth_th_qso_n12,bounds_error=False,fill_value=0)
wth_pr_qso_n12=np.array([quad(wth_f_qso_n12,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_dlo_n12b=interp1d(tharr_th,tharr_th*wth_th_dlo_n12b,bounds_error=False,fill_value=0)
wth_pr_dlo_n12b=np.array([quad(wth_f_dlo_n12b,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_qso_n12b=interp1d(tharr_th,tharr_th*wth_th_qso_n12b,bounds_error=False,fill_value=0)
wth_pr_qso_n12b=np.array([quad(wth_f_qso_n12b,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_dlo_g16=interp1d(tharr_th,tharr_th*wth_th_dlo_g16,bounds_error=False,fill_value=0)
wth_pr_dlo_g16=np.array([quad(wth_f_dlo_g16,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_qso_g16=interp1d(tharr_th,tharr_th*wth_th_qso_g16,bounds_error=False,fill_value=0)
wth_pr_qso_g16=np.array([quad(wth_f_qso_g16,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])
wth_f_qsu=interp1d(tharr_th,tharr_th*wth_th_qsu,bounds_error=False,fill_value=0)
wth_pr_qsu=np.array([quad(wth_f_qsu,th-dth/2,th+dth/2)[0]/(th*dth) for th in tharr])

if plot_stuff :
    plt.figure()
    plt.plot(tharr_th,wth_th_dlo_n12,'r-')
    plt.plot(tharr_th,wth_th_dlo_n12b,'r--')
    plt.plot(tharr_th,wth_th_dlo_g16,'r-.')
    plt.plot(tharr_th,wth_th_qso_n12,'b-')
    plt.plot(tharr_th,wth_th_qso_n12b,'b--')
    plt.plot(tharr_th,wth_th_qso_g16,'b-.')
    plt.plot(tharr_th,wth_th_qsu,'k-')

i_good=np.where(tharr<thmax)[0]; ndof=len(i_good);

#Fitting the 2PCF difference
def fit_bias_single(wth_dlo,wth_pr_dlo,covar_dlo) :
    #Data vectors and covariances
    dv=wth_dlo[i_good]; tv=wth_pr_dlo[i_good]; cv=(covar_dlo[i_good,:])[:,i_good]; icv=np.linalg.inv(cv)
    #chi^2
    #Analytic Best-fit and errors
    sigma_b=1./np.sqrt(np.dot(tv,np.dot(icv,tv)))
    b_bf=np.dot(tv,np.dot(icv,dv))/np.dot(tv,np.dot(icv,tv))
    pv=b_bf*tv
    chi2=np.dot(dv-pv,np.dot(icv,dv-pv))
    pte=1-st.chi2.cdf(chi2,ndof-1)
    return b_bf,sigma_b,chi2,pte

b_dlo_n12,sb_dlo_n12,chi2_dlo_n12,pte_dlo_n12=fit_bias_single(wth_dlo_n12,wth_pr_dlo_n12,covar_dlo_n12)
b_dlo_n12b,sb_dlo_n12b,chi2_dlo_n12b,pte_dlo_n12b=fit_bias_single(wth_dlo_n12b,wth_pr_dlo_n12b,covar_dlo_n12b)
b_dlo_g16,sb_dlo_g16,chi2_dlo_g16,pte_dlo_g16=fit_bias_single(wth_dlo_g16,wth_pr_dlo_g16,covar_dlo_g16)
b_qsu,sb_qsu,chi2_qsu,pte_qsu=fit_bias_single(wth_qsu,wth_pr_qsu,covar_qsu)
line_out+="%.3lE %.3lE %.3lE "%(b_qsu,sb_qsu,chi2_qsu)
line_out+="%.3lE %.3lE %.3lE "%(b_dlo_n12,sb_dlo_n12,chi2_dlo_n12)
line_out+="%.3lE %.3lE %.3lE "%(b_dlo_n12b,sb_dlo_n12b,chi2_dlo_n12b)
line_out+="%.3lE %.3lE %.3lE "%(b_dlo_g16,sb_dlo_g16,chi2_dlo_g16)
print " QSO bias"
print "   b_QSO = %.3lf +- %.3lf"%(b_qsu,sb_qsu)
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_qsu,ndof-1,pte_qsu)
print " Bias from QSO-DLA difference"
print "  N12"
print "   b_DLA = %.3lf +- %.3lf"%(b_dlo_n12,sb_dlo_n12)
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dlo_n12,ndof-1,pte_dlo_n12)
print "  N12B"
print "   b_DLA = %.3lf +- %.3lf"%(b_dlo_n12b,sb_dlo_n12b)
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dlo_n12b,ndof-1,pte_dlo_n12b)
print "  G16"
print "   b_DLA = %.3lf +- %.3lf"%(b_dlo_g16,sb_dlo_g16)
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dlo_g16,ndof-1,pte_dlo_g16)
print " ------"
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    ax.errorbar(tharr,1E3*wth_dlo_n12,yerr=1E3*np.sqrt(np.diag(covar_dlo_n12)),fmt='ro',
                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
    ax.plot(tharr_th,b_dlo_n12*1E3*wth_th_dlo_n12,'r-',lw=2,label='${\\rm best\\,\\,fit}$')
#    ax.errorbar(tharr,1E3*wth_dlo_n12b,yerr=1E3*np.sqrt(np.diag(covar_dlo_n12b)),fmt='go',
#                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
#    ax.plot(tharr_th,b_dlo_n12b*1E3*wth_th_dlo_n12b,'g-',lw=2,label='${\\rm best\\,\\,fit}$')
#    ax.errorbar(tharr,1E3*wth_dlo_g16,yerr=1E3*np.sqrt(np.diag(covar_dlo_g16)),fmt='bo',
#                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
#    ax.plot(tharr_th,b_dlo_g16*1E3*wth_th_dlo_g16,'b-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.errorbar(tharr,1E3*wth_qsu,yerr=1E3*np.sqrt(np.diag(covar_qsu)),fmt='ko',
                label='$\\kappa\\times{\\rm QSU}$')
    ax.plot(tharr_th,b_dlo_n12*1E3*wth_th_qsu,'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,thmax])
    ax.set_ylim([-0.1,0.35])
    ax.set_xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=14)
    ax.set_ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle\\times10^3$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
#    plt.legend(loc='upper right',frameon=False,fontsize=14)
#    plt.savefig("doc/wth_x1.pdf",bbox_inches='tight')


#Fitting both 2PCFs with b_DLA and b_QSO
def fit_bias_both(wth_dla,wth_qso,wth_pr_dlo,wth_pr_qso,covar_all) :
    dv=np.concatenate((wth_dla[i_good],wth_qso[i_good]));
    tv1=np.concatenate((wth_pr_dlo[i_good],np.zeros(ndof)));
    tv2=np.concatenate((wth_pr_qso[i_good],wth_pr_qso[i_good]))
    tv=np.array([tv1,tv2])
    cv=np.zeros([2*ndof,2*ndof])
    cv[:ndof,:ndof]=covar_all[:ndof,:ndof]
    cv[:ndof,ndof:]=covar_all[:ndof,nth:nth+ndof]
    cv[ndof:,:ndof]=covar_all[nth:nth+ndof,:ndof]
    cv[ndof:,ndof:]=covar_all[nth:nth+ndof,nth:nth+ndof]
    icv=np.linalg.inv(cv)

    cov_b=np.linalg.inv(np.dot(tv,np.dot(icv,np.transpose(tv))))
    b_bf=np.dot(cov_b,np.dot(tv,np.dot(icv,np.transpose(dv))))
    sigma_b=np.sqrt(np.diag(cov_b))
    pv=np.dot(b_bf,tv);
    chi2=np.dot((dv-pv),np.dot(icv,(dv-pv)));
    pte=1-st.chi2.cdf(chi2,2*ndof-2)
    return b_bf,cov_b,chi2,pte

b_dla_n12,cb_dla_n12,chi2_dla_n12,pte_dla_n12=fit_bias_both(wth_dla_n12,wth_qso_n12,wth_pr_dlo_n12,wth_pr_qso_n12,covar_all_n12)
b_dla_n12b,cb_dla_n12b,chi2_dla_n12b,pte_dla_n12b=fit_bias_both(wth_dla_n12b,wth_qso_n12b,wth_pr_dlo_n12b,wth_pr_qso_n12b,covar_all_n12b)
b_dla_g16,cb_dla_g16,chi2_dla_g16,pte_dla_g16=fit_bias_both(wth_dla_g16,wth_qso_g16,wth_pr_dlo_g16,wth_pr_qso_g16,covar_all_g16)
line_out+="%.3lE %.3lE %.3lE %.3lE %.3lE %.3lE "%(b_dla_n12[0],b_dla_n12[1],cb_dla_n12[0,0],cb_dla_n12[0,1],cb_dla_n12[1,1],chi2_dla_n12)
line_out+="%.3lE %.3lE %.3lE %.3lE %.3lE %.3lE "%(b_dla_n12b[0],b_dla_n12b[1],cb_dla_n12b[0,0],cb_dla_n12b[0,1],cb_dla_n12b[1,1],chi2_dla_n12b)
line_out+="%.3lE %.3lE %.3lE %.3lE %.3lE %.3lE "%(b_dla_g16[0],b_dla_g16[1],cb_dla_g16[0,0],cb_dla_g16[0,1],cb_dla_g16[1,1],chi2_dla_g16)
print " Bias from simultaneous fit to QSOs and DLAs"
print "  N12"
print "   b_DLA = %.3lf +- %.3lf"%(b_dla_n12[0],np.sqrt(np.diag(cb_dla_n12))[0])
print "   b_QSO = %.3lf +- %.3lf"%(b_dla_n12[1],np.sqrt(np.diag(cb_dla_n12))[1])
print "   r = %.3lE"%(cb_dla_n12[0,1]/(cb_dla_n12[0,0]*cb_dla_n12[1,1]))
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dla_n12,2*ndof-2,pte_dla_n12)
print "  N12B"
print "   b_DLA = %.3lf +- %.3lf"%(b_dla_n12b[0],np.sqrt(np.diag(cb_dla_n12b))[0])
print "   b_QSO = %.3lf +- %.3lf"%(b_dla_n12b[1],np.sqrt(np.diag(cb_dla_n12b))[1])
print "   r = %.3lE"%(cb_dla_n12b[0,1]/(cb_dla_n12b[0,0]*cb_dla_n12b[1,1]))
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dla_n12b,2*ndof-2,pte_dla_n12b)
print "  G16"
print "   b_DLA = %.3lf +- %.3lf"%(b_dla_g16[0],np.sqrt(np.diag(cb_dla_g16))[0])
print "   b_QSO = %.3lf +- %.3lf"%(b_dla_g16[1],np.sqrt(np.diag(cb_dla_g16))[1])
print "   r = %.3lE"%(cb_dla_g16[0,1]/(cb_dla_g16[0,0]*cb_dla_g16[1,1]))
print "   chi^2 = %.3lE, ndof = %d, PTE = %.3lE"%(chi2_dla_g16,2*ndof-2,pte_dla_g16)
print " "
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    ax.plot(tharr_th,b_dla_n12[0]*1E3*wth_th_dlo_n12+b_dla_n12[1]*1E3*wth_th_qso_n12,'b-',lw=2)
    ax.plot(tharr_th,b_dla_n12[1]*1E3*wth_th_qso_n12,'r-',lw=2)
    ax.errorbar(tharr,1E3*wth_dla_n12,yerr=1E3*np.sqrt(np.diag(covar_all_n12[:nth,:nth])),fmt='bo',label='$\\kappa\\times{\\rm DLA}$')
    ax.errorbar(tharr,1E3*wth_qso_n12,yerr=1E3*np.sqrt(np.diag(covar_all_n12[nth:,nth:])),fmt='ro',label='$\\kappa\\times{\\rm QSO}$')
#    ax.plot(tharr_th,b_dla_n12b[0]*1E3*wth_th_dlo_n12b+b_dla_n12b[1]*1E3*wth_th_qso_n12b,'b-',lw=2)
#    ax.plot(tharr_th,b_dla_n12b[1]*1E3*wth_th_qso_n12b,'r-',lw=2)
#    ax.errorbar(tharr,1E3*wth_dla_n12b,yerr=1E3*np.sqrt(np.diag(covar_all_n12b[:nth,:nth])),fmt='bo',label='$\\kappa\\times{\\rm DLA}$')
#    ax.errorbar(tharr,1E3*wth_qso_n12b,yerr=1E3*np.sqrt(np.diag(covar_all_n12b[nth:,nth:])),fmt='ro',label='$\\kappa\\times{\\rm QSO}$')
#    ax.plot(tharr_th,b_dla_g16[0]*1E3*wth_th_dlo_g16+b_dla_g16[1]*1E3*wth_th_qso_g16,'b-',lw=2)
#    ax.plot(tharr_th,b_dla_g16[1]*1E3*wth_th_qso_g16,'r-',lw=2)
#    ax.errorbar(tharr,1E3*wth_dla_g16,yerr=1E3*np.sqrt(np.diag(covar_all_g16[:nth,:nth])),fmt='bo',label='$\\kappa\\times{\\rm DLA}$')
#    ax.errorbar(tharr,1E3*wth_qso_g16,yerr=1E3*np.sqrt(np.diag(covar_all_g16[nth:,nth:])),fmt='ro',label='$\\kappa\\times{\\rm QSO}$')
#    ax.plot([-1,-1],[-1,-1],'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,thmax])
    ax.set_ylim([-0.15,0.5])
    ax.set_xlabel('$\\theta\\,\\,[{\\rm deg}]$',fontsize=14)
    ax.set_ylabel('$\\left\\langle\\kappa(\\theta)\\right\\rangle\\times10^3$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
#    plt.savefig("doc/wth_x2.pdf",bbox_inches='tight')

line_out+="%.3lE %.3lE"%(1.0,1.0)
if plot_stuff :
    plt.show()
with open('data/results.txt','a') as outfile:
    outfile.write(line_out+"\n")
