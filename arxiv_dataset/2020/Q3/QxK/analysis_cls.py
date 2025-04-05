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
    print "Usage : analysis_cls.py nlb nside aposcale lthr plot_stuff[0,1]"
    exit(1)

verbose=False
#nlb
nlb=int(sys.argv[1])
#Resolution of the CMB lensing map
nside=int(sys.argv[2])
#Apodization scale in degrees
aposcale=float(sys.argv[3])
#Maximum scale to use in the analysis
l_thr=float(sys.argv[4])
#Plot stuff
do_plot_stuff=int(sys.argv[5])




plot_stuff=False
if do_plot_stuff>0 :
    plot_stuff=True

#Create output directory
outdir="outputs_ell2_2002_ns%d_nlb%d_apo%.3lf"%(nside,nlb,aposcale)
outdir+="/"

line_out="Cl_Ns%d_Nl%d_apo%.1lf "%(nside,nlb,aposcale)

fname_alldata=outdir+"cl_qxk_all"

data_dla_n12=(fits.open(cmm.fname_dla_n12))[1].data
data_dla_n12b=(fits.open(cmm.fname_dla_n12b))[1].data
data_dla_g16=(fits.open(cmm.fname_dla_g16))[1].data
data_qso=(fits.open(cmm.fname_qso))[1].data

d=np.load(fname_alldata+'.npz')

larr=d['ll']
cell_dla_n12=d['cell_dla_n12']; cell_qso_n12=d['cell_qso_n12']; cell_dlo_n12=cell_dla_n12-cell_qso_n12
cell_dla_n12b=d['cell_dla_n12b']; cell_qso_n12b=d['cell_qso_n12b']; cell_dlo_n12b=cell_dla_n12b-cell_qso_n12b
cell_dla_g16=d['cell_dla_g16']; cell_qso_g16=d['cell_qso_g16']; cell_dlo_g16=cell_dla_g16-cell_qso_g16
cell_qsu=d['cell_qsu']
nell=len(larr)

if verbose :
    print " Computing covariance matrices"
nsims=len(d['randoms'])
mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])
mean_all_n12b=np.mean(d['randoms_2'][:,1,1,:],axis=0)
covar_all_n12b=np.mean(d['randoms_2'][:,1,1,:,None]*d['randoms_2'][:,1,1,None,:],axis=0)-mean_all_n12b[:,None]*mean_all_n12b[None,:]
corr_all_n12b=covar_all_n12b/np.sqrt(np.diag(covar_all_n12b)[None,:]*np.diag(covar_all_n12b)[:,None])
mean_all_g16=np.mean(d['randoms_2'][:,2,1,:],axis=0)
covar_all_g16=np.mean(d['randoms_2'][:,2,1,:,None]*d['randoms_2'][:,2,1,None,:],axis=0)-mean_all_g16[:,None]*mean_all_g16[None,:]
corr_all_g16=covar_all_g16/np.sqrt(np.diag(covar_all_g16)[None,:]*np.diag(covar_all_g16)[:,None])

'''
mean_n12_n12b=np.zeros(4*nell)
mean_n12_g16 =np.zeros(4*nell)
covar_n12_n12b=np.zeros([4*nell,4*nell])
covar_n12_g16=np.zeros([4*nell,4*nell])
for i in np.arange(nsims) :
    print i
    vec=np.zeros(4*nell)
    vec2=np.zeros([4*nell,4*nell])

    vec[0*nell:1*nell]=d['randoms'][i,0,1,:];
    vec[1*nell:2*nell]=d['randoms'][i,3,1,:];
    vec[2*nell:3*nell]=d['randoms'][i,1,1,:];
    vec[3*nell:4*nell]=d['randoms'][i,4,1,:];
    vec2=vec[:,None]*vec[None,:]
    mean_n12_n12b+=vec
    covar_n12_n12b+=vec2

    vec[0*nell:1*nell]=d['randoms'][i,0,1,:];
    vec[1*nell:2*nell]=d['randoms'][i,3,1,:];
    vec[2*nell:3*nell]=d['randoms'][i,2,1,:];
    vec[3*nell:4*nell]=d['randoms'][i,5,1,:];
    vec2=vec[:,None]*vec[None,:]
    mean_n12_g16+=vec
    covar_n12_g16+=vec2
mean_n12_n12b/=nsims
mean_n12_g16/=nsims
covar_n12_n12b=covar_n12_n12b/nsims-mean_n12_n12b[:,None]*mean_n12_n12b[None,:]
covar_n12_g16=covar_n12_g16/nsims-mean_n12_g16[:,None]*mean_n12_g16[None,:]
'''

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
    def plot_corr(mat,name) :
        plt.figure()
        ax=plt.gca()
        ax.set_title(name,fontsize=16)
        im=ax.imshow(mat,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
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
    cl_dc_n12=ccl.angular_cl(cosmo,clt_dlo_n12,clt_cmbl,larr_b,l_limber=-1) 
    cl_qc_n12=ccl.angular_cl(cosmo,clt_qso_n12,clt_cmbl,larr_b,l_limber=-1)
    cl_dc_n12b=ccl.angular_cl(cosmo,clt_dlo_n12b,clt_cmbl,larr_b,l_limber=-1)
    cl_qc_n12b=ccl.angular_cl(cosmo,clt_qso_n12b,clt_cmbl,larr_b,l_limber=-1)
    cl_dc_g16=ccl.angular_cl(cosmo,clt_dlo_g16,clt_cmbl,larr_b,l_limber=-1)
    cl_qc_g16=ccl.angular_cl(cosmo,clt_qso_g16,clt_cmbl,larr_b,l_limber=-1)
    cl_uc=ccl.angular_cl(cosmo,clt_qsu,clt_cmbl,larr_b,l_limber=-1)
    np.savetxt(outdir+"cls_th.txt",np.transpose([larr_b,
                                                 cl_dc_n12,cl_qc_n12,
                                                 cl_dc_n12b,cl_qc_n12b,
                                                 cl_dc_g16,cl_qc_g16,
                                                 cl_uc]))
larr_th,cl_dc_n12,cl_qc_n12,cl_dc_n12b,cl_qc_n12b,cl_dc_g16,cl_qc_g16,cl_uc=np.loadtxt(outdir+"cls_th.txt",unpack=True)

#Binning theory
cl_th_dlo_n12=cl_dc_n12
cl_f_dlo_n12=interp1d(larr_th,cl_th_dlo_n12,bounds_error=False,fill_value=0)
cl_pr_dlo_n12=np.array([quad(cl_f_dlo_n12,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_qso_n12=cl_qc_n12
cl_f_qso_n12=interp1d(larr_th,cl_th_qso_n12,bounds_error=False,fill_value=0)
cl_pr_qso_n12=np.array([quad(cl_f_qso_n12,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_dlo_n12b=cl_dc_n12b
cl_f_dlo_n12b=interp1d(larr_th,cl_th_dlo_n12b,bounds_error=False,fill_value=0)
cl_pr_dlo_n12b=np.array([quad(cl_f_dlo_n12b,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_qso_n12b=cl_qc_n12b
cl_f_qso_n12b=interp1d(larr_th,cl_th_qso_n12b,bounds_error=False,fill_value=0)
cl_pr_qso_n12b=np.array([quad(cl_f_qso_n12b,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_dlo_g16=cl_dc_g16
cl_f_dlo_g16=interp1d(larr_th,cl_th_dlo_g16,bounds_error=False,fill_value=0)
cl_pr_dlo_g16=np.array([quad(cl_f_dlo_g16,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_qso_g16=cl_qc_g16
cl_f_qso_g16=interp1d(larr_th,cl_th_qso_g16,bounds_error=False,fill_value=0)
cl_pr_qso_g16=np.array([quad(cl_f_qso_g16,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])
cl_th_qsu=cl_uc
cl_f_qsu=interp1d(larr_th,cl_th_qsu,bounds_error=False,fill_value=0)
cl_pr_qsu=np.array([quad(cl_f_qsu,l-nlb*0.5,l+nlb*0.5)[0]/nlb for l in larr])

#Binning theory
if plot_stuff :
    plt.figure()
    plt.plot(larr_th,cl_dc_n12,'r-')
    plt.plot(larr_th,cl_dc_n12b,'r--')
    plt.plot(larr_th,cl_dc_g16,'r-.')
    plt.plot(larr_th,cl_qc_n12,'b-')
    plt.plot(larr_th,cl_qc_n12b,'b--')
    plt.plot(larr_th,cl_qc_g16,'b-.')
    plt.plot(larr_th,cl_uc,'k-')
    plt.loglog()

i_good=np.where(larr<l_thr)[0]; ndof=len(i_good);

#Fitting the 2PCF difference
def fit_bias_single(cell_dlo,cl_pr_dlo,covar_dlo) :
    #Data vectors and covariances
    dv=cell_dlo[i_good]; tv=cl_pr_dlo[i_good]; cv=(covar_dlo[i_good,:])[:,i_good]; icv=np.linalg.inv(cv)
    #chi^2
    #Analytic Best-fit and errors
    sigma_b=1./np.sqrt(np.dot(tv,np.dot(icv,tv)))
    b_bf=np.dot(tv,np.dot(icv,dv))/np.dot(tv,np.dot(icv,tv))
    pv=b_bf*tv
    chi2=np.dot(dv-pv,np.dot(icv,dv-pv))
    pte=1-st.chi2.cdf(chi2,ndof-1)
    return b_bf,sigma_b,chi2,pte

b_dlo_n12,sb_dlo_n12,chi2_dlo_n12,pte_dlo_n12=fit_bias_single(cell_dlo_n12,cl_pr_dlo_n12,covar_dlo_n12)
b_dlo_n12b,sb_dlo_n12b,chi2_dlo_n12b,pte_dlo_n12b=fit_bias_single(cell_dlo_n12b,cl_pr_dlo_n12b,covar_dlo_n12b)
b_dlo_g16,sb_dlo_g16,chi2_dlo_g16,pte_dlo_g16=fit_bias_single(cell_dlo_g16,cl_pr_dlo_g16,covar_dlo_g16)
b_qsu,sb_qsu,chi2_qsu,pte_qsu=fit_bias_single(cell_qsu,cl_pr_qsu,covar_qsu)
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
    ax.errorbar(larr,cell_dlo_n12,yerr=np.sqrt(np.diag(covar_dlo_n12)),fmt='ro',
                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
    ax.plot(larr_th,b_dlo_n12*cl_th_dlo_n12,'r-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.errorbar(larr,cell_qsu,yerr=np.sqrt(np.diag(covar_qsu)),fmt='ko',
                label='$\\kappa\\times({\\rm DLA}-{\\rm QSO})$')
    ax.plot(larr_th,b_qsu*cl_th_qsu,'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,l_thr])
#    ax.set_ylim([-0.1,0.35])
    ax.set_xlabel('$\\ell$',fontsize=14)
    ax.set_ylabel('$C^{\\kappa\\delta}_\\ell$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
#    plt.legend(loc='upper right',frameon=False,fontsize=14)

#Fitting both 2PCFs with b_DLA and b_QSO
def fit_bias_both(cell_dla,cell_qso,cl_pr_dlo,cl_pr_qso,covar_all) :
    dv=np.concatenate((cell_dla[i_good],cell_qso[i_good]));
    tv1=np.concatenate((cl_pr_dlo[i_good],np.zeros(ndof)));
    tv2=np.concatenate((cl_pr_qso[i_good],cl_pr_qso[i_good]))
    tv=np.array([tv1,tv2])
    cv=np.zeros([2*ndof,2*ndof])
    cv[:ndof,:ndof]=covar_all[:ndof,:ndof]
    cv[:ndof,ndof:]=covar_all[:ndof,nell:nell+ndof]
    cv[ndof:,:ndof]=covar_all[nell:nell+ndof,:ndof]
    cv[ndof:,ndof:]=covar_all[nell:nell+ndof,nell:nell+ndof]
    icv=np.linalg.inv(cv)

    cov_b=np.linalg.inv(np.dot(tv,np.dot(icv,np.transpose(tv))))
    b_bf=np.dot(cov_b,np.dot(tv,np.dot(icv,np.transpose(dv))))
    sigma_b=np.sqrt(np.diag(cov_b))
    pv=np.dot(b_bf,tv);
    chi2=np.dot((dv-pv),np.dot(icv,(dv-pv)));
    pte=1-st.chi2.cdf(chi2,2*ndof-2)
    return b_bf,cov_b,chi2,pte

b_dla_n12,cb_dla_n12,chi2_dla_n12,pte_dla_n12=fit_bias_both(cell_dla_n12,cell_qso_n12,cl_pr_dlo_n12,cl_pr_qso_n12,covar_all_n12)
b_dla_n12b,cb_dla_n12b,chi2_dla_n12b,pte_dla_n12b=fit_bias_both(cell_dla_n12b,cell_qso_n12b,cl_pr_dlo_n12b,cl_pr_qso_n12b,covar_all_n12b)
b_dla_g16,cb_dla_g16,chi2_dla_g16,pte_dla_g16=fit_bias_both(cell_dla_g16,cell_qso_g16,cl_pr_dlo_g16,cl_pr_qso_g16,covar_all_g16)
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
    ax.plot(larr_th,b_dla_n12[0]*cl_th_dlo_n12+b_dla_n12[1]*cl_th_qso_n12,'b-',lw=2)
    ax.plot(larr_th,b_dla_n12[1]*cl_th_qso_n12,'r-',lw=2)
    ax.errorbar(larr,cell_dla_n12,yerr=np.sqrt(np.diag(covar_all_n12[:nell,:nell])),fmt='bo',label='$\\kappa\\times{\\rm DLA-N12}$')
    ax.errorbar(larr,cell_qso_n12,yerr=np.sqrt(np.diag(covar_all_n12[nell:,nell:])),fmt='ro',label='$\\kappa\\times{\\rm QSO-N12}$')
    #ax.plot(larr_th,b_dla_n12b[0]*cl_th_dlo_n12b+b_dla_n12b[1]*cl_th_qso_n12b,'g-',lw=2)
    #ax.plot(larr_th,b_dla_n12b[1]*cl_th_qso_n12b,'y-',lw=2)
    #ax.errorbar(larr,cell_dla_n12b,yerr=np.sqrt(np.diag(covar_all_n12b[:nell,:nell])),fmt='go',label='$\\kappa\\times{\\rm DLA-N12B}$')
    #ax.errorbar(larr,cell_qso_n12b,yerr=np.sqrt(np.diag(covar_all_n12b[nell:,nell:])),fmt='yo',label='$\\kappa\\times{\\rm QSO-N12B}$')
    #ax.plot(larr_th,b_dla_g16[0]*cl_th_dlo_g16+b_dla_g16[1]*cl_th_qso_g16,'m-',lw=2)
    #ax.plot(larr_th,b_dla_g16[1]*cl_th_qso_g16,'k-',lw=2)
    #ax.errorbar(larr,cell_dla_g16,yerr=np.sqrt(np.diag(covar_all_g16[:nell,:nell])),fmt='mo',label='$\\kappa\\times{\\rm DLA-G16}$')
    #ax.errorbar(larr,cell_qso_g16,yerr=np.sqrt(np.diag(covar_all_g16[nell:,nell:])),fmt='ko',label='$\\kappa\\times{\\rm QSO-G16}$')
    #ax.plot([-1,-1],[-1,-1],'k-',lw=2,label='${\\rm best\\,\\,fit}$')
    ax.set_xlim([0,600])
    ax.set_ylim([-4E-7,8E-7])
    ax.set_xlabel('$\\ell$',fontsize=14)
    ax.set_ylabel('$C^{\\kappa\\delta}_\\ell$',fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
line_out+="%.3lE %.3lE"%(1.0,1.0)
if plot_stuff :
    plt.show()
with open('data/results.txt','a') as outfile:
    outfile.write(line_out+"\n")
exit(1)

def fit_bias_4way(cell_dla1,cell_qso1,cell_dla2,cell_qso2,cl_pr_dlo1,cl_pr_qso1,cl_pr_dlo2,cl_pr_qso2,covar_all) :
    dv=np.concatenate((cell_dla1[i_good],cell_qso1[i_good],cell_dla2[i_good],cell_qso2[i_good]))
    tv1=np.concatenate((cl_pr_dlo1[i_good],np.zeros(ndof),np.zeros(ndof),np.zeros(ndof)))
    tv2=np.concatenate((np.zeros(ndof),np.zeros(ndof),cl_pr_dlo2[i_good],np.zeros(ndof)))
    tv3=np.concatenate((cl_pr_qso1[i_good],cl_pr_qso1[i_good],np.zeros(ndof),np.zeros(ndof)))
    tv4=np.concatenate((np.zeros(ndof),np.zeros(ndof),cl_pr_qso2[i_good],cl_pr_qso2[i_good]))
    tv=np.array([tv1,tv2,tv3,tv4])
    cv=np.zeros([4*ndof,4*ndof])
    for i in np.arange(4) :
        for j in np.arange(4) :
            cv[i*ndof:(i+1)*ndof,j*ndof:(j+1)*ndof]=covar_all[i*nell:i*nell+ndof,j*nell:j*nell+ndof]
    icv=np.linalg.inv(cv)
    cov_b=np.linalg.inv(np.dot(tv,np.dot(icv,np.transpose(tv))))
    b_bf=np.dot(cov_b,np.dot(tv,np.dot(icv,np.transpose(dv))))
    
    cb=np.array([[cov_b[0,0],cov_b[0,1]],[cov_b[1,0],cov_b[1,1]]])
    db=np.array([b_bf[0],b_bf[1]])
    u=np.ones(2); icb=np.linalg.inv(cb); 
    mb=np.dot(u,np.dot(icb,db))/np.dot(u,np.dot(icb,u))
    chi2=np.dot((db-mb*u),np.dot(icb,(db-mb*u)))

    print b_bf[0],b_bf[1],np.sqrt(cov_b[0,0]),np.sqrt(cov_b[1,1]),cov_b[0,1]/np.sqrt(cov_b[0,0]*cov_b[1,1]),chi2,1-st.chi2.cdf(chi2,1)
    return 1-st.chi2.cdf(chi2,1)
print "N12-N12B"
pte_n12_n12b=fit_bias_4way(cell_dla_n12,cell_qso_n12,cell_dla_n12b,cell_qso_n12b,cl_pr_dlo_n12,cl_pr_qso_n12,cl_pr_dlo_n12b,cl_pr_qso_n12b,covar_n12_n12b)
print "N12-G16"
pte_n12_g16=fit_bias_4way(cell_dla_n12,cell_qso_n12,cell_dla_g16,cell_qso_g16,cl_pr_dlo_n12,cl_pr_qso_n12,cl_pr_dlo_g16,cl_pr_qso_g16,covar_n12_g16)
line_out+="%.3lE %.3lE"%(pte_n12_n12b,pte_n12_g16)
if plot_stuff :
    plt.show()
with open('data/results.txt','a') as outfile:
    outfile.write(line_out+"\n")
