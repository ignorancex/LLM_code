import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyfits as pf
import common as cmm
import sys
import scipy.stats as st
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

if len(sys.argv)!=2 :
    print "Usage: plotting.py which_fig"
    exit(1)

col_N12="#0180EF"
col_N12B="#CC0066"
col_G16="#EF9001"

if sys.argv[1]=='table' or sys.argv[1]=='all' :
    res=np.genfromtxt('data/results.txt',dtype=None)
    print "Dataset   Direct subtraction                         |  Simultaneous fit"
    print "          C_l                   wth                  |  C_l                  wth"
    print "QSU       %.3lf+-%.3lf( %.1lf%%)   %.3lf+-%.3lf(%.1lf%%)  |  %.3lf+-%.3lf(%.1lf%%)  %.3lf+-%.3lf(%.1lf%%)"%(res[10][1],res[10][2],100*(1-st.chi2.cdf(res[10][3],19)),
                                                                                                                      res[3][1],res[3][2],100*(1-st.chi2.cdf(res[3][3],31)),10,10,10,10,10,10)
    print "N12       %.3lf+-%.3lf(%.1lf%%)   %.3lf+-%.3lf(%.1lf%%)  |  %.3lf+-%.3lf(%.1lf%%)  %.3lf+-%.3lf(%.1lf%%)"%(res[10][4],res[10][5],100*(1-st.chi2.cdf(res[10][6],19)),
                                                                                                                      res[3][4],res[3][5],100*(1-st.chi2.cdf(res[3][6],31)),
                                                                                                                      res[10][13],np.sqrt(res[10][15]),100*(1-st.chi2.cdf(res[10][18],38)),
                                                                                                                      res[3][13],np.sqrt(res[3][15]),100*(1-st.chi2.cdf(res[3][18],62)))
    print "N12B      %.3lf+-%.3lf(%.1lf%%)   %.3lf+-%.3lf(%.1lf%%)  |  %.3lf+-%.3lf(%.1lf%%)  %.3lf+-%.3lf(%.1lf%%)"%(res[10][7],res[10][8],100*(1-st.chi2.cdf(res[10][9],19)),
                                                                                                                      res[3][7],res[3][8],100*(1-st.chi2.cdf(res[3][9],31)),
                                                                                                                      res[10][19],np.sqrt(res[10][21]),100*(1-st.chi2.cdf(res[10][24],38)),
                                                                                                                      res[3][19],np.sqrt(res[3][21]),100*(1-st.chi2.cdf(res[3][24],62)))
    print "G16       %.3lf+-%.3lf(%.1lf%%)   %.3lf+-%.3lf( %.1lf%%)  |  %.3lf+-%.3lf(%.1lf%%)  %.3lf+-%.3lf(%.1lf%%)"%(res[10][10],res[10][11],100*(1-st.chi2.cdf(res[10][12],19)),
                                                                                                                      res[3][10],res[3][11],100*(1-st.chi2.cdf(res[3][12],31)),
                                                                                                                      res[10][25],np.sqrt(res[10][27]),100*(1-st.chi2.cdf(res[10][30],38)),
                                                                                                                      res[3][25],np.sqrt(res[3][27]),100*(1-st.chi2.cdf(res[3][30],62)))
if sys.argv[1]=='fig1a' or sys.argv[1]=='all' :
    data_QSO=(fits.open(cmm.fname_qso))[1].data
    data_DLA_N12=(fits.open(cmm.fname_dla_n12))[1].data
    data_DLA_N12B=(fits.open(cmm.fname_dla_n12b))[1].data
    data_DLA_G16=(fits.open(cmm.fname_dla_g16))[1].data
    
    plt.figure(); ax=plt.gca()
    hn12,b=np.histogram(data_DLA_N12['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dn12=hn12; zn12=(b[1:]+b[:-1])*0.5
    ax.plot(zn12,dn12,'-',color=col_N12,lw=2,label='DLAs N12')

    hn12b,b=np.histogram(data_DLA_N12B['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dn12b=hn12b; zn12b=(b[1:]+b[:-1])*0.5
    ax.plot(zn12b,dn12b,'--',color=col_N12B,lw=2,label='DLAs N12B')

    hg16,b=np.histogram(data_DLA_G16['z_abs'],bins=30,range=[1.6,5.5],normed=True)
    dg16=hg16; zg16=(b[1:]+b[:-1])*0.5
    ax.plot(zg16,dg16,'-',color=col_G16,lw=2,label='DLAs G16')

    ax.set_xlabel('$z$',fontsize=15)
    ax.set_ylabel('$N(z),\\,\\,({\\rm normalized})$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.set_xlim([1.8,5])    
    ax.set_ylim([0,1.3])
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig('doc/nz_dla.pdf',bbox_inches='tight')


if sys.argv[1]=='fig1b' or sys.argv[1]=='all' :
    data_QSO=(fits.open(cmm.fname_qso))[1].data
    data_DLA_N12=(fits.open(cmm.fname_dla_n12))[1].data
    data_DLA_N12B=(fits.open(cmm.fname_dla_n12b))[1].data
    data_DLA_G16=(fits.open(cmm.fname_dla_g16))[1].data

    plt.figure(); ax=plt.gca()
    hqso,b=np.histogram(data_QSO['Z_PIPE'],bins=60,range=[1.6,5.5],normed=True)
    dqso=hqso; zqso=(b[1:]+b[:-1])*0.5
    ax.plot(zqso,dqso,'k-',lw=2,label='All QSOs')

    hn12,b=np.histogram(data_DLA_N12['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dn12=hn12; zn12=(b[1:]+b[:-1])*0.5
    ax.plot(zn12,dn12,'-',color=col_N12,lw=2,label='QSOs w. DLAs N12')

    hn12b,b=np.histogram(data_DLA_N12B['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dn12b=hn12b; zn12b=(b[1:]+b[:-1])*0.5
    ax.plot(zn12b,dn12b,'--',color=col_N12B,lw=2,label='QSOs w. DLAs N12B')

    hg16,b=np.histogram(data_DLA_G16['zqso'],bins=30,range=[1.6,5.5],normed=True)
    dg16=hg16; zg16=(b[1:]+b[:-1])*0.5
    ax.plot(zg16,dg16,'-',color=col_G16,lw=2,label='QSOs w. DLAs G16')

    ax.set_xlabel('$z$',fontsize=15)
    ax.set_ylabel('$N(z),\\,\\,({\\rm normalized})$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    ax.set_xlim([1.8,5])    
    ax.set_ylim([0,1.35])
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig('doc/nz_qso.pdf',bbox_inches='tight')

if sys.argv[1]=='fig2a' or sys.argv[1]=='all' :
    d=np.load("outputs_ell2_2002_ns2048_nlb50_apo0.000/cl_qxk_all.npz")
    
    igood=np.where(d['ll']<1002)[0]; nl=len(igood); nlt=len(d['ll']);
    mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
    covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
    corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])
    corr_all_n12p=np.zeros([2*nl,2*nl])
    corr_all_n12p[:nl,:nl]=corr_all_n12[:nlt,:nlt][igood,:][:,igood]
    corr_all_n12p[:nl,nl:]=corr_all_n12[:nlt,nlt:][igood,:][:,igood]
    corr_all_n12p[nl:,:nl]=corr_all_n12[nlt:,:nlt][igood,:][:,igood]
    corr_all_n12p[nl:,nl:]=corr_all_n12[nlt:,nlt:][igood,:][:,igood]

    plt.figure(); ax=plt.gca()
    #ax.set_title("$C_\ell\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_all_n12p,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.text(0.05,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes,color='w')
    ax.text(0.05,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='w')
    ax.text(0.55,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes,color='w')
    ax.text(0.55,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='w')
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_cls.pdf',bbox_inches='tight')

if sys.argv[1]=='fig2b' or sys.argv[1]=='all' :
    d=np.load("outputs_thm3.0_ns2048_nb16_wiener/wth_qxk_all.npz")
    
    mean_all_n12=np.mean(d['randoms_2'][:,0,1,:],axis=0)
    covar_all_n12=np.mean(d['randoms_2'][:,0,1,:,None]*d['randoms_2'][:,0,1,None,:],axis=0)-mean_all_n12[:,None]*mean_all_n12[None,:]
    corr_all_n12=covar_all_n12/np.sqrt(np.diag(covar_all_n12)[None,:]*np.diag(covar_all_n12)[:,None])

    plt.figure(); ax=plt.gca()
    #ax.set_title("$\\xi(\\theta)\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_all_n12,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.text(0.27,0.45,'${\\rm DLA-DLA}$',transform=ax.transAxes,color='k')
    ax.text(0.27,0.95,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='k')
    ax.text(0.77,0.95,'${\\rm QSO-QSO}$',transform=ax.transAxes,color='k')
    ax.text(0.77,0.45,'${\\rm DLA-QSO}$',transform=ax.transAxes,color='k')
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_wth.pdf',bbox_inches='tight')

if sys.argv[1]=='fig2c' or sys.argv[1]=='all' :
    d=np.load("outputs_ell2_2002_ns2048_nlb50_apo0.000/cl_qxk_all.npz")
    
    igood=np.where(d['ll']<1002)[0]; nl=len(igood); nlt=len(d['ll']);
    mean_dlo_n12=np.mean(d['randoms'][:,7,1,:],axis=0)
    covar_dlo_n12=np.mean(d['randoms'][:,7,1,:,None]*d['randoms'][:,7,1,None,:],axis=0)-mean_dlo_n12[:,None]*mean_dlo_n12[None,:]
    corr_dlo_n12=covar_dlo_n12/np.sqrt(np.diag(covar_dlo_n12)[None,:]*np.diag(covar_dlo_n12)[:,None])
    corr_dlo_n12p=corr_dlo_n12[igood,:][:,igood]

    plt.figure(); ax=plt.gca()
    #ax.set_title("$C_\ell\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_dlo_n12p,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    ax.set_xticks([0,3,6,9,12,15,18])
    ax.set_yticks([0,3,6,9,12,15,18])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_cls_b.pdf',bbox_inches='tight')

if sys.argv[1]=='fig2d' or sys.argv[1]=='all' :
    d=np.load("outputs_thm3.0_ns2048_nb16_wiener/wth_qxk_all.npz")
    
    mean_dlo_n12=np.mean(d['randoms'][:,7,1,:],axis=0)
    covar_dlo_n12=np.mean(d['randoms'][:,7,1,:,None]*d['randoms'][:,7,1,None,:],axis=0)-mean_dlo_n12[:,None]*mean_dlo_n12[None,:]
    corr_dlo_n12=covar_dlo_n12/np.sqrt(np.diag(covar_dlo_n12)[None,:]*np.diag(covar_dlo_n12)[:,None])

    plt.figure(); ax=plt.gca()
    #ax.set_title("$\\xi(\\theta)\\textrm{-based\\,\\,corr.\\,\\,matrix}$",fontsize=16)
    im=ax.imshow(corr_dlo_n12,origin='lower',interpolation='nearest',cmap=plt.get_cmap('bone'))
    cb=plt.colorbar(im,ax=ax)
    ax.set_xlabel('${\\rm bin}\\,i$',fontsize=15)
    ax.set_ylabel('${\\rm bin}\\,j$',fontsize=15)
    ax.set_xticks([0,3,6,9,12,15])
    ax.set_yticks([0,3,6,9,12,15])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.savefig('doc/corrmat_wth_b.pdf',bbox_inches='tight')

if sys.argv[1]=='fig3a' or sys.argv[1]=='all' :
    outdir="outputs_ell2_2002_ns2048_nlb50_apo0.000/"
    d=np.load(outdir+"cl_qxk_all.npz")
    res=np.genfromtxt('data/results.txt',dtype=None)
    b_DLO=res[10][4]; s_DLO=res[10][5]
    b_DLA=res[10][13]; b_QSO=res[10][14];
    s_DLA=np.sqrt(res[10][15]);  s_QSO=np.sqrt(res[10][17]); 

    larr_th,cl_dc_n12,cl_qc_n12,cl_dc_n12b,cl_qc_n12b,cl_dc_g16,cl_qc_g16,cl_uc=np.loadtxt(outdir+"cls_th.txt",unpack=True)
    
    larr=d['ll']; nl=len(larr)
    cell_dla_n12=d['cell_dla_n12']
    cell_qso_n12=d['cell_qso_n12']
    cell_dlo_n12=cell_dla_n12-cell_qso_n12;
    sell_dla_n12=np.std(d['randoms_2'][:,0,1,:],axis=0)[:nl]
    sell_qso_n12=np.std(d['randoms_2'][:,0,1,:],axis=0)[nl:]
    sell_dlo_n12=np.std(d['randoms'][:,7,1,:],axis=0)
    plt.figure(); ax=plt.gca()
    ax.errorbar(larr,cell_dla_n12,yerr=sell_dla_n12,fmt='ro',label='DLA+QSO')
    ax.errorbar(larr,cell_qso_n12,yerr=sell_qso_n12,fmt='bs',label='QSO')
    ax.errorbar(larr,cell_dlo_n12,yerr=sell_dlo_n12,fmt='kd',label='DLA')
    ax.plot(larr_th,b_DLA*cl_dc_n12+b_QSO*cl_qc_n12,'r-')
    ax.plot(larr_th,b_QSO*cl_qc_n12,'b-')
    ax.plot(larr_th,b_DLO*cl_dc_n12,'k-',label='Best-fit')
    ax.plot([0,600],[0,0],'k--',lw=1)
    ax.set_xlim([40,600])
    ax.set_ylim([-2E-7,6E-7])
    ax.set_xlabel('$\\ell$',fontsize=15)
    ax.set_ylabel('$C^{\\kappa,\\alpha}_\\ell$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig("doc/cls_result.pdf",bbox_inches='tight')

if sys.argv[1]=='fig3b' or sys.argv[1]=='all' :
    outdir="outputs_thm3.0_ns2048_nb16_wiener/"
    d=np.load(outdir+"wth_qxk_all.npz")
    res=np.genfromtxt('data/results.txt',dtype=None)
    b_DLO=res[3][4]; s_DLO=res[3][5]
    b_DLA=res[3][13]; b_QSO=res[3][14];
    s_DLA=np.sqrt(res[3][15]);  s_QSO=np.sqrt(res[3][17]); 

    tharr_th,wth_th_dlo_n12,wth_th_qso_n12,wth_th_dlo_n12b,wth_th_qso_n12b,wth_th_dlo_g16,wth_th_qso_g16,wth_th_qsu=np.loadtxt(outdir+"wth_th.txt",unpack=True)
    
    tharr=d['th']; nth=len(tharr)
    wth_dla_n12=d['wth_dla_n12'];
    wth_qso_n12=d['wth_qso_n12'];
    wth_dlo_n12=wth_dla_n12-wth_qso_n12
    sth_dla_n12=np.std(d['randoms_2'][:,0,1,:],axis=0)[:nth]
    sth_qso_n12=np.std(d['randoms_2'][:,0,1,:],axis=0)[nth:]
    sth_dlo_n12=np.std(d['randoms'][:,7,1,:],axis=0)
    plt.figure(); ax=plt.gca()
    ax.errorbar(tharr,1E3*wth_dla_n12,yerr=1E3*sth_dla_n12,fmt='ro',label='DLA+QSO')
    ax.errorbar(tharr,1E3*wth_qso_n12,yerr=1E3*sth_qso_n12,fmt='bs',label='QSO')
    ax.errorbar(tharr,1E3*wth_dlo_n12,yerr=1E3*sth_dlo_n12,fmt='kd',label='DLA')
    ax.plot(tharr_th,b_DLA*1E3*wth_th_dlo_n12+b_QSO*1E3*wth_th_qso_n12,'r-')
    ax.plot(tharr_th,b_QSO*1E3*wth_th_qso_n12,'b-')
    ax.plot(tharr_th,b_DLO*1E3*wth_th_dlo_n12,'k-',label='Best-fit')
    #ax.plot([0,3,[0,0],'k--',lw=1)
    ax.set_xlim([0,3])
    ax.set_ylim([-0.15,0.5])
    ax.set_xlabel('$\\theta\\,(^\\circ)$',fontsize=15)
    ax.set_ylabel('$10^3\\times\\xi^{\\kappa,\\alpha}(\\theta)$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig("doc/wth_result.pdf",bbox_inches='tight')

if sys.argv[1]=='fig4' or sys.argv[1]=='all' :
    outdir="outputs_ell2_2002_ns2048_nlb50_apo0.000/"
    d=np.load(outdir+"cl_qxk_all.npz")
    res=np.genfromtxt('data/results.txt',dtype=None)
    b_DLO_N12=res[10][4]; s_DLO_N12=res[10][5]
    b_DLO_N12B=res[10][7]; s_DLO_N12B=res[10][8]
    b_DLO_G16=res[10][10]; s_DLO_G16=res[10][11]

    larr_th,cl_dc_n12,cl_qc_n12,cl_dc_n12b,cl_qc_n12b,cl_dc_g16,cl_qc_g16,cl_uc=np.loadtxt(outdir+"cls_th.txt",unpack=True)
    
    larr=d['ll']; nl=len(larr)
    cell_dla_n12=d['cell_dla_n12']; cell_qso_n12=d['cell_qso_n12'];
    cell_dlo_n12=cell_dla_n12-cell_qso_n12; sell_dlo_n12=np.std(d['randoms'][:,7,1,:],axis=0)
    cell_dla_n12b=d['cell_dla_n12b']; cell_qso_n12b=d['cell_qso_n12b'];
    cell_dlo_n12b=cell_dla_n12b-cell_qso_n12b; sell_dlo_n12b=np.std(d['randoms'][:,8,1,:],axis=0)
    cell_dla_g16=d['cell_dla_g16']; cell_qso_g16=d['cell_qso_g16'];
    cell_dlo_g16=cell_dla_g16-cell_qso_g16; sell_dlo_g16=np.std(d['randoms'][:,9,1,:],axis=0)
    
    plt.figure(); ax=plt.gca()
    ax.errorbar(larr,cell_dlo_n12,yerr=sell_dlo_n12,fmt='o',color=col_N12,label='N12')
    ax.errorbar(larr,cell_dlo_n12b,yerr=sell_dlo_n12,fmt='s',color=col_N12B,label='N12B')
    ax.errorbar(larr,cell_dlo_g16,yerr=sell_dlo_n12,fmt='d',color=col_G16,label='G16')
    ax.plot(larr_th,b_DLO_N12*cl_dc_n12,'-',color=col_N12)
    ax.plot(larr_th,b_DLO_G16*cl_dc_g16,'-',color=col_G16)
    ax.plot(larr_th,b_DLO_N12B*cl_dc_n12b,'--',color=col_N12B)
    ax.plot([-1,-1],[-1,-1],'k-',label='Best-fit')
    ax.plot([0,600],[0,0],'k--',lw=1)
    ax.set_xlim([40,600])
    ax.set_ylim([-2E-7,4.5E-7])
    ax.set_xlabel('$\\ell$',fontsize=15)
    ax.set_ylabel('$C^{\\kappa,{\\rm DLA}}_\\ell$',fontsize=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc='upper right',frameon=False,fontsize=14)
    plt.savefig("doc/cls_syst.pdf",bbox_inches='tight')

if sys.argv[1]=='fig5' or sys.argv[1]=='all' :
    bdla=2.
    #col_Planck=col_N12
    #col_S3=col_N12B
    #col_S4=col_G16
    #col_Planck="#CC0000"
    #col_S3="#004C99"
    #col_S4="#D8BB00"
    col_Planck="#CC0000"
    col_S3="#D8BB00"
    col_S4="#770077"

    data=np.load("4casts.npy").item()
    def add_rectangle(x,s,ax,fc,label=None) :
        ax.fill_between([x-0.25,x+0.25],[bdla-s,bdla-s],[bdla+s,bdla+s],facecolor=fc,edgecolor=None,label=label)
    plt.figure()
    ax=plt.gca()
    add_rectangle(1,data['sdss']['planck'],ax,col_Planck,label='Planck')
    add_rectangle(2,data['desi']['planck'],ax,col_Planck)
    add_rectangle(3,data['4most']['planck'],ax,col_Planck)
    add_rectangle(1,data['sdss']['s3'],ax,col_S3,label='S3')
    add_rectangle(2,data['desi']['s3'],ax,col_S3)
    add_rectangle(3,data['4most']['s3'],ax,col_S3)
    add_rectangle(1,data['sdss']['s4'],ax,col_S4,label='S4')
    add_rectangle(2,data['desi']['s4'],ax,col_S4)
    add_rectangle(3,data['4most']['s4'],ax,col_S4)
    ax.plot([0.5,3.5],[bdla,bdla],'k-',lw=1)
    ax.plot([0.5,3.5],[1.5,1.5],'k--',lw=1)
    ax.set_xlim([0.5,3.5])
    ax.set_xlabel('Galaxy survey',fontsize=15)
    ax.set_ylabel('$\\sigma\\left(b_{\\rm DLA}\\right)$',fontsize=15)
    plt.yticks([1.2,1.6,2.0,2.4,2.8])#,['SDSS','DESI','4MOST'])
    plt.xticks([1,2,3],['SDSS','DESI','4MOST'])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.subplots_adjust(bottom=0.15)
    plt.legend(loc='upper right')
    plt.savefig("doc/4casts.pdf",bbox_inches='tight')
    
plt.show()
