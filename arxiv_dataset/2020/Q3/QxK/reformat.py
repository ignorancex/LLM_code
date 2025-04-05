import numpy as np
import healpy as hp
import os
import pyfits as pf
import healpy as hp
from astropy.io import fits
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import common as cmm

r=hp.Rotator(coord=['C','G'])

print " Generating CMB lensing products"
for nside in [1024,2048] :
    for w_wiener in [True,False] :
        fname_mask=cmm.fname_mask_cmbl_prefix+"_%d"%nside
        if w_wiener :
            fname_mask+="_wiener"
        fname_mask+=".fits"
        fname_kappa=cmm.fname_kappa_cmbl_prefix+"_%d"%nside
        if w_wiener :
            fname_kappa+="_wiener"
        fname_kappa+=".fits"

        #Mask
        if not os.path.isfile(fname_mask) :
            print "   Generating mask "+fname_mask
            mask=hp.ud_grade(hp.read_map(cmm.fname_mask_cmbl_orig,verbose=False),nside_out=nside)
            mask[mask<1.0]=0
            hp.write_map(fname_mask,mask)

        #Kappa
        if not os.path.isfile(fname_kappa) :
            print "   Generating kappa "+fname_kappa
            almk=hp.read_alm(cmm.fname_alm_cmbl)
            if w_wiener :
                ll,nll,cll=np.loadtxt(cmm.fname_kappa_cl,unpack=True)
                cl=np.zeros(int(ll[-1]+1)); cl[int(ll[0]):]=cll
                nl=np.zeros(int(ll[-1]+1)); nl[int(ll[0]):]=nll
                wl=(cl-nl)/np.maximum(cl,np.ones_like(cl)*1E-10)
                almk=hp.almxfl(almk,wl)
            k=hp.alm2map(almk,nside,verbose=False)
            k*=hp.read_map(fname_mask,verbose=False)
            hp.write_map(fname_kappa,k)

print " Generating QSO products"
if not (os.path.isfile(cmm.fname_qso) and 
        os.path.isfile(cmm.fname_dla_n12) and os.path.isfile(cmm.fname_dla_n12b) and 
        os.path.isfile(cmm.fname_dla_g16) and os.path.isfile(cmm.fname_mask_qso)) :

        print "  DLA"
        #Read G16 catalog
        data_dla_g16=np.genfromtxt(cmm.fname_dla_g16_orig,
                                   dtype='i8,S18,i8,i8,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                                   names=['ThingID','SDSS','Plate','MJD','Fiber','RA','DEC','zqso',
                                          'SNRSpec','b_z_DLA','B_z_DLA','logpn','logpy','logpn_y','logpy_y',
                                          'pn','py','z_abs','log_NHI'])
        #Quality cut
        is_good=np.where(data_dla_g16['py']>0.9)[0]
        data_dla_g16=data_dla_g16[is_good]
        th_dla_g16_c =(90-data_dla_g16['DEC'])*np.pi/180.
        phi_dla_g16_c=data_dla_g16['RA']*np.pi/180.
        th_dla_g16_g,phi_dla_g16_g=r(th_dla_g16_c,phi_dla_g16_c)
        b_dla_g16=90-180*th_dla_g16_g/np.pi
        l_dla_g16=180*phi_dla_g16_g/np.pi
        ndla_g16=len(th_dla_g16_g)        
    
        #Read N12 catalog
        data_dla_n12=np.genfromtxt(cmm.fname_dla_n12_orig,skip_header=2,
                                   dtype='i8,S16,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                                   names=['ThingID','MJD-plate-fiber','RA','DEC','zqso','fDLA','fBAL',
                                          'BI','CNR','z_abs','NHI','cc','log(Pcc)','saap','Dcore','Pcore',
                                          'Fcore','Ecore','Ccore','fl','CII_1334','SiII_1526','FeII_1608',
                                          'AlII_1670','FeII_2344','FeII_2374','FeII_2382','FeII_2586',
                                          'FeII_2600','MgII_2796','MgII_2803','MgI_2852'])
        th_dla_n12_c =(90-data_dla_n12['DEC'])*np.pi/180.
        phi_dla_n12_c=data_dla_n12['RA']*np.pi/180.
        th_dla_n12_g,phi_dla_n12_g=r(th_dla_n12_c,phi_dla_n12_c)
        b_dla_n12=90-180*th_dla_n12_g/np.pi
        l_dla_n12=180*phi_dla_n12_g/np.pi
        ndla_n12=len(th_dla_n12_g)

        #Select N12B sample
        n12b=np.where(data_dla_n12['CNR']>2)[0]
        data_dla_n12b=data_dla_n12[n12b]
        th_dla_n12b_c =(90-data_dla_n12b['DEC'])*np.pi/180.
        phi_dla_n12b_c=data_dla_n12b['RA']*np.pi/180.
        th_dla_n12b_g,phi_dla_n12b_g=r(th_dla_n12b_c,phi_dla_n12b_c)
        b_dla_n12b=90-180*th_dla_n12b_g/np.pi
        l_dla_n12b=180*phi_dla_n12b_g/np.pi
        ndla_n12b=len(data_dla_n12b)
        
        print "  QSO"
        #Read QSO data
        data_qso=(fits.open(cmm.fname_qso_orig))[1].data
        th_qso_c =(90-data_qso['DEC'])*np.pi/180.
        phi_qso_c=data_qso['RA']*np.pi/180.
        th_qso_g,phi_qso_g=r(th_qso_c,phi_qso_c)
        b_qso=90-180*th_qso_g/np.pi
        l_qso=180*phi_qso_g/np.pi
        nqso=len(th_qso_g)

        #Create QSO mask
        print "  Mask"
        ipix_qso=hp.ang2pix(cmm.nside_qso,th_qso_g,phi_qso_g)
        mp_qso,bins=np.histogram(ipix_qso,range=[0,hp.nside2npix(cmm.nside_qso)],bins=hp.nside2npix(cmm.nside_qso))
        mask_qso=np.zeros(hp.nside2npix(cmm.nside_qso)); mask_qso[mp_qso>0]=1;
        #Write QSO mask
        hp.write_map(cmm.fname_mask_qso,mask_qso)

        #Compute QSO weights
        pz_qso,bins=np.histogram(data_qso['Z_PIPE'],range=[-0.1,7.5],bins=50,normed=True)
        pz_parent_n12,bins=np.histogram(data_dla_n12['zqso'  ],range=[-0.1,7.5],bins=50,normed=True)
        pz_dla_n12,bins=np.histogram(data_dla_n12['z_abs' ],range=[-0.1,7.5],bins=50,normed=True)
        pz_parent_g16,bins=np.histogram(data_dla_g16['zqso' ],range=[-0.1,7.5],bins=50,normed=True)
        pz_parent_n12b,bins=np.histogram(data_dla_n12b['zqso' ],range=[-0.1,7.5],bins=50,normed=True)
        zarr=0.5*(bins[1:]+bins[:-1])

        igood=np.where(pz_qso>0)[0]
        w_n12_t=np.zeros_like(pz_qso);  w_n12_t[igood]=pz_parent_n12[igood]/pz_qso[igood]; w_n12_t[zarr>5.5]=0
        w_g16_t=np.zeros_like(pz_qso);  w_g16_t[igood]=pz_parent_g16[igood]/pz_qso[igood]; w_g16_t[zarr>5.5]=0
        w_n12b_t=np.zeros_like(pz_qso); w_n12b_t[igood]=pz_parent_n12b[igood]/pz_qso[igood]; w_n12b_t[zarr>5.5]=0
        wfunc=interp1d(zarr,w_n12_t ,bounds_error=False,fill_value=0,kind='nearest'); w_qso_n12 =wfunc(data_qso['Z_PIPE'])
        wfunc=interp1d(zarr,w_g16_t ,bounds_error=False,fill_value=0,kind='nearest'); w_qso_g16 =wfunc(data_qso['Z_PIPE'])
        wfunc=interp1d(zarr,w_n12b_t,bounds_error=False,fill_value=0,kind='nearest'); w_qso_n12b=wfunc(data_qso['Z_PIPE'])

        plt.figure()
        plt.plot(zarr,w_n12_t ,'r-',label='QSO weights, N12')
        plt.plot(zarr,w_n12b_t,'g-',label='QSO weights, N12B')
        plt.plot(zarr,w_g16_t ,'b-',label='QSO weights, G16')
        plt.legend(loc='upper left')
        plt.xlabel('$z$',fontsize=16)
        plt.ylabel('$w(z)$',fontsize=16)
        plt.savefig('doc/weights.pdf',bbox_inches='tight')

        plt.figure()
        plt.plot(zarr,pz_qso        ,'k-',label='QSO-ALL')
        plt.plot(zarr,pz_parent_n12 ,'r-',label='N12 Parent')
        plt.plot(zarr,pz_parent_n12b,'g-',label='N12B Parent')
        plt.plot(zarr,pz_parent_g16 ,'b-',label='G16 Parent')
        plt.legend(loc='upper left')
        plt.xlabel('$z$',fontsize=16)
        plt.ylabel('$N(z)$',fontsize=16)
        plt.savefig('doc/nz.pdf',bbox_inches='tight')
        plt.show()

        w_dla_n12 =np.ones(ndla_n12)
        w_dla_n12b=np.ones(ndla_n12b)
        w_dla_g16 =np.ones(ndla_g16)

        #Write DLA N12 catalog
        tbhdu=pf.new_table([pf.Column(name='ThingID',format='K',array=data_dla_n12['ThingID']),
                            pf.Column(name='RA'     ,format='D',array=data_dla_n12['RA']),
                            pf.Column(name='DEC'    ,format='D',array=data_dla_n12['DEC']),
                            pf.Column(name='zqso'   ,format='D',array=data_dla_n12['zqso']),
                            pf.Column(name='z_abs'  ,format='D',array=data_dla_n12['z_abs']),
                            pf.Column(name='NHI'    ,format='D',array=data_dla_n12['NHI']),
                            pf.Column(name='fDLA'   ,format='D',array=data_dla_n12['fDLA']),
                            pf.Column(name='B'      ,format='D',array=b_dla_n12),
                            pf.Column(name='L'      ,format='D',array=l_dla_n12),
                            pf.Column(name='W'      ,format='D',array=w_dla_n12)])
        tbhdu.writeto(cmm.fname_dla_n12,clobber=True)

        #Write DLA N12B catalog
        tbhdu=pf.new_table([pf.Column(name='ThingID',format='K',array=data_dla_n12b['ThingID']),
                            pf.Column(name='RA'     ,format='D',array=data_dla_n12b['RA']),
                            pf.Column(name='DEC'    ,format='D',array=data_dla_n12b['DEC']),
                            pf.Column(name='zqso'   ,format='D',array=data_dla_n12b['zqso']),
                            pf.Column(name='z_abs'  ,format='D',array=data_dla_n12b['z_abs']),
                            pf.Column(name='NHI'    ,format='D',array=data_dla_n12b['NHI']),
                            pf.Column(name='fDLA'   ,format='D',array=data_dla_n12b['fDLA']),
                            pf.Column(name='B'      ,format='D',array=b_dla_n12b),
                            pf.Column(name='L'      ,format='D',array=l_dla_n12b),
                            pf.Column(name='W'      ,format='D',array=w_dla_n12b)])
        tbhdu.writeto(cmm.fname_dla_n12b,clobber=True)

        #Write DLA G16 catalog
        tbhdu=pf.new_table([pf.Column(name='ThingID',format='K',array=data_dla_g16['ThingID']),
                            pf.Column(name='RA'     ,format='D',array=data_dla_g16['RA']),
                            pf.Column(name='DEC'    ,format='D',array=data_dla_g16['DEC']),
                            pf.Column(name='zqso'   ,format='D',array=data_dla_g16['zqso']),
                            pf.Column(name='z_abs'  ,format='D',array=data_dla_g16['z_abs']),
                            pf.Column(name='log_NHI',format='D',array=data_dla_g16['log_NHI']),
                            pf.Column(name='py'     ,format='D',array=data_dla_g16['py']),
                            pf.Column(name='B'      ,format='D',array=b_dla_g16),
                            pf.Column(name='L'      ,format='D',array=l_dla_g16),
                            pf.Column(name='W'      ,format='D',array=w_dla_g16)])
        tbhdu.writeto(cmm.fname_dla_g16,clobber=True)

        #Write QSO catalog
        unihi=np.zeros_like(data_qso['UNIFORM']); unihi[:]=data_qso['UNIFORM']; unihi[data_qso['Z_PIPE']<2]=0;
        tbhdu=pf.new_table([pf.Column(name='SDSS_NAME',format='18A',array=data_qso['SDSS_NAME']),
                            pf.Column(name='UNIFORM'  ,format='I'  ,array=data_qso['UNIFORM']),
                            pf.Column(name='UNIHI'    ,format='I'  ,array=unihi),
                            pf.Column(name='RA'       ,format='D'  ,array=data_qso['RA']),
                            pf.Column(name='DEC'      ,format='D' ,array=data_qso['DEC']),
                            pf.Column(name='Z_PIPE'   ,format='D' ,array=data_qso['Z_PIPE']),
                            pf.Column(name='B'        ,format='D' ,array=b_qso),
                            pf.Column(name='L'        ,format='D' ,array=l_qso),
                            pf.Column(name='W_DUM'    ,format='D' ,array=np.ones_like(w_qso_n12)),
                            pf.Column(name='W_N12'    ,format='D' ,array=w_qso_n12),
                            pf.Column(name='W_N12B'   ,format='D' ,array=w_qso_n12b),
                            pf.Column(name='W_G16'    ,format='D' ,array=w_qso_g16)])
        tbhdu.writeto(cmm.fname_qso,clobber=True)
