import healpy as hp
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.signal import savgol_filter
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.4,Om0=0.315,Ob0=0.0224/(0.674)**2,Neff=2.99)
import astropy.constants as const
from scipy.interpolate import interp1d
import camb
from camb import model, initialpower
import emcee

from IPython.display import display, Math
import corner

# 计算 cross-correlation spectrum

class corr:
    
    '''
    cmb_file: alm fits, CMB data
    cmb_mask: map fits, CMB mask
    q_data: structure, QSO data, in a certain structure so that redshift,
            ra,dec could be obtained by q_data['z'],q_data['ra'],q_data['dec']
    '''
    
    def __init__(self,cmb_file,cmb_mask,q_data,nside,mask_threshold=-5,q_mask=None,smooth_kernel=10):
        
        self.nside = nside
        
        self.cmb_alm = hp.fitsfunc.read_alm(cmb_file,hdu=1)           
        self.cmb_map = hp.sphtfunc.alm2map(self.cmb_alm,nside=self.nside)   # convert cmb_alm to a map 
        self.cmb_mask = hp.fitsfunc.read_map(cmb_mask) 
        
        self.q_z = q_data['z']
        self.q_ra = q_data['ra']
        self.q_dec = q_data['dec']
        
        q_sc = SkyCoord(ra=self.q_ra,
                        dec=self.q_dec,unit='deg',frame='icrs')
        q = q_sc.galactic          # galactic coordinate system
        
        self.q_l = q.l.degree
        self.q_b = q.b.degree
        
        self.q_map = None          # QSO map                                
        self.q_mask = q_mask       # mask for QSO map with Nside=self.nside
        
        self.kq_mask = None        # shared masked region
        self.kq_cmb = None         # shared region in cmb map 
        self.kq_q = None           # shared region in QSO map
        

        self.smooth_q_mask = None
        self.smooth_cmb_mask = None
        self.smooth_kernel = smooth_kernel

        

        
    def cal_q_mask(self):    # calculate the masked region only for QSO map with Nside=self.nside
        
        q_indice = hp.pixelfunc.ang2pix(32,self.q_l,
                                        self.q_b,lonlat=True)  # downgrade the resolution to Nside=32   
        q_map = np.zeros(hp.nside2npix(32), dtype=np.float)    
        
        for i in range(len(q_indice)):   # construct QSO map with Nside=32      
            q_map[q_indice[i]] += 1

        mask_q = np.zeros(len(q_map))   # construct QSO mask with Nside=32               
        for i in range(len(q_map)):
            if q_map[i] == 0:           # identify empty pixels
                mask_q[i] = 0
            else:
                mask_q[i] = 1      
        
           
        self.q_mask = hp.pixelfunc.ud_grade(mask_q,nside_out=self.nside)  # upgrade the mask to Nside=self.nside

    def cal_qso_overdensity_map(self):  # calculate QSO overdensity map
        
        q_indice = hp.pixelfunc.ang2pix(self.nside,self.q_l,     # construct QSO map with Nside=self.nside
                                self.q_b,lonlat=True)     
        q_map = np.zeros(hp.nside2npix(self.nside), 
                                      dtype=np.float)   
        
        for i in range(len(q_indice)):                       
            q_map[q_indice[i]] += 1

        n_tot = np.sum(self.q_mask)        # the total amount of pixels
        q_mean = len(q_indice)/n_tot                      # the average number density (n QSO per pixel)

        q_map = (q_map-q_mean)/q_mean                     # overdensity    
        self.q_map = q_map        
        
        
    def cal_kq_mask(self): # calculate the shared masked region for the two maps

        self.smooth_q_mask = hp.sphtfunc.smoothing(self.q_mask,fwhm=np.deg2rad(self.smooth_kernel/60))
        self.smooth_cmb_mask = hp.sphtfunc.smoothing(self.cmb_mask,fwhm=np.deg2rad(self.smooth_kernel/60))                                    
    
        mask = self.smooth_cmb_mask*self.smooth_q_mask
        #mask = hp.sphtfunc.smoothing(self.q_mask*self.cmb_mask,fwhm=np.deg2rad(10/60))
        #mask = self.q_mask*self.cmb_mask 
            
        self.kq_mask = mask  
        
        
        
    def corr(self,lmax=1200):         
        
        cmb_map = hp.ma(self.cmb_map)                    # construct new cmb map using the shared mask    
        cmb_map = cmb_map*self.kq_mask
        
        q_map = hp.ma(self.q_map)                        # construct new QSO map using the shared mask
        q_map = q_map*self.kq_mask
        
        f_kq = np.sum(self.kq_mask*self.q_mask*self.cmb_mask)/len(self.kq_mask)    # the fraction of the shared sky region
       
        c_kq = (1/f_kq)*hp.sphtfunc.anafast(cmb_map,          # cross-correlation
                                            q_map,nspec=lmax)        
        
        return c_kq
    
    def bin_corr(self,c_kq,l_min=30,l_max=1200,band=15):   # bin the power-spectrum
        
        '''
        c_kq: array, cross-power spectrum to be binned
        l_min: int, the beginnning l you want
        l_max: int, the maximum l you want
        band: int, the total number of data points in the final results
        '''
        
        width = int((l_max-l_min)/band)                    # number of l in each bin
        bin_Cl = np.zeros((band,1))
        bin_l = np.zeros((band,1))
        
        for i in range(band):
            ell_min = l_min+width*i
            ell_max = l_min+(i+1)*width
            ell_seq = np.arange(ell_min,ell_max,1)
            weights = (2*ell_seq+1)/np.sum(2*ell_seq+1)
            bin_l[i] = np.sum(weights*ell_seq)
            bin_Cl[i] = np.sum(weights*c_kq[ell_min:ell_max])
        
        return bin_l, bin_Cl
    
    
    def errorbar(self,c_kq,band=15,lmin=30,lmax=1200): # calculate errorbar for each data point
        
        '''
        c_kq: array, cross-power spectrum to be binned and estimate errorbars
        band: int, the total number of data points in the final results
        l_min: int, the beginnning l you want
        l_max: int, the maximum l you want
        '''
            
        width = int((lmax-lmin)/band)
        
        f_k = np.sum(self.cmb_mask)/len(self.cmb_mask)  # the sky fraction of CMB map
        f_q = np.sum(self.q_mask)/len(self.q_mask)        # the sky fraction of QSO map
        f_kq = np.sum(self.kq_mask)/len(self.kq_mask)    # the shared sky fraction of both maps
        
        rec_sig2= np.zeros((band,1))
        
        q_map = hp.ma(self.q_map)                   # auto power spectrum would be different if mask_q is not set
        #q_map.mask = np.logical_not(self.q_mask)
        q_map = q_map*self.q_mask
        
        c_qq = (1/f_q)*hp.sphtfunc.anafast(q_map,q_map,nspec=lmax) # auto correlation for QSO map
        
        c_kk = (1/f_k)*hp.sphtfunc.alm2cl(self.cmb_alm,nspec=lmax) # auto correlation for CMB map
        
        denom_l = c_kq**2+c_kk*c_qq  # the reciprocal of var
        
        for a in range(len(rec_sig2)):  # calculate var for each bin
            l_max = lmin+width*(a+1)
            l_min = lmin+width*a
            rec_sig2[a] = np.sum([(f_kq*(2*i+1)/denom_l[i]) for i in range(l_min+1,l_max)])
        
        sig2 = 1/rec_sig2  
        
        return np.sqrt(sig2)

class bias:  # theoretical cross-power spectrum and fit a best scaling parameter for bias model
    
    def __init__(self,q_z,z_min,z_max,s,data,z_reso=100,lmax=1200,redshift_reweight=False,start=1): 
        
        '''
        q_z: 1-D array, the redshift distribution of QSO samples
        z_min: float, the minimum redshift
        z_max: float, the maximum redshift
        data: 3-D array, including index l, coefficient c_kl and its errorbar
        z_reso: int, the applied resolution for redshift
        lmax: int, the maximum spectrum index l
        '''

        self.q_z = q_z                                      # the redshift distribution of QSO samples
        self.z_range = np.linspace(z_min,z_max,z_reso)      # the redshift range the samples cover
        self.dz = (self.z_range[2:]-self.z_range[:-2])/2    # infinitesimal dz
        
        self.s = s 
        
        self.data_l = data[0][start:]     # spectrum index l
        self.data_ckl = data[1][start:]   # cross-correlation coefficient C_kl
        self.data_err = data[2][start:]   # errorbar for C_kl
        
        self.outlier_l = []
        self.outlier_ckl = []
        self.outlier_err = []
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.4,
                           ombh2=0.0224, omch2=0.120,omk=0)  # cosmological model
        pars.InitPower.set_params(ns=0.965)
        
        pars.set_matter_power(redshifts=self.z_range)  
        pars.NonLinear = model.NonLinear_both
        
        self.pk_nonlin = camb.get_matter_power_interpolator(pars,nonlinear=True,           # the non-linear matter power spectrum
                                                            hubble_units=False,
                                                            k_hunit=False,kmax=20,
                                                            zmin=z_min-0.2,zmax=z_max+0.1)
        

        
        self.b_fid = lambda x : 0.278*((1+x)**2-6.565) + 2.393
        
        self.dN = None                         # the redshift distribution of QSO samples
                
        c = const.c.to('km/s').value           # light velocity in km/s
       
        h = cosmo.h                            # h corresponding to Hubble Constant 
        Om0 = cosmo.Om0
        self.z_cmb = 1100                      # the last scattering surface
        
        def H(z):                              # Hubble parameter
            return (cosmo.H(z)).value
            
        def chi(z):                            # comoving distance
            return (cosmo.comoving_distance(z).value)
        
        def W(z):                              # CMB lensing kernel 
            
            c1 = 3*(H(0)**2)*Om0/(2*c*H(z))
            c2 = (1+z)*chi(z)
            c3 = (1-chi(z)/chi(self.z_cmb))
            return c1*c2*c3
        
        self.z = self.z_range[1:-1]           # the used redshift range when intergrating
        
        self.win0 = H(self.z)/(chi(self.z)**2)*W(self.z)*(1/c) # part of the final integral term independent of bias and index l 
                          
        
        bin_num=z_reso # calculate the redshift distribution based on QSO samples
        
        
        if redshift_reweight is not False:
            dz, bins_edges = np.histogram(redshift_reweight,
                                          bins=bin_num,density=True)   
            print('redshift reweighted!')
        else:
            dz, bin_edges = np.histogram(self.q_z,
                                         bins=bin_num,density=True)
        dN = savgol_filter(dz,51,3) 
        
        self.dN = dN

        
        # mag_bia        
        maxqz = 100
        z_set = np.linspace(self.z,maxqz,10**3).T # shape = (z bins,1E5) 
        diff_z = np.diff(z_set,axis=1)  # shape = (z bins,1E5-1)
        inv_chi_zs = 1/(chi(z_set)[:,1:])      # shape = (z bins,1E5-1)
        chi_z_p = np.repeat(chi(self.z),diff_z.shape[-1]).reshape((len(self.z),diff_z.shape[-1])) # shape = (z bins,1E5-1)
        
        dNz = []
        for i in self.z:
            dN_z = np.histogram(self.q_z,range=(i,maxqz),bins=z_set.shape[1]-1)
            dN_z = dN_z[0]/(dN_z[1][2]-dN_z[1][1])/len(self.q_z)
            dNz.append(dN_z)
        dNz = np.array(dNz)
        #dNz = np.repeat(np.array([dNz]),len(self.z),axis=0).reshape((len(self.z),diff_z.shape[-1]))
        
        g0 = (dNz)*(1-chi_z_p*inv_chi_zs) # shape = (z bins,1E5-1)
        
        g = chi(self.z)/c*np.sum(g0*diff_z,axis=1) # shape = (z bins,)
        #print(np.dot(dNz[0],diff_z[0]))
        
        self.mag_f = 3/2/H(self.z)*Om0*H(0)**2*(5*self.s-2)*g
        
        
    def theoretical_curve(self,a,x=range(1200),line=True):  # calculate the theoretical curve
        
        '''
        a: float, scaling parameter
        x: array, your wanted index l range
        line: bool, to obtain a curve (True) or several selected points (False) ?
        '''
        
        b = a*self.b_fid(self.z_range)  # revised bias
        f = b*self.dN                   # window function for QSO 
        ff = f[1:-1]+self.mag_f         # to match the used redshift range
        
        w = np.ones(self.z.shape)       # weights for matter power spectrum
    

        
        
        win = (ff+self.mag_f)*self.win0              # part of the final in integral term independent of index l
        chi_z = (cosmo.comoving_distance(self.z).value)
        
        if line:                        # give a complete curve
            
            ls = np.arange(min(x),max(x)+1,0.5,dtype=np.float64) 
            cl = np.zeros(ls.shape)
            for i,l in enumerate(ls):
                k = l/chi_z
                w[:] = 1
                #w[k<1e-4] = 0
                #w[k>=20] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)  # integrate over redshift
                
            return ls,cl
        
        else:                         # give seleted data points 
            cl = np.zeros((len(x),1))
            for i,l in enumerate(x):
                k = l/chi_z
                w[:] = 1
                w[k<1e-4] = 0
                w[k>=10] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)                
            return cl
    
    def MCMC(self,func=theoretical_curve,initial=1,sep=1e-4,nwalker=10,nstep=5000,discard=500):  # use MCMC to find a best fitting
        
        '''
        initial: float, initial guess for the result and initial position for the walkers
        sep: float, average seperation between walkers
        nwalker: int, number of walkers
        nstep: int, steps for each walker
        discard: int, the burning part to be discarded in the MC chain, must smaller than nstep
        '''
        
        def log_prior(a):
            if -5.0 < a < 5:
                return 0.0
            return -np.inf

        def log_likelihood(a, x, y, yerr):
            model = func(self,a,x,False)
            sigma2 = yerr** 2 
            return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))    


        def log_probability(a, x, y, yerr):
            lp = log_prior(a)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(a, x, y, yerr)
        
        pos = initial + sep * np.random.randn(nwalker, 1)
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(self.data_l, self.data_ckl, self.data_err))
        sampler.run_mcmc(pos, nstep, progress=True)
        
        
        labels = ["a"]
        '''
        # plot MC chain
        
        fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
       
        ax.plot(samples[:, :, 0], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[0])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("step number");
        '''
        # plot the probability distribution and give a result
        
        flat_samples = sampler.get_chain(discard=discard,flat=True)
       

        fig = corner.corner(flat_samples, labels=labels)
        
        mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[0])
        display(Math(txt))
        
        return mcmc[1],q[0],q[1]
    
    def opt_minimize(self,initial=1,func=theoretical_curve):
        
        from scipy.optimize import minimize
        
        def fun(a,x,y,yerr):
            
            model = func(self,a,x,line=False)
            sigma2 = yerr ** 2
            return 0.5 * np.sum((y - model) ** 2 / sigma2 - np.log(sigma2))
    
        res = minimize(fun,initial,args=(self.data_l, self.data_ckl, self.data_err))
        
        txt = "\mathrm{{{1}}} = {0:.3f}"
        txt = txt.format(res['x'][0],'a')
        display(Math(txt))
        
        return res['x']
    
    def opt_curvefit(self,initial=1,func=theoretical_curve,cov=False):
        from scipy.optimize import curve_fit
        
        def curve(x,a):
            model = func(self,a,x,line=False)
            return model.flatten()
        
        if cov is False:
            print('no covariance matrix!')
            cov=(self.data_err).flatten()
          
       
        popt,pcov = curve_fit(curve,xdata=np.array(self.data_l),ydata=self.data_ckl.flatten(),p0=[1],
                              sigma=cov,absolute_sigma=True)
        std = np.sqrt(np.diag(pcov))
        
        txt = "\mathrm{{{2}}} = {0:.3f}\pm{1:.3f}"
        txt = txt.format(popt[0],std[0],'a')
        display(Math(txt))
        
        return popt[0],std
  
    def outlier(self,model_ckl): 
        
        list_ckl = list(self.data_ckl.flatten())
        list_l = list(self.data_l)
        list_err = list(self.data_err.flatten())
        
        #std = np.std(model_ckl-list_ckl)
        #print(std)
        
        excess = []
        for i,ckl in enumerate(list(self.data_ckl.flatten())):
            if abs(model_ckl[i]-ckl) > 2*list(self.data_err.flatten())[i]:
                print(i,abs(model_ckl[i]-ckl),list(self.data_err.flatten())[i])
                self.outlier_l.append(list(self.data_l)[i])
                self.outlier_ckl.append(ckl)
                self.outlier_err.append(list(self.data_err.flatten())[i])
                list_ckl.remove(ckl)
                list_l.remove(list(self.data_l)[i])
                list_err.remove(list(self.data_err.flatten())[i])

        #print((list_l))
        self.data_ckl = np.array(list_ckl)
        self.data_l = np.array(list_l)
        self.data_err = np.array(list_err)
    
class bias_dla:  # theoretical cross-power spectrum and fit a best scaling parameter for dla bias model
    
    def __init__(self,q_z,dla_z,s,data,n_dla=1,z_reso=120,lmax=1200,start=1): 
        
        '''
        q_z: 1-D array, the redshift distribution of QSO samples
        z_min: float, the minimum redshift
        z_max: float, the maximum redshift
        data: 3-D array, including index l, coefficient c_kl and its errorbar
        z_reso: int, the applied resolution for redshift
        lmax: int, the maximum spectrum index l
        '''

        self.q_z = q_z                                      # the redshift distribution of QSO samples
        self.dla_z = dla_z                                  # the redshift distribution of DLA samples
        
        z_min = min([min(q_z),min(dla_z)])
        z_max = max([max(q_z),max(dla_z)])
        
        #print(z_min,z_max)
        self.z_range = np.linspace(z_min,z_max,z_reso)      # the redshift range the samples cover
        self.dz = (self.z_range[2:]-self.z_range[:-2])/2    # infinitesimal dz
        
        self.s = s 
        self.n_dla = n_dla
        
        self.data_l = data[0][start:]     # spectrum index l
        self.data_ckl = data[1][start:]   # cross-correlation coefficient C_kl
        self.data_err = data[2][start:]   # errorbar for C_kl
        
        self.outlier_l = []
        self.outlier_ckl = []
        self.outlier_err = []  
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=cosmo.H0.value,
                           ombh2=(cosmo.h)**2*(cosmo.Ob0), omch2=0.122)  # cosmological model
        pars.InitPower.set_params(ns=0.965)
        
        pars.set_matter_power(redshifts=self.z_range)  
        pars.NonLinear = model.NonLinear_both
        
        self.pk_nonlin = camb.get_matter_power_interpolator(pars,nonlinear=True,           # the non-linear matter power spectrum
                                                            hubble_units=False,
                                                            k_hunit=False,kmax=20,
                                                            zmin=z_min-0.2,zmax=z_max+0.1)
        
        self.b_fid = lambda x : 0.278*((1+x)**2-6.565) + 2.393

        c = const.c.to('km/s').value           # light velocity in km/s
       
        h = cosmo.h                            # h corresponding to Hubble Constant 
        Om0 = cosmo.Om0
        self.z_cmb = 1100                      # the last scattering surface
        
        def H(z):                              # Hubble parameter
            return (cosmo.H(z)).value
            
        def chi(z):                            # comoving distance
            return (cosmo.comoving_distance(z).value)
        
        def W(z):                              # CMB lensing kernel 
            
            c1 = 3*(H(0)**2)*Om0/(2*c*H(z))
            c2 = (1+z)*chi(z)
            c3 = (1-chi(z)/chi(self.z_cmb))
            return c1*c2*c3
        
        self.z = self.z_range[1:-1]           # the used redshift range when intergrating
        
        self.win0 = H(self.z)/(chi(self.z)**2)*W(self.z)*(1/c) # part of the final integral term independent of bias and index l 
                          
        # calculate the redshift distribution 
        bin_num=z_reso
        dN_q,zq = np.histogram(self.q_z,range=(z_min,z_max),bins=bin_num)
        dN_d,zd = np.histogram(self.dla_z,range=(z_min,z_max),bins=bin_num)
        dNq_z = dN_q/(zq[2]-zq[1])
        dNd_z = dN_d/(zd[2]-zd[1])
        dNq = np.array(dNq_z)/(len(self.q_z))
        dNd = np.array(dNd_z)/(len(self.dla_z))
        #print(np.dot(dNq,np.diff(zq))+np.dot(dNd,np.diff(zd)))
        
        self.dNq = dNq #savgol_filter(dNq,21,3)
        self.dNd = dNd #savgol_filter(dNd,21,3)
        '''
        plt.plot(zq[:-1],self.dNq,label='q')
        plt.plot(zd[:-1],self.dNd,label='dla')
        plt.legend()
        plt.show()
        '''
        # mag_bia        
        maxqz = 100
        z_set = np.linspace(self.z,maxqz,10**3).T # shape = (z bins,1E5) 
        diff_z = np.diff(z_set,axis=1)  # shape = (z bins,1E5-1)
        inv_chi_zs = 1/(chi(z_set)[:,1:])      # shape = (z bins,1E5-1)
        chi_z_p = np.repeat(chi(self.z),diff_z.shape[-1]).reshape((len(self.z),diff_z.shape[-1])) # shape = (z bins,1E5-1)
        
        dNz = []
        for ind,i in enumerate(self.z):
            dN_z,zz = np.histogram(self.q_z,range=(i,maxqz),bins=z_set.shape[1]-1)
            dN_z = dN_z/(zz[2]-zz[1])/len(self.q_z)
            dNz.append(dN_z)
        dNz = np.array(dNz)
       
        g0 = (dNz)*(1-chi_z_p*inv_chi_zs) # shape = (z bins,1E5-1)
        
        g = chi(self.z)/c*np.sum(g0*diff_z,axis=1) # shape = (z bins,)
        #print(np.dot(dNz[0],diff_z[0]))
        self.mag_f = 3/2/H(self.z)*Om0*H(0)**2*(5*self.s-2)*g
        
        
    def theoretical_curve(self,aq,b_dla,x=range(1200),line=True):  # calculate the theoretical curve
        
        '''
        a: float, scaling parameter
        x: array, your wanted index l range
        line: bool, to obtain a curve (True) or several selected points (False) ?
        '''
        
        bq = aq*self.b_fid(self.z_range)  # revised bias for QSO
        bd = b_dla                        # revised bias for DLA
        f = bq*self.dNq + self.n_dla * bd*self.dNd     # window function for QSO 
        ff = f[1:-1] + self.mag_f         # to match the used redshift range
        
        w = np.ones(self.z.shape)       # weights for matter power spectrum
    

        
        
        win = (ff+self.mag_f)*self.win0              # part of the final in integral term independent of index l
        chi_z = (cosmo.comoving_distance(self.z).value)
        
        if line:                        # give a complete curve
            
            ls = np.arange(min(x),max(x)+1,0.5,dtype=np.float64) 
            cl = np.zeros(ls.shape)
            for i,l in enumerate(ls):
                k = l/chi_z
                w[:] = 1
                w[k<1e-4] = 0
                w[k>=20] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)  # integrate over redshift
                
            return ls,cl
        
        else:                         # give seleted data points 
            cl = np.zeros((len(x),1))
            for i,l in enumerate(x):
                k = l/chi_z
                w[:] = 1
                w[k<1e-4] = 0
                w[k>=10] = 0
                cl[i] = np.dot(self.dz,w*self.pk_nonlin.P(self.z,k,grid=False)*win)                
            return cl
    
    def MCMC(self,func=theoretical_curve,distri_aq=[1,1],initial=1,sep=1e-2,nwalker=10,nstep=5000,discard=500,fig_title='MCMC',cov=False):  # use MCMC to find a best fitting
        
        '''
        initial: float, initial guess for the result and initial position for the walkers
        sep: float, average seperation between walkers
        nwalker: int, number of walkers
        nstep: int, steps for each walker
        discard: int, the burning part to be discarded in the MC chain, must smaller than nstep
        '''
        
        def log_prior(a):
            aq,ad = a
            mu_aq = distri_aq[0]
            s = distri_aq[1]
            
            ln_p1 =-0.5*np.sum((aq-mu_aq)**2/s**2+np.log(2*np.pi*s**2)) 
            ln_p2 = -np.inf
            if 0. < ad < 15 :
                ln_p2 = 0.0
            
            return ln_p1+ln_p2

        def log_likelihood(a, x, y, yerr):
            aq,ad = a
            model = func(self,aq,ad,x,False).flatten()
            sigma2 = yerr ** 2 
            if cov is False:
               
                return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))  
            else:
                return -0.5 * np.sum( np.dot((y-model).T, np.linalg.solve(cov, (y-model))) + np.linalg.det(cov) )
                #return -0.5 * np.dot(np.dot((y-model).T,np.linalg.inv(cov)),(y-model))[0][0] + np.linalg.det(cov)


        def log_probability(a, x, y, yerr):
            lp = log_prior(a)
            if not np.isfinite(lp):
                return -np.inf
            return (lp + log_likelihood(a, x, y, yerr))   
        
        pos = initial + sep * np.random.randn(nwalker, 2)
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(self.data_l.flatten(), self.data_ckl.flatten(), self.data_err.flatten()))
        sampler.run_mcmc(pos, nstep, progress=True)
        
        
        labels = [r"$a_{QSO}$",r"$b_{DLA}$"]
        
        # plot MC chain
        '''
        fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        
        ax.plot(samples[:, :, 1], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[1])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("step number");
        '''
        # plot the probability distribution and give a result
        
        flat_samples = sampler.get_chain(discard=discard,flat=True)
        

        
        mcmc_q = np.percentile(flat_samples[:, 0], [16, 50, 84])
        q = np.diff(mcmc_q)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc_q[1], q[0], q[1], labels[0].split('$')[1])
        display(Math(txt))
        
        mcmc_d95 = np.percentile(flat_samples[:, 1], [90])
        txt = "\mathrm{{{1}}} = {0:.3f}"
        txt = txt.format(mcmc_d95[0], labels[1].split('$')[1]+"(90\%)")
        display(Math(txt))
        
        
        mcmc_d = np.percentile(flat_samples[:, 1], [16,50,84])
        d = np.diff(mcmc_d)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc_d[1], d[0], d[1], labels[1].split('$')[1])
        display(Math(txt))
        

        
        fig = corner.corner(flat_samples,bins=50,color='k',labels=labels,truths=[mcmc_q[1],mcmc_d[1]],quantiles=[0.16,0.84],plot_contours=True)
        
        for ax in fig.get_axes():
            xl = ax.get_xlabel()
            ax.set_xlabel(xl, fontsize=20)
            yl = ax.get_ylabel()
            ax.set_ylabel(yl, fontsize=20)
            
        plt.savefig('%s.pdf'%fig_title)
        
        fig = corner.corner(flat_samples,bins=50,color='k',labels=labels,quantiles=[0.95],plot_contours=True)
        
        for ax in fig.get_axes():
            xl = ax.get_xlabel()
            ax.set_xlabel(xl, fontsize=20)
            yl = ax.get_ylabel()
            ax.set_ylabel(yl, fontsize=20)
            
        plt.savefig('%s_95.pdf'%fig_title)
        
        return mcmc_q[1],mcmc_d[1],d[0],d[1],mcmc_d95[0]
    
    def post_prob(self,a,x,y,yerr,func=theoretical_curve,distri_aq=[1,1],cov=False):  
        
        '''
        initial: float, initial guess for the result and initial position for the walkers
        sep: float, average seperation between walkers
        nwalker: int, number of walkers
        nstep: int, steps for each walker
        discard: int, the burning part to be discarded in the MC chain, must smaller than nstep
        '''
        
        def log_prior(a):
            aq,ad = a
            mu_aq = distri_aq[0]
            s = distri_aq[1]
            
            ln_p1 =-0.5*np.sum((aq-mu_aq)**2/s**2+np.log(2*np.pi*s**2)) 
            
            ln_p2 = -np.inf
            if 0. < ad < 15:
                ln_p2 = 0.0
            
            return ln_p1+ln_p2

        def log_likelihood(a, x, y, yerr):
            aq,ad = a
            model = func(self,aq,ad,x,False)
            sigma2 = yerr ** 2 
            if cov is False:
                return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))   
            else:
                return -0.5 * np.dot(np.dot((y-model).T,np.linalg.inv(cov)),(y-model))[0][0]


        def log_probability(a, x, y, yerr):
            lp = log_prior(a)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(a, x, y, yerr)    
        
        
        return log_probability(a,x,y,yerr)
    
    
    def opt_curvefit(self,initial=1,func=theoretical_curve,cov=False):
        from scipy.optimize import curve_fit
        
        def curve(x,aq,ad):
            model = func(self,aq,ad,x,line=False)
            return model.flatten()
        
        if cov is False:
            cov=(self.data_err).flatten()
            
        popt,pcov = curve_fit(curve,xdata=np.array(self.data_l),ydata=self.data_ckl.flatten(),p0=[1,1],bounds=([0.4,0.],[1.5,10.]),
                              sigma=cov,absolute_sigma=True)
        std = np.sqrt(np.diag(pcov))
        
        txt = "\mathrm{{{2}}} = {0:.3f}\pm{1:.3f}, \mathrm{{{5}}} = {3:.3f}\pm{4:.3f}"
        txt = txt.format(popt[0],std[0],'a_{QSO}',popt[1],std[1],'a_{DLA}')
        display(Math(txt))
        
        return popt,std
  
    def outlier(self,model_ckl): 
        
        list_ckl = list(self.data_ckl.flatten())
        list_l = list(self.data_l)
        list_err = list(self.data_err.flatten())
        
        #std = np.std(model_ckl-list_ckl)
        #print(std)
        
        excess = []
        for i,ckl in enumerate(list(self.data_ckl.flatten())):
            if abs(model_ckl[i]-ckl) > 2*list(self.data_err.flatten())[i]:
                print(i,abs(model_ckl[i]-ckl),list(self.data_err.flatten())[i])
                self.outlier_l.append(list(self.data_l)[i])
                self.outlier_ckl.append(ckl)
                self.outlier_err.append(list(self.data_err.flatten())[i])
                list_ckl.remove(ckl)
                list_l.remove(list(self.data_l)[i])
                list_err.remove(list(self.data_err.flatten())[i])

        #print((list_l))
        self.data_ckl = np.array(list_ckl).reshape((len(list_ckl),1))
        self.data_l = np.array(list_l).reshape((len(list_ckl),1))
        self.data_err = np.array(list_err).reshape((len(list_ckl),1))
        
        
from colossus.cosmology import cosmology
cosmology.setCosmology('planck15')
from colossus.lss import bias as bias_col
from colossus.lss.peaks import massFromPeakHeight
import emcee

class halomass:
    
    def __init__(self,b,z,delta_b=[1,1]):
            self.b = b
            self.delta_b_low = delta_b[0]
            self.delta_b_high = delta_b[1]
            self.z = z
            self.nu = None
            self.nu_upper = None
            self.nu_lower = None
            self.halomass = None
            self.halomass_upper = None
            self.halomass_lower = None
    def MCMC_nu(self,initial=1,sep=1e-4,nwalker=10,nstep=500,discard=40):

        def log_prior(nu):
            if 0 < nu < 5:
                return 0.0
            return -np.inf

        def log_likelihood(nu):

            bias_model = bias_col.modelTinker10(nu,self.z,mdef='200m')
            sigma2 = self.delta_b**2 
            return -0.5 * np.sum((self.b - bias_model) ** 2 / sigma2 + np.log(sigma2))

        def log_probability(nu):
            lp = log_prior(nu)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(nu)

        pos = 1 + 1e-4 * np.random.randn(nwalker, 1)
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos, nstep, progress=True)
        
        labels = ["$\\nu$"]
        
        # plot MCMC chain    
        fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        
        ax.plot(samples[:, :, 0], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[0])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("step number");
        
        # plot the probability distribution 

        flat_samples = sampler.get_chain(discard=200,flat=True)
        print(flat_samples.shape)

        fig = corner.corner(flat_samples, labels=labels)

        # give a result

        mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], '\\nu')
        display(Math(txt))
        
        self.nu = mcmc[1]
        self.nu_upper = mcmc[0]
        self.nu_lower = mcmc[2]
        
    def fsolve_nu(self):
        '''
        Delta = 200
        y = np.log10(Delta)
        A = 1+0.24*y*np.exp(-(4/y)**4)
        a = 0.44*y-0.88
        B = 0.183
        b = 1.5
        C = 0.019+0.107*y+0.19*np.exp(-(4/y)**4)
        c = 2.4
        '''
        def bias(nu,delta=0):
            '''
            delta_c = 1.686
            par1 = -A*(nu**a)/(delta_c**a+nu**a)
            par2 = B*nu**b
            par3 = C*nu**c
            
            return 1+par1+par2+par3 
            '''
            return bias_col.modelTinker10(nu,self.z,mdef='200m')-(self.b+delta)
        from scipy.optimize import fsolve

        result1 = fsolve(bias,0.5,args=[0])
        result2 = fsolve(bias,0.5,args=[-self.delta_b_low])
        result3 = fsolve(bias,0.5,args=[self.delta_b_high])
        
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
        txt = txt.format(self.b,self.delta_b_low,self.delta_b_high,'b')
        display(Math(txt))
        
        txt = "\mathrm{{{3}}} = {0:.3f}_{{{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(result1[0],result2[0]-result1[0],result3[0]-result1[0],'\\nu')
        display(Math(txt))
        
        self.nu = result1[0]
        self.nu_lower = result2[0]
        self.nu_upper = result3[0]
        

    def nu2mass(self):

        self.halomass = massFromPeakHeight(self.nu,self.z)
        self.halomass_upper = massFromPeakHeight(self.nu_upper,self.z)
        self.halomass_lower = massFromPeakHeight(self.nu_lower,self.z)

        txt = "\mathrm{{{3}}} = {0:.3f}_{{{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(np.log10(self.halomass), np.log10(self.halomass_upper)-np.log10(self.halomass), 
                         np.log10(self.halomass_lower)-np.log10(self.halomass), 'log_{10}(M/M_\odot h ^{-1})')
        display(Math(txt))

        
        
    
