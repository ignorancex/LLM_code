# A naive fitting template based on LSE/MCMC written by Yingjie Zhu@CLaSP, Umich
# Generally based on tutorials on emcee & george  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import ticker
from matplotlib import rcParams
from matplotlib import patches
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
from IPython.display import display, Math
from astropy.modeling import models, fitting, Fittable1DModel, Parameter
import emcee
from scipy.special import wofz
#import george
#from george import kernels
#from multiprocessing import Pool

class FeXII_fit:
    def __init__(self,data,wvl,p0,fit_type = "Single",err = None,stray_int = 0, stray_fwhm=1,stray_wvl=0):
        self.data = data
        self.wvl = wvl
        self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],51)
        self.shape = data.shape
        self.fit_type = fit_type
        self.result = np.tile(p0,(self.shape[0],1))
        self.result_err = np.zeros_like(self.result)
        self.mcmc_result = np.tile(np.append(p0,-2),(self.shape[0],1))
        self.mcmc_result_err = np.zeros((self.shape[0],2,5))
        self.error = err
        self.stray_int = stray_int
        self.stray_fwhm = stray_fwhm
        self.stray_wvl = stray_wvl
    
    def plot(self,plot_fit = True,plot_mcmc = False):
        if self.shape[0] == 32:
            fig, axes = plt.subplots(8,4,figsize=(16,16))
        elif self.shape[0] == 16:
            fig, axes = plt.subplots(4,4,figsize=(16,8))
        else:
            nrows = np.floor_divide(self.shape[0],4) + 1
            fig, axes = plt.subplots(nrows,4,figsize=(16,nrows*3))
        
        
        for ii, ax_ in enumerate(axes.flatten()):
            if ii < self.shape[0]:
                if self.error is None:
                    ln1, = ax_.step(self.wvl,self.data[ii,:],where="mid",color="#E87A90",label = r"$I_{\rm obs}$",lw=2)
                else:
                    ln1 = ax_.errorbar(self.wvl,self.data[ii,:],yerr = self.error[ii,:],ds='steps-mid',color="#E87A90",capsize=2,
                    label = r"$I_{\rm obs}$",lw=1.5)
                if plot_fit:
                    if self.fit_type == "Single":
                        if plot_mcmc:
                            g_fit = myGaussian1D(intensity=self.mcmc_result[ii,0], mean=self.mcmc_result[ii,1], 
                            fwhm = self.mcmc_result[ii,2],bg = self.mcmc_result[ii,3],stray_int = self.stray_int,
                            stray_fwhm = self.stray_fwhm,stray_wvl=self.stray_wvl)

                            if self.stray_int != 0:
                                g_true = myGaussian1D(intensity=self.mcmc_result[ii,0], mean=self.mcmc_result[ii,1], 
                                            fwhm = self.mcmc_result[ii,2],bg = self.mcmc_result[ii,3],stray_int = 0, stray_fwhm = 1,
                                            stray_wvl=self.stray_wvl)
                                if self.stray_wvl == 0:
                                    g_stray = myGaussian1D(intensity=self.stray_int, mean=self.mcmc_result[ii,1], 
                                            fwhm = self.stray_fwhm,bg = 0,stray_int = 0, stray_fwhm = 1,stray_wvl=self.stray_wvl)
                                else:
                                    g_stray = myGaussian1D(intensity=self.stray_int, mean=self.stray_wvl, 
                                            fwhm = self.stray_fwhm,bg = 0,stray_int = 0, stray_fwhm = 1,stray_wvl=self.stray_wvl)



                        else:
                            g_fit = myGaussian1D(intensity=self.result[ii,0], mean=self.result[ii,1], 
                            fwhm = self.result[ii,2],bg = self.result[ii,3],stray_int = self.stray_int,
                            stray_fwhm = self.stray_fwhm,stray_wvl=self.stray_wvl)     
                            if self.stray_int != 0:
                                g_true = myGaussian1D(intensity=self.result[ii,0], mean=self.result[ii,1], 
                                        fwhm = self.result[ii,2],bg = self.result[ii,3],stray_int = 0, 
                                        stray_fwhm = 1,stray_wvl=self.stray_wvl)
                                if self.stray_wvl == 0:
                                    g_stray = myGaussian1D(intensity=self.stray_int, mean=self.result[ii,1], 
                                        fwhm = self.stray_fwhm,bg = 0,stray_int = 0, stray_fwhm = 1,stray_wvl=self.stray_wvl)  
                                else:
                                    g_stray = myGaussian1D(intensity=self.stray_int, mean=self.stray_wvl, 
                                        fwhm = self.stray_fwhm,bg = 0,stray_int = 0, stray_fwhm = 1,stray_wvl=self.stray_wvl)          
                        ln2, = ax_.plot(self.wvl_plot,g_fit(self.wvl_plot),color="#FC9F40",ls="-",label = r"$I_{\rm fit}$",lw=1.5)
                        if self.stray_int != 0:
                            ln3, = ax_.plot(self.wvl_plot,g_true(self.wvl_plot),color="#58B2DC",ls="-",label = r"$I_{\rm pure}$",lw=1.5)
                            ln4, = ax_.plot(self.wvl_plot,g_stray(self.wvl_plot),color="#86C166",ls="-",label = r"$I_{\rm stray}$",lw=1.5)
                    if self.fit_type == "Double":
                        pass
                        '''g_fit = myGaussian1D(amplitude=self.result[ii,0], mean=self.result[ii,1], 
                        stddev = self.result[ii,2],bg = self.result[ii,6],)
                        ax_.plot(self.wvl,g_fit(self.wvl),color="#FFCB05",ls="-.")
                        g_fit = myGaussian1D(amplitude=self.result[ii,3], mean=self.result[ii,4], 
                        stddev = self.result[ii,5],bg = self.result[ii,6],)
                        ax_.plot(self.wvl,g_fit(self.wvl),color="#FFCB05",ls="-.")'''
        
        if self.stray_int != 0:
            leg = [ln1, ln2, ln3, ln4]
            axes[0,0].legend(leg,[leg_.get_label() for leg_ in leg],bbox_to_anchor=(-0.03,1.1,1,0.2), loc="upper left",ncol=4,fontsize=14)
        else:
            leg = [ln1, ln2]
            axes[0,0].legend(leg,[leg_.get_label() for leg_ in leg],bbox_to_anchor=(-0.03,1.1,1,0.2), loc="upper left",ncol=2,fontsize=14)
        

            
    def run_lse(self,ignore_err=False):
        if self.fit_type == "Single":
            for ii in range(self.shape[0]):
                if True:
                    g_init = myGaussian1D(intensity=np.max(self.data[ii,:])*np.sqrt(2*np.pi)*self.result[ii,2] - self.stray_int, mean=self.wvl[np.argmax(self.data[ii,:])],
                    fwhm = self.result[ii,2],bg = self.result[ii,3],stray_int = self.stray_int,stray_fwhm = self.stray_fwhm,stray_wvl = self.stray_wvl)
                else:
                    g_init = myGaussian1D(intensity=self.result[ii-1,0], mean=self.result[ii-1,1],
                    fwhm = self.result[ii-1,2],bg = self.result[ii-1,3],stray_int = self.stray_int,stray_fwhm = self.stray_fwhm,stray_wvl = self.stray_wvl)
                g_init.stray_int.fixed = True
                g_init.stray_fwhm.fixed = True
                g_init.stray_wvl.fixed = True
                fit_g = fitting.LevMarLSQFitter()
                if (self.error is None) or ignore_err:
                    g = fit_g(g_init, self.wvl, self.data[ii,:])
                else:
                    g = fit_g(g_init, self.wvl, self.data[ii,:])
                    #g_fit_diff = np.abs(g(self.wvl) - self.data[ii,:]) 
                    g = fit_g(g, self.wvl, self.data[ii,:],weights=1/self.error[ii,:])
                #print(g)

                if fit_g.fit_info['param_cov'] is None or np.abs(g.fwhm.value)>0.25:
                    print(ii,"fit again")
                    g_init = myGaussian1D(intensity=np.max(self.data[ii,:])*np.sqrt(2*np.pi)*self.result[ii,2] - self.stray_int, mean=self.wvl[np.argmax(self.data[ii,:])],
                    fwhm = self.result[ii,2],bg = self.result[ii-1,3],stray_int = self.stray_int,stray_fwhm = self.stray_fwhm,stray_wvl = self.stray_wvl)
                    g_init.stray_int.fixed = True
                    g_init.stray_fwhm.fixed = True
                    g_init.stray_wvl.fixed = True
                    if (self.error is None) or ignore_err:
                        g = fit_g(g_init, self.wvl, self.data[ii,:])
                    else:
                        g = fit_g(g_init, self.wvl, self.data[ii,:],weights=1/np.square(self.error[ii,:]))
                
                self.result[ii,:] = np.array([g.intensity.value,g.mean.value, np.abs(g.fwhm.value), g.bg.value])

                
                if fit_g.fit_info['param_cov'] is None:
                    nan_array = np.empty(4)
                    nan_array[:] = np.nan
                    self.result_err[ii,:] = nan_array
                    #self.result[ii,:] = nan_array
                else:
                    self.result_err[ii,:] = np.sqrt(np.diag(fit_g.fit_info['param_cov']))

        if self.fit_type == "Double":
            pass
            '''for ii in range(self.shape[0]):
                g_init = double_Gaussian1D(amplitude_1=self.result[ii,0], mean_1=self.result[ii,1], stddev_1 = self.result[ii,2],
                amplitude_2=self.result[ii,3], mean_2=self.result[ii,4], stddev_2 = self.result[ii,5], bg = 0.)
                fit_g = fitting.LevMarLSQFitter()
                if self.error is None:
                    g = fit_g(g_init, self.wvl, self.data[ii,:],weights=1/(np.sqrt(self.data[ii,:]*600.0)/600.0))
                else:
                    g = fit_g(g_init, self.wvl, self.data[ii,:],weights=1/np.square(self.error[ii,:]) )
                self.result[ii,:] = np.array([g.amplitude_1.value,g.mean_1.value, np.abs(g.stddev_1.value), 
                g.amplitude_2.value,g.mean_2.value, np.abs(g.stddev_2.value),g.bg.value])
                if fit_g.fit_info['param_cov'] is None:
                    nan_array = np.empty(7)
                    nan_array[:] = np.nan
                    self.result_err[ii,:] = nan_array
                    self.result[ii,:] = nan_array
                else:
                    self.result_err[ii,:] = np.sqrt(np.diag(fit_g.fit_info['param_cov']))'''

            
    def run_mcmc(self,burnstep = 500,nstep = 2000, stepsize = 1.e-4,progress=False,lse=True,factor_f=False,credit=68):
        if factor_f:
            if lse:
                self.run_lse()
            for ii in range(self.shape[0]):
            #for ii in range(2):
                if lse:
                    p0 = np.append(self.result[ii,:],np.log(0.2))
                else:
                    p0 = np.array([np.max(self.data[ii,:]) * np.sqrt(2 * np.pi) * self.result[ii,2],
                     self.wvl[np.argmax(self.data[ii,:])], self.result[ii,2], 0.1, np.log(0.2)])
                ndim = len(p0)
                nwalkers = 32
                p_start = [p0 + np.array([1e-4*np.random.randn()*p0[0],1e-4*np.random.randn(),
                1e-4*np.random.randn(),1e-4*p0[3]*np.random.randn(),0.005*np.random.randn()])
                    for i in range(nwalkers)]

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob2, 
                                                args=(p0,self.wvl,self.data[ii,:],self.error))

                #print("Running burn-in...")
                p_start, _, _ = sampler.run_mcmc(p_start, burnstep,progress=progress)
                sampler.reset()

                #print("Running production...")
                sampler.run_mcmc(p_start, nstep,progress=progress)
            
                flat_samples = sampler.get_chain(discard=100, thin=3, flat=True)
                for jj in range(ndim ):
                    mcmc = np.percentile(flat_samples[:, jj], [50-credit/2, 50, 50+credit/2])
                    q = np.diff(mcmc)
                    #print(type(np.array(mcmc[1])))
                    self.mcmc_result[ii,jj] = mcmc[1]
                    self.mcmc_result_err[ii,:,jj] = np.array([q[0],q[1]])
        else:
            if lse:
                self.run_lse()
            for ii in range(self.shape[0]):
                print("\r","node {:d}".format(ii),end="")
            #for ii in range(2):
                if lse:
                    p0 = self.result[ii,:]
                else:
                    p0 = np.array([np.max(self.data[ii,:])*np.sqrt(2 * np.pi)*self.result[ii,2], self.wvl[np.argmax(self.data[ii,:])],
                     self.result[ii,2], 0.1])
                ndim = len(p0)
                nwalkers = 32
                
                p_start = [p0 + np.array([0.2*np.random.randn()*p0[0],3e-2*np.random.randn(),1e-2*np.random.randn(),0.05*p0[3]*np.random.randn()])
                    for i in range(nwalkers)]

                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob1, 
                                                args=(p0,self.wvl,self.data[ii,:],self.error))


                #print("Running burn-in...")
                p_start, _, _ = sampler.run_mcmc(p_start, burnstep,progress=progress)
                sampler.reset()

                #print("Running production...")
                sampler.run_mcmc(p_start, nstep,progress=progress)
            
                flat_samples = sampler.get_chain(discard=100, thin=3, flat=True)
                for jj in range(ndim):
                    mcmc = np.percentile(flat_samples[:, jj], [50-credit/2, 50, 50+credit/2])
                    q = np.diff(mcmc)
                    #print(type(np.array(mcmc[1])))
                    self.mcmc_result[ii,jj] = mcmc[1]
                    self.mcmc_result_err[ii,:,jj] = np.array([q[0],q[1]])
                
    def get_para(self,mcmc=False):
        if mcmc:
            return self.mcmc_result
        else:
            return self.result
    
    def get_error(self,mcmc=False):
        error_return = []
        if mcmc:
            return self.mcmc_result_err
        else:
            return self.result_err
    
    def show(self,mcmc=False):
        if mcmc:
            for ii, p_ in enumerate(self.mcmc_result):
                display(Math("\lambda = {:.3f}_{{{:.3f}}}^{{{:.3f}}}\  \mathrm{{Int}} = {:.2e}_{{{:.1e}}}^{{{:.1e}}} \  \mathrm{{FHWM}} = {:.2e}_{{{:.1e}}}^{{{:.1e}}}".format(p_[1],
                self.mcmc_result_err[ii,0,1],self.mcmc_result_err[ii,1,1],p_[0],self.mcmc_result_err[ii,0,0],self.mcmc_result_err[ii,1,0],
                p_[2],self.mcmc_result_err[ii,0,2],self.mcmc_result_err[ii,1,2])))
        else: 
            for ii, p_ in enumerate(self.result):
                display(Math("\lambda = {:.3f}\pm {:.3f}\  \mathrm{{Int}} = {:.2e} \pm {:.2e} \  \mathrm{{FHWM}} = {:.1e} \pm {:.1e}".format(self.result[ii,1],
                self.result_err[ii,1],self.result[ii,0],self.result_err[ii,0],self.result[ii,2],self.result_err[ii,2])))
 
    def lnlike1(self,p,x,y,yerr):
        intensity, mean, fwhm, bg = p
        g_model = myGaussian1D(intensity=intensity, mean=mean, fwhm = fwhm,bg = bg,
         stray_int = self.stray_int, stray_fwhm = self.stray_fwhm,stray_wvl = self.stray_wvl)
        return -0.5 * np.sum(((y - g_model(x))/yerr) ** 2 + 2*np.log(yerr))
    
    def lnprior1(self,p,p0):
        intensity, mean, fwhm, bg = p
        intensity, mean_0, fwhm_0, bg_0 = p0
        #if np.abs((bg-bg_0)/bg_0)>2:
        #    return -np.inf
        #if np.abs(mean - mean_0)>0.1:
        #    return -np.inf
        #if np.abs((stddev-stddev_0)/stddev_0)>0.5:
        #    return -np.inf
        #if np.abs((amp-amp_0)/amp_0)>0.5:
        #    return -np.inf
        if fwhm > 0.2:
            return -np.inf
        return 0.0

    def lnprob1(self,p,p0, x, y, yerr):
        lp = FeXII_fit.lnprior1(self,p=p,p0=p0)
        if yerr is None:
            yerr = np.sqrt(y*600.0)/600.0
        return lp + FeXII_fit.lnlike1(self,p, x, y, yerr) if np.isfinite(lp) else -np.inf     

    def lnlike2(self,p,x,y,yerr):
        intensity, mean, fwhm, bg, log_f = p
        g_model = myGaussian1D(intensity=intensity, mean=mean, fwhm = fwhm,bg = bg,
         stray_int = self.stray_int, stray_fwhm = self.stray_fwhm,stray_wvl = self.stray_wvl)
        model = g_model(x)
        sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))   

    def lnprior2(self,p,p0):
        intensity, mean, fwhm, bg, log_f = p
        intensity_0, mean_0, fwhm_0, bg_0, log_f0 = p0
        #if np.abs((bg-bg_0)/bg_0)>2:
        #    return -np.inf
        #if np.abs(mean - mean_0)>0.1:
        #    return -np.inf
        #if np.abs((stddev-stddev_0)/stddev_0)>0.5:
        #    return -np.inf
        #if np.abs((amp-amp_0)/amp_0)>0.5:
        #    return -np.inf
        if fwhm > 0.2:
            return -np.inf
        if log_f > -1.6:
            return -np.inf
        return 0.0
    
    def lnprob2(self,p,p0, x, y, yerr):
        lp = FeXII_fit.lnprior2(self,p,p0)
        if yerr is None:
            yerr = np.sqrt(y*600.0)/600.0
        return lp + FeXII_fit.lnlike2(self,p, x, y, yerr) if np.isfinite(lp) else -np.inf    

#A Gaussian fitting model
class myGaussian1D(Fittable1DModel):
    intensity = Parameter()
    mean = Parameter()
    fwhm = Parameter()
    bg = Parameter()
    stray_int = Parameter()
    stray_fwhm = Parameter()
    stray_wvl = Parameter()

    @staticmethod
    def evaluate(x, intensity, mean, fwhm,bg,stray_int,stray_fwhm,stray_wvl):
        if stray_wvl == 0:
            return 2.355*intensity/np.sqrt(2 * np.pi)/fwhm * np.exp((-(1 / (2. * (fwhm/2.355)**2)) * (x - mean)**2)) + bg +\
            2.355*stray_int/np.sqrt(2 * np.pi)/stray_fwhm * np.exp((-(1 / (2. * (stray_fwhm/2.355)**2)) * (x - mean)**2))
        else:
            return 2.355*intensity/np.sqrt(2 * np.pi)/fwhm * np.exp((-(1 / (2. * (fwhm/2.355)**2)) * (x - mean)**2)) + bg +\
            2.355*stray_int/np.sqrt(2 * np.pi)/stray_fwhm * np.exp((-(1 / (2. * (stray_fwhm/2.355)**2)) * (x - stray_wvl)**2))

class myVoigt1D(Fittable1DModel):
    intensity = Parameter()
    mean = Parameter()
    g_fwhm = Parameter()
    l_fwhm = Parameter()
    bg = Parameter()
    stray_int = Parameter()
    stray_fwhm = Parameter()
    stray_wvl = Parameter()

    @staticmethod
    def evaluate(x, intensity, mean, g_fwhm,l_fwhm,bg,stray_int,stray_fwhm,stray_wvl):
        z = ((x-mean) + 1j*l_fwhm/2) / (g_fwhm/2 * np.sqrt(2))
        V = np.real(wofz(z))
        I_voigt = V / (g_fwhm/2*np.sqrt(2*np.pi))*intensity + bg
        if stray_wvl == 0:
            return I_voigt + bg +\
            2.355*stray_int/np.sqrt(2 * np.pi)/stray_fwhm * np.exp((-(1 / (2. * (stray_fwhm/2.355)**2)) * (x - mean)**2))
        else:
            return I_voigt + bg +\
            2.355*stray_int/np.sqrt(2 * np.pi)/stray_fwhm * np.exp((-(1 / (2. * (stray_fwhm/2.355)**2)) * (x - stray_wvl)**2))


class double_Gaussian1D(Fittable1DModel):
    amplitude_1 = Parameter()
    mean_1 = Parameter()
    stddev_1 = Parameter()
    amplitude_2 = Parameter()
    mean_2 = Parameter()
    stddev_2 = Parameter()
    bg = Parameter()

    @staticmethod
    def evaluate(x, amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2,bg):
        return amplitude_1 * np.exp((-(1 / (2. * stddev_1**2)) * (x - mean_1)**2)) +\
            amplitude_2 * np.exp((-(1 / (2. * stddev_2**2)) * (x - mean_2)**2)) + bg

    @staticmethod
    def fit_deriv(x, amplitude_1, mean_1, stddev_1, amplitude_2, mean_2, stddev_2, bg):
        #stddev_2 = stddev_1
        d_amplitude_1 = np.exp((-(1 / (stddev_1**2)) * (x - mean_1)**2))
        d_mean_1 = (2 * amplitude_1 *
                  np.exp((-(1 / (stddev_1**2)) * (x - mean_1)**2)) *
                  (x - mean_1) / (stddev_1**2))
        d_stddev_1 = (2 * amplitude_1 *
                    np.exp((-(1 / (stddev_1**2)) * (x - mean_1)**2)) *
                    ((x - mean_1)**2) / (stddev_1**3))
        d_amplitude_2 = np.exp((-(1 / (stddev_2**2)) * (x - mean_2)**2))
        d_mean_2 = (2 * amplitude_2 *
                  np.exp((-(1 / (stddev_2**2)) * (x - mean_2)**2)) *
                  (x - mean_2) / (stddev_2**2))
        d_stddev_2 = (2 * amplitude_2 *
                    np.exp((-(1 / (stddev_2**2)) * (x - mean_2)**2)) *
                    ((x - mean_2)**2) / (stddev_2**3))
        return np.array([d_amplitude_1, d_mean_1, d_stddev_1,
        d_amplitude_2, d_mean_2, d_stddev_2,np.ones_like(d_amplitude_1)])