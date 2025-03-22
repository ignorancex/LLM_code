# -*- coding: utf-8 -*-
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   class Decomp of BISP-1
#
# --- 
#
#   @author: V.Pelgrims
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from astropy.io import ascii
from astropy.table import Table

from dynesty import utils as dyfunc
import dynesty
import corner

#import bisp1_logll as logL

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Decomp:
    '''
    Object Decomp.
    Wraps the dynesty sampler and allows for specific access and manipulations
    '''
    def __init__(self,dySampler):
        '''
            Initialization is simply the wrap.
        '''
        self.sampler = dySampler
    #

    def PosteriorSamples(self,samplerSize=10000):
        '''
            Reads dyresults and return weighted posterior samples
            Optional argument: 'samplerSize' specifies the maximum number of
                        posterior samples to return.
        '''
        dyresults = self.sampler.results
        samples = dyresults.samples
        weights = np.exp(dyresults.logwt-dyresults.logl[-1])
        weights /= np.sum(weights)
        posteriors = dyfunc.resample_equal(samples,weights)
        post_length = len(posteriors[:,0])
        ind_select = np.random.choice(np.arange(post_length),\
                                np.minimum(samplerSize,post_length),\
                                    replace=False)
        return posteriors[ind_select,:]

    def SummaryStat(self):
        '''
            Reads dyresults and returns evidence, error on evidence,
            and maximum log-likelihood
        '''
        dyresults = self.sampler.results
        Z_evidence = dyresults.logz[-1]
        err_Z_evidence = dyresults.logzerr[-1]
        max_logl = dyresults.logl[-1]
        return Z_evidence,err_Z_evidence,max_logl
    
    def Values_at_MaxLikelihood(self):
        '''
            Extracts parameter values at maximum log-likelihood
        '''
        dyresults = self.sampler.results
        bf_values = dyresults.samples[-1]
        return bf_values
    
    def CloudProperties(self,samplesSize=10000):
        '''
            Extracts some cloud properties from the posterior distributions.
            Posteriors are treated as if they were Gaussian. Only the mean
            and standard deviation are reported.
            Outputs an astropy table.
        '''
        
        samples = self.PosteriorSamples(samplesSize)
        Nclouds = int(samples.shape[1]/6)
        
        tab = []
        for nn in range(Nclouds):
            m_plx = np.mean(samples[:,0+nn*6])
            s_plx = np.std(samples[:,0+nn*6])
            m_q = np.mean(samples[:,1+nn*6])
            s_q = np.std(samples[:,1+nn*6])
            m_u = np.mean(samples[:,2+nn*6])
            s_u = np.std(samples[:,2+nn*6])

            Cov = np.cov(samples[:,1+nn*6],samples[:,2+nn*6])
            if np.linalg.det(Cov) <= 0.0:
                # if it does not work, add a little noise!
                Cov *= (1+ 1.e-4*np.random.randn(4).reshape(2,2))
            
            s = np.asarray([m_q,m_u])
            dMaha_0 = np.sqrt(np.dot(s,np.dot(np.linalg.inv(Cov),s)))
            tab_nn = [m_plx,m_q,m_u,s_plx,s_q,s_u,dMaha_0]
            tab.append(tab_nn)
        tab = np.asarray(tab)

        rows = []
        [rows.append('Cloud '+str(nn)) for nn in range(Nclouds)]
        table = Table([rows,tab[:,0],tab[:,3],tab[:,1],\
                        tab[:,4],tab[:,2],tab[:,5],tab[:,6]],\
                        names=['Clouds','plxC','err_plxC','qC','err_qC',\
                                'uC','err_uC','dMaha0'],\
                        dtype=['U8','f8','f8','f8','f8','f8','f8','f8'])
        return table
    
    def printCloudProperties(self,samplesSize=10000,numformat='%.4f'):
        '''
            Print in the terminal the astropy table outputed by CloudProperties
        '''
        propTable = self.CloudProperties(samplesSize=samplesSize)
        #
        propTable['plxC'].format = numformat
        propTable['err_plxC'].format = numformat
        propTable['qC'].format = numformat
        propTable['err_qC'].format = numformat
        propTable['uC'].format = numformat
        propTable['err_uC'].format = numformat
        propTable['dMaha0'].format = numformat
        print(propTable)
        return

    def showPosteriors(self,samplesSize=10000):
        '''
            Prints on the front a corner plot of all the model parameters.
        '''
        samples = self.PosteriorSamples(samplesSize)
        ndim = samples.shape[1]
        Nclouds = int(ndim/6)
        labels = [r'$\varpi_{\mathrm{C}}$ [mas]',\
                    r'$q_{\mathrm{C}}$ [$\%$]',\
                    r'$u_{\mathrm{C}}$ [$\%$]',\
                    r'$C_{qq}^{\mathrm{int}}$ [$\%^2$]',\
                    r'$C_{uu}^{\mathrm{int}}$ [$\%^2$]',\
                    r'$C_{qu}^{\mathrm{int}}$ [$\%^2$]']
        plt.rcParams['font.size'] = 9
        #
        f_size = np.minimum(Nclouds*6,9.5)
        f = plt.figure(figsize=(f_size,f_size))
        # convert polarization in [%]
        for ci in range(Nclouds):
            samples[:,[1+6*ci,2+6*ci]] *= 100
            samples[:,[3+6*ci,4+6*ci,5+6*ci]] *= 10000
        #
        corner.corner(samples,smooth=1.,color='C0',\
                        labels=labels*Nclouds,\
                        label_kwargs={'fontsize':6,'usetex':True},\
                        quantiles=[0.16,0.5,0.84],\
                        show_titles=True,\
                        title_kwargs={'fontsize':6,'usetex':True},\
                        fig=f,labelpad=.04,plot_datapoints=False)
        plt.show()
        return
    
    def showPosteriors_meanPol(self,samplesSize=10000):
        '''
            Same as showPosteriors but only show (plx,q,u) for each layer.
            The parameters characterizing the intrinsic scatter are dropped.
        '''
        samples = self.PosteriorSamples(samplesSize)
        ndim = samples.shape[1]
        Nclouds = int(ndim/6)
        labels = [r'$\varpi_{\mathrm{C}}$ [mas]',\
                    r'$q_{\mathrm{C}}$ [$\%$]',\
                    r'$u_{\mathrm{C}}$ [$\%$]']
        plt.rcParams['font.size'] = 12
        #
        f_size = np.minimum(Nclouds*6,9.5)
        f = plt.figure(figsize=(f_size,f_size))
        # convert polarization in [%]
        # and extract required columns
        cols = []
        for ci in range(Nclouds):
            samples[:,[1+6*ci,2+6*ci]] *= 100
            cols = np.concatenate((cols,ci*6+np.arange(3)))
        samples = samples[:,cols.astype('int')]
        #
        corner.corner(samples,smooth=1.,color='C0',\
                        labels=labels*Nclouds,\
                        label_kwargs={'fontsize':8,'usetex':True},\
                        quantiles=[0.16,0.5,0.86],\
                        show_titles=True,\
                        title_kwargs={'fontsize':8,'usetex':True},\
                        fig=f,labelpad=.04,plot_datapoints=False)
        plt.show()
        return

    def overPlot_QUMUClouds(self,**kwargs):
        '''
            Overplots on an existing figure the results for each layer
            in the (q,u)-mu plane with 2.5,16,50,84,97.5 percentiles.
        '''
        #
        samples = self.PosteriorSamples()
        Nclouds = int(samples.shape[1]/6)
        
        cols = np.arange(Nclouds) * 6
        
        plx_c = samples[:,cols]/1000.   # cloud's plx in as
        
        q_c = np.cumsum(samples[:,1+cols]*100.,axis=1)  # cumulative of clouds' q Stokes in %
        u_c = np.cumsum(samples[:,2+cols]*100.,axis=1)  # cumulative of clouds' u Stokes in %

        xlims = plt.gca().get_xlim()
        ylims = plt.gca().get_ylim()
        
        # look at the distributions
        plx_ = np.percentile(plx_c,[97.5,84,50,16,2.5],axis=0).T
        dmsC_ = 5.*np.log10(1./plx_)-5. # distance modulus of clouds
        
        qC_ = np.percentile(q_c,[2.5,16,50,84,97.5],axis=0).T
        uC_ = np.percentile(u_c,[2.5,16,50,84,97.5],axis=0).T
        
        # prepare for plotting
        if Nclouds == 1:
            x = np.asarray([dmsC_[0,4],xlims[1]])[None,:]
        else:
            x = np.asarray([dmsC_[:,4],np.concatenate((dmsC_[1:,0],np.asarray([xlims[1]])))]).T
        #
        if np.median(dmsC_[:,2])<xlims[0]:
            x[0,0] = xlims[0]
            dmsC_[0,:-1] *= 0
        else:
            pass
        #
        for nn in range(Nclouds):
            # print q and u
            plt.fill_between(x[nn],qC_[nn,0],qC_[nn,-1],
                                color='C2',alpha=.25,linewidth=.5,linestyle=':')
            plt.fill_between(x[nn],qC_[nn,1],qC_[nn,-2],
                                color='C2',alpha=.25,linewidth=.5,linestyle='--')
            #
            plt.fill_between(x[nn],uC_[nn,0],uC_[nn,-1],
                                color='C4',alpha=.25,linewidth=.5,linestyle=':')
            plt.fill_between(x[nn],uC_[nn,1],uC_[nn,-2],
                                color='C4',alpha=.25,linewidth=.5,linestyle='--')
            # print mu_c
            plt.fill_betweenx(ylims,dmsC_[nn,0],dmsC_[nn,-1],
                                color='C3',alpha=.2,linewidth=1.5,linestyle=':')
            plt.fill_betweenx(ylims,dmsC_[nn,1],dmsC_[nn,-2],
                                color='C3',alpha=.2,linewidth=1.5,linestyle='--')
            #
            if nn == 0:
                plt.plot(x[nn],[qC_[nn,2],qC_[nn,2]],
                                color='C2',linewidth=1,label=r'$q_{\rm{C}}$ mod.')
                plt.plot(x[nn],[uC_[nn,2],uC_[nn,2]],
                                color='C4',linewidth=1,label=r'$u_{\rm{C}}$ mod.')
                #
                plt.plot([dmsC_[nn,2],dmsC_[nn,2]],ylims,
                                color='C3',linewidth=1,label=r'$\mu_{\rm{C}}$ mod.')
                #
            else:
                plt.plot(x[nn],[qC_[nn,2],qC_[nn,2]],
                                color='C2',linewidth=1)
                plt.plot(x[nn],[uC_[nn,2],uC_[nn,2]],
                                color='C4',linewidth=1)
                #
                plt.plot([dmsC_[nn,2],dmsC_[nn,2]],ylims,
                                color='C3',linewidth=1)
                #
            #
        #
        return
    #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all!                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
