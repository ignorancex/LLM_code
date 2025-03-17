# -*- coding: utf-8 -*-
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   class Stars of BISP-1
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

class Stars(object):
    '''
        Object Stars.
        Loads and formats a star data sample and allow for quick visualization.
    '''
    def __init__(self,starSample):
        '''
            Initialization of the object from input.
            Input must be either
            - a (7,nStar) array with (q,u,plx,s_q,s_u,c_qu,s_plx)
            - or a filename pointing to a catalog in csv or ascii format.
            ! units !
                plx and s_plx must be in [mas]
                polarization q,u and uncertainties in fraction. (1 = 100%)
        '''
        if isinstance(starSample,str):
            starSample = Stars._getArray_from_Cat(starSample)
        #
        self.stararray = starSample
        self.plx_min = np.min(starSample[2])
        self.plx_max = np.max(starSample[2])
        self.qu_max = np.max([np.abs(starSample[0]),np.abs(starSample[1])])
        self.nStar = len(starSample[0])
        self.NoLayerLL = Stars._get_NoLayerLL(starSample)
        #
    #

    def _getArray_from_Cat(catname):
        '''
            Opens a catalog of starlight polarization data and returns
            an array for later uses. The names of the columns need to follow
            'q_obs','u_obs','plx','s_q','s_u','s_plx' (and 'c_qu', option)
        '''
        data_ = ascii.read(catname)
        #
        # check all the required fields are present in the data file
        required = ['q_obs','u_obs','s_q','s_u','plx','s_plx']
        if all(req in data_.keys() for req in required):
            pass
        else:
            message = ("Missing keys or wrong spelling in data table. \n "+\
                        "There should be: 'q_obs','u_obs','s_q','s_u','plx','s_plx' " +\
                            "with, optionally, 'c_qu'.")
            raise ValueError(message)
        #
        qV = np.asarray(data_['q_obs'])
        uV = np.asarray(data_['u_obs'])
        e_qV = np.asarray(data_['s_q'])
        e_uV = np.asarray(data_['s_u'])
        plx_obs = np.asarray(data_['plx'])       # plx in mas
        e_plx = np.asarray(data_['s_plx'])       # s_plx in mas
        #
        the_array = np.asarray([qV,uV,plx_obs,e_qV,e_uV,0*e_qV,e_plx])
        if 'c_qu' in data_.keys():
            the_array[5] = np.asarray(data_['c_qu'])
        #
        return the_array
    
    def _get_NoLayerLL(starSample):
        '''
            Compute the (log-)likelihood of the data given no dust layer,
            i.e. assuming that the measured polarization are due to noise
            only.
        '''
        stararray = starSample
        qs = stararray[0]; us = stararray[1]
        Cqq = stararray[3]**2 ; Cuu = stararray[4]**2 ; Cqu = stararray[5]
        detC = Cqq * Cuu - Cqu**2
        invCqq = Cuu / detC
        invCuu = Cqq / detC
        invCqu = - Cqu /detC
        chi2 = qs**2 * invCqq + us**2 * invCuu + 2 * qs * us * invCqu
        
        ll_i = np.sum(np.log(np.exp(-.5*chi2)/(2*np.pi*np.sqrt(detC))))
        return ll_i
    


    #@classmethod
    def showData_QUMU(self,**kwargs):
        '''
            Generates a (q,u),mu plot of the data.
            **kwargs={ylim,xlim,show_error,usetex}
        '''
        qs,us,plxs = self.stararray[:3,:]
        plxs = plxs/1000.   # convert plx from [mas] to [as]
        dms = 5.*np.log10(1./plxs) - 5. # compute distance modulus
        qs = 100. * np.copy(qs) ; us = 100. * np.copy(us)  # convert pol to [%]
        # preparing a figure frame with ad hoc size and ranges.
        plt.figure()
        plt.plot(dms,qs,'+',markersize=2,color='C0')
        plt.plot(dms,us,'+',markersize=2,color='C1')
        plt.gca().set_ylim(ymin = np.minimum(np.percentile(qs,1),\
                                    np.percentile(us,1)) - 0.2,\
                            ymax = np.maximum(np.percentile(qs,99),\
                                np.percentile(us,99)) + 0.2)
        ylims = plt.gca().get_ylim()
        xlims = plt.gca().get_xlim()
        #
        usetex = True
        show_error = True
        if kwargs is not None:
            for key,val in kwargs.items():
                if key == 'ylim':
                    ylims = val
                elif key == 'xlim':
                    xlims = val
                elif key == 'show_error':
                    show_error = val
                elif key == 'usetex':
                    usetex = val
                else:
                    raise ('Invalid arguments. Ignored.')

        plt.clf()
        #
        if show_error == False:
            plt.plot(dms,qs,'o',markerfacecolor='none',markersize=4,\
                        markeredgewidth=1.5,color='darkgreen',alpha=.5,\
                            label=r'$q_{\rm{V}}$ obs.')
            plt.plot(dms,us,'d',markerfacecolor='none',markersize=4,\
                        markeredgewidth=1.5,color='blue',alpha=.5,\
                            label=r'$u_{\rm{V}}$ obs.')
        elif show_error == True:
            #
            dms_bl = 5*(np.log10(1000./(self.stararray[2]+self.stararray[6]))-1)
            dms_up = 5*(np.log10(1000./(self.stararray[2]-self.stararray[6]))-1)
            qs = self.stararray[0]*100
            us = self.stararray[1]*100
            s_qs = self.stararray[3]*100
            s_us = self.stararray[4]*100
            plt.errorbar(x=dms,y=qs,xerr=[dms-dms_bl,dms_up-dms],\
                            yerr=s_qs,fmt='o',markerfacecolor='none',\
                            markersize=4,markeredgewidth=1.5,alpha=.25,\
                            color='darkgreen',capsize=0,elinewidth=1,\
                            label=r'$q_{\rm{V}}$ obs.');
            plt.errorbar(x=dms,y=us,xerr=[dms-dms_bl,dms_up-dms],\
                            yerr=s_us,fmt='d',markerfacecolor='none',\
                            markersize=4,markeredgewidth=1.5,alpha=.25,\
                            color='blue',capsize=0,elinewidth=1,\
                            label=r'$u_{\rm{V}}$ obs.');
            plt.plot(dms,qs,'o',markerfacecolor='none',markersize=4,\
                            markeredgewidth=1.5,color='darkgreen',alpha=.5)
            plt.plot(dms,us,'d',markerfacecolor='none',markersize=4,\
                            markeredgewidth=1.5,color='blue',alpha=.5)
            #

        # print the cosmetic
        plt.gca().set_ylim(ylims)
        plt.gca().set_xlim(xlims)
        plt.xlabel(r'$\mu$ Distance Modulus',fontsize=16,usetex=usetex)
        plt.ylabel(r'$(q_{\rm{V}},u_{\rm{V}})$ Stellar Stokes [\%]',fontsize=16,\
                    usetex=usetex)
        #
        return
    #

    def showData_QUscatter(self,**kwargs):
        '''
            Generates a scatter plot of the (q,u) data including uncertainties
            and coloured according to distance (modulus)
            **kwargs={cmin,cmax,usetex}
        '''

        mu = 5*(np.log10(1000./self.stararray[2])-1.)
        qs,us,s_qs,s_us = self.stararray[[0,1,3,4],:] * 100.

        plt.figure();
        ax = plt.gca();

        cmin = np.min(mu) ; cmax = np.max(mu)
        usetex = True
        if kwargs is not None:
            for key,val in kwargs.items():
                if key == 'cmin':
                    cmin = val
                elif key == 'cmax':
                    cmax = val
                elif key == 'usetex':
                    usetex = val
                else:
                    raise ('Invalid arguments. Ignored.')
        #
        s_c = np.copy(mu)
        s_c[mu<=cmin] = cmin
        s_c[mu>=cmax] = cmax
        s = ax.scatter(qs,us,c=s_c);

        norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        d_color = np.array([(mapper.to_rgba(v)) for v in s_c])

        for q,u,eq,eu,color in zip(qs,us,s_qs,s_us,d_color):
            plt.plot(q, u, 'o', color=color)
            plt.errorbar(q, u, xerr=eq,yerr=eu, lw=1, capsize=3, color=color)

        cb = plt.colorbar(s)
        cb.set_label(r'$\mu$ Distance Modulus')
        plt.axis('square')
        plt.xlabel(r'$q_{\rm{V}}$ [\%] Stellar Stokes',fontsize=16,usetex=True)
        plt.ylabel(r'$u_{\rm{V}}$ [\%] Stellar Stokes',fontsize=16,usetex=True)
        plt.show()
        #
        return
    #

#

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all!                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
