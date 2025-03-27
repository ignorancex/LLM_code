# -*- coding: utf-8 -*-
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   class Bisp of BISP-1
#
#   Bayesian Inference of Starlight Polarization in 1D
#
# --- 
#
#   Implementation of the method to perform the decomposition of optical
#   polarization of stars with known distances along distance as presented in
#   Pelgrims et al. 2022, A&A, ..., ...
#
#   Relies on children classes: Stars, Priors, Decomp   
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

from . import bisp1_logll as logL
from .Decomp import *

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Bisp:
    '''
        Object Bisp
        Main object. Contains information on star data and priors and allows
        for the application of the maximum-likelihood analysis of the data
        for the different models (assumed different number of layers along the
        LOS) and to access the results and compare the performance of the
        different models.
    '''
    def __init__(self,stars,priors,**kwargs):
        '''
            Initialization from input:
            - stars: a Stars object
            - priors: a Priors object
            - **kwargs: used to control ```dynesty``` settings
        '''
        self.stars = stars
        self.priors = priors
        self.testedModel = []
        #
        # settings for dynesty inference  - - - - - - - - - - - - - -
        self.default_dynestycontrol()
        if kwargs is not None:
            self.update_dynestycontrol(kwargs)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
    # Call and run the different possible models:
    # One-layer model:
    # ---------------
    def run_OneLayer(self,fromFresh=True):
        '''
            Run the 1-layer model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '1Layer'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.OneLayer.getTypeList(),\
                                        self.priors.OneLayer.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if existing previous run has been performed
        if ((not hasattr(self,'OneLayer')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.OneLayer.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.OneLayer = Result
        #
        if 'OneLayer' not in self.testedModel:
            self.testedModel += ['OneLayer']
        #
        return

    # Two-layer model:
    # ----------------
    def run_TwoLayers(self,fromFresh=True):
        '''
            Run the 2-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '2Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.TwoLayers.getTypeList(),\
                                        self.priors.TwoLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'TwoLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,2*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.TwoLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.TwoLayers = Result
        #
        if 'TwoLayers' not in self.testedModel:
            self.testedModel += ['TwoLayers']
        #
        return

    # Three-layer model:
    # ------------------
    def run_ThreeLayers(self,fromFresh=True):
        '''
            Run the 3-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '3Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.ThreeLayers.getTypeList(),\
                                        self.priors.ThreeLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'ThreeLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,3*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.ThreeLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.ThreeLayers = Result
        #
        if 'ThreeLayers' not in self.testedModel:
            self.testedModel += ['ThreeLayers']
        #
        return

    # Four-layer model:
    # ------------------
    def run_FourLayers(self,fromFresh=True):
        '''
            Run the 4-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '4Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.FourLayers.getTypeList(),\
                                        self.priors.FourLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'FourLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,4*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.FourLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.FourLayers = Result
        #
        if 'FourLayers' not in self.testedModel:
            self.testedModel += ['FourLayers']
        #
        return

    
    # Five-layer model:
    # ------------------
    def run_FiveLayers(self,fromFresh=True):
        '''
            Run the 5-layers model on the stars using the priors
        '''
        # setup the sampling experiment
        # Point to the right log-likelihood function
        def loglike(theta):
            modelname = '5Layers'
            return logL.def_model(theta,modelname,self.stars.stararray)
        # Point to the right prior function
        def prior_transform(unibox):
            return logL.def_prior_IntSc(unibox,\
                                        self.priors.FiveLayers.getTypeList(),\
                                        self.priors.FiveLayers.getValueList(),\
                                        self.stars.stararray)
        # initialization of dynesty sampler
        init_dyn = self.dynesty_settings
        # check if a run has been performed previously
        if ((not hasattr(self,'FiveLayers')) or fromFresh):
            # if not, initialize it
            sampler = dynesty.NestedSampler(loglike,prior_transform,5*6,\
                                                nlive=init_dyn['nlive'],\
                                                sample='auto')
        else:
            # if yes, load it to start from there
            sampler = self.FiveLayers.sampler
        #
        # run dynesty sampler
        sampler.run_nested(dlogz=init_dyn['dlogz'],\
                            maxiter=init_dyn['maxiter'],\
                                print_progress=not(init_dyn['silent']))
        # wrappe the result as a `Decomp` object
        Result = Decomp(sampler)
        # add the results
        self.FiveLayers = Result
        #
        if 'FiveLayers' not in self.testedModel:
            self.testedModel += ['FiveLayers']
        #
        return

    # # - - - - # #

    def GetSummaryStat(self):
        '''
            Searches for already evaluated models and extract the summary
            statistics: evidence, error on evidence, AIC, and also compute
            the probability of each tested model that it is actually the
            best model (in the pool of available model) that minimizes the
            loss of information.
            Returns an astropy table.
        '''
        testedModel = []
        sumStats = []
        
        # by default, consider NoLayer model
        testedModel.append('NoLayer')
        stats = [np.nan,np.nan,self.stars.NoLayerLL]
        stats = np.concatenate((stats,[2*0 - 2* stats[2]]))
        sumStats.append(stats)
        #
        #
        if hasattr(self,'OneLayer'):
            testedModel.append('OneLayer')
            stats = np.asarray(self.OneLayer.SummaryStat())
            stats = np.concatenate((stats,[2*6 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'TwoLayers'):
            testedModel.append('TwoLayers')
            stats = np.asarray(self.TwoLayers.SummaryStat())
            stats = np.concatenate((stats,[2*12 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'ThreeLayers'):
            testedModel.append('ThreeLayers')
            stats = np.asarray(self.ThreeLayers.SummaryStat())
            stats = np.concatenate((stats,[2*18 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'FourLayers'):
            testedModel.append('FourLayers')
            stats = np.asarray(self.FourLayers.SummaryStat())
            stats = np.concatenate((stats,[2*24 - 2*stats[2]]))
            sumStats.append(stats)
        if hasattr(self,'FiveLayers'):
            testedModel.append('FiveLayers')
            stats = np.asarray(self.FiveLayers.SummaryStat())
            stats = np.concatenate((stats,[2*30 - 2*stats[2]]))
            sumStats.append(stats)
        #
        if sumStats != []:
            sumStats = np.asarray(sumStats)
            # sumStats[:,3] records the AIC obtained for each model
            # from whose probabilities to minimizing the information loss are get:
            P_jm = np.exp((np.min(sumStats[:,3]) - sumStats[:,3])/2.)
    
            table = Table([testedModel,sumStats[:,0],sumStats[:,1],sumStats[:,2],\
                            sumStats[:,3],P_jm],\
                            names=['Model','Z','errZ','max-LL','AIC','Pbest'],\
                            dtype=['U16','f8','f8','f8','f8','f8'])
        else:
            table = Table([['None'],[0],[0],[0],[0],[0]],\
                            names=['Model','Z','errZ','max-LL','AIC','Pbest'],\
                            dtype=['U16','f8','f8','f8','f8','f8'])
        #
        return table

    def printSummaryStat(self,numformat='%.4f'):
        '''
            Prints on the terminal the outputed table from GetSummaryStat().
            numformat (optional) is used to specify the numerical format to
            be used while printing.
        '''
        sumTable = self.GetSummaryStat()
        sumTable['Z'].format = numformat
        sumTable['errZ'].format = numformat
        sumTable['max-LL'].format = numformat
        sumTable['AIC'].format = numformat
        sumTable['Pbest'].format = numformat
        print(sumTable)
        return
    
    def writeSummaryStat(self,filename,numformat='%.4f'):
        '''
            Writes on the disk the output of GetSUmmaryStat()
            Input:
                - filename: str to specify the location and name of the file
                            where to write the file.
                - numformat: to specify the numerical format to be used
        '''
        sumTable = self.GetSummaryStat()
        sumTable['Z'].format = numformat
        sumTable['errZ'].format = numformat
        sumTable['max-LL'].format = numformat
        sumTable['AIC'].format = numformat
        sumTable['Pbest'].format = numformat
        if filename[-4:] != '.csv':
            filename + '.csv'
        ascii.write(sumTable,comment='#',delimiter=',',\
                        overwrite=True,output = filename)
        return
    
    def update_dynestycontrol(self,settingList):
        '''
            Updates the list of settings to control the behavior of dynesty.
            Input:
                - settingList: a dictionary to update specific variables
        '''
        prev_settings = self.dynesty_settings
        for key,val in settingList.items():
            if key in prev_settings:
                prev_settings[key] = val
            #
        #
        self.dynesty_settings = prev_settings
        return

    def default_dynestycontrol(self):
        '''
            Sets the default settings to control the nested sampling
            experiments using dynesty.
        '''
        self.dynesty_settings = {'nlive':500,'maxiter':10000,'dlogz':0.1,\
                                    'silent':True}
        return


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all!                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
