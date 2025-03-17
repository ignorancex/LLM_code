# -*- coding: utf-8 -*-
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   classes Priors of BISP-1
#
# --- 
#
#   @author: V.Pelgrims
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from astropy.io import ascii
from astropy.table import Table

#from dynesty import utils as dyfunc
#import dynesty
#import corner

#import bisp1_logll as logL

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class LayerPriors:
    '''
        Object LayerPriors.
        Used to determine types and values of the priors for each of the six
        parameters characterizing one layer.
        
        Parameters are: plx, q, u, Cqq, Cuu, Cqu.
        
        _type must be 'Flat' or 'Gauss' for uniform or Gaussian priors.
        
        _val1 and _val2 are minimum and maximum if prior type = 'Flat'
        _val1 and _val2 are mean and stddev is prior type = 'Gauss'
        
        **Note** that for the plx parameters, values in the input array are
        reverted for the 'Flat' prior (historical reason)
    '''
    def __init__(self,types,values):
        self.plx_type = types[0]
        if types[0] == 'Flat':
            self.plx_val2 = values[0]
            self.plx_val1 = values[6]
        elif types[0] == 'Gauss':
            self.plx_val1 = values[0]
            self.plx_val2 = values[6]
        else:
            print('outch')
        #
        self.q_type = types[1]
        self.q_val1 = values[1]
        self.q_val2 = values[7]
        #
        self.u_type = types[2]
        self.u_val1 = values[2]
        self.u_val2 = values[8]
        #
        self.Cqq_type = types[3]
        self.Cqq_val1 = values[3]
        self.Cqq_val2 = values[9]
        #
        self.Cuu_type = types[4]
        self.Cuu_val1 = values[4]
        self.Cuu_val2 = values[10]
        #
        self.Cqu_type = types[5]
        self.Cqu_val1 = values[5]
        self.Cqu_val2 = values[11]
        #
#
    
class ModelPriors:
    '''
        Object ModelPriors.
        Gather LayerPriors object to build models with given name
        corresponding to a given number of layers along sightlines.
        
        Functions getTypeList() and getValueList() return the list of list
        of list with types and values for each parameters for each layer
        for a given model.
    '''
    def __init__(self,modelName,types,values):
        if modelName == 'OneLayer':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.nlayers = 1
        elif modelName == 'TwoLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.nlayers = 2
        elif modelName == 'ThreeLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.nlayers = 3
        elif modelName == 'FourLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.layer_4 = LayerPriors(types[3],values[3])
            self.nlayers = 4
        elif modelName == 'FiveLayers':
            self.layer_1 = LayerPriors(types[0],values[0])
            self.layer_2 = LayerPriors(types[1],values[1])
            self.layer_3 = LayerPriors(types[2],values[2])
            self.layer_4 = LayerPriors(types[3],values[3])
            self.layer_5 = LayerPriors(types[4],values[4])
            self.nlayers = 5
        else:
            raise ValueError('unknown modelName')
        #

    #
    def getTypeList(self):
        ty_list = []
        def getTyp(tylist,theL):
            tylist.append([theL.plx_type,theL.q_type,theL.u_type,\
                            theL.Cqq_type,theL.Cuu_type,theL.Cqu_type])
            return tylist

        if self.nlayers>=1:
            ty_list = getTyp(ty_list,self.layer_1)
        if self.nlayers>=2:
            ty_list = getTyp(ty_list,self.layer_2)
        if self.nlayers>=3:
            ty_list = getTyp(ty_list,self.layer_3)
        if self.nlayers>=4:
            ty_list = getTyp(ty_list,self.layer_4)
        if self.nlayers>=5:
            ty_list = getTyp(ty_list,self.layer_5)
        #
        return ty_list


    def getValueList(self):
        va_list = []
        def getVal(valist,theL):
            if theL.plx_type == 'Flat':
                valist.append([theL.plx_val2,theL.q_val1,theL.u_val1,\
                                theL.Cqq_val1,theL.Cuu_val1,theL.Cqu_val1,\
                                theL.plx_val1,theL.q_val2,theL.u_val2,\
                                    theL.Cqq_val2,theL.Cuu_val2,theL.Cqu_val2])
            else: # plx_type = 'Gauss'
                valist.append([theL.plx_val1,theL.q_val1,theL.u_val1,\
                                theL.Cqq_val1,theL.Cuu_val1,theL.Cqu_val1,\
                                theL.plx_val2,theL.q_val2,theL.u_val2,\
                                    theL.Cqq_val2,theL.Cuu_val2,theL.Cqu_val2])
            return valist
        #
        if self.nlayers>=1:
            va_list = getVal(va_list,self.layer_1)
        if self.nlayers>=2:
            va_list = getVal(va_list,self.layer_2)
        if self.nlayers>=3:
            va_list = getVal(va_list,self.layer_3)
        if self.nlayers>=4:
            va_list = getVal(va_list,self.layer_4)
        if self.nlayers>=5:
            va_list = getVal(va_list,self.layer_5)
        #
        return va_list


class Priors:
    def __init__(self,stars,inputTypes=None,inputValues=None):
        '''
            Initialization of the object.
            Default (using `stars`) or using the types and values given in
            optional arguments.
            Input:
                - `stars`: a Stars object
                - inputTypes: a list (up to 5 elements) of lists (number of
                                layer) of list (length of 6 for all 6 model
                                parameters per layer) to specify the type of
                                priors.
                - inputValues: a list (up to 5 elements) of lists (number of
                                layer) of list (length of 12 for all 6 model
                                parameters per layer) to specify the range of
                                priors or mean and stddev.
                    see updatePriors()

            **Note  **: the default flat prior on cloud parallaxes are defined
                        by the stars. By default the decomposition will search
                        for clouds within the range encompased by the stars'
                        plx. This might not be the case in general!
            **Note 2**: There is an hard-coded threshold of minimum 5 stars
                        between each cloud, when more than one cloud are to be
                        considered. Be careful if you intend to change this and
                        make sure this is also propagated to bisp_logll.
        '''

        plx_s = np.sort(stars.stararray[2])

        def feedit(nclouds):
            typ = []
            val = []
            for nn in range(nclouds):
                typ.append(['Flat','Flat','Flat','Flat','Flat','Flat'])
                val.append([stars.plx_max,-stars.qu_max,-stars.qu_max,\
                                    0.,0.,-1.e-4,\
                                    stars.plx_min,stars.qu_max,stars.qu_max,\
                                        1.e-4,1.e-4,1.e-4])
                val[nn][6] = plx_s[5*(nclouds-1-nn)]
                # the '5' is our minimum allowed stars in between clouds.
                # It is necessary to comply with computation of prior_transform
                # in bisp_logll.
            return typ,val

        # OneLayer:
        ty,va = feedit(1)
        self.OneLayer = ModelPriors('OneLayer',ty,va)

        # TwoLayers:
        ty,va = feedit(2)
        self.TwoLayers = ModelPriors('TwoLayers',ty,va)

        # ThreeLayers:
        ty,va = feedit(3)
        self.ThreeLayers = ModelPriors('ThreeLayers',ty,va)

        # FourLayers:
        ty,va = feedit(4)
        self.FourLayers = ModelPriors('FourLayers',ty,va)

        # FiveLayers:
        ty,va = feedit(5)
        self.FiveLayers = ModelPriors('FiveLayers',ty,va)

        # update priors if needed
        self.updatePriors(inputTypes=inputTypes,inputValues=inputValues)


    #
    def updatePriors(self,inputTypes=None,inputValues=None):
        '''
            Update the default priors according to the inputs.
        
            Inputs are list of list (of list):
                [list_for_oneLayerModel,list_for_twoLayerModel,...]
            with
                list_for_oneLayerModel = [list_of_param]
                list_for_twoLayerModel = [list_for_1st_layer,list_for_2nd_layer]
                ...

            Parameters are read in the order: plx,q,u,Cqq,Cuu,Cqu.
            Values must be specified as:
            [max_plx,min_q,min_u,min_Cqq,min_Cuu,min_Cqu,
                min_plx,max_q,max_u,max_Cqq,max_Cuu,max_Cqu] if all 'Flat'
            or
            [mean_plx,mean_q,mean_u,mean_Cqq,mean_Cuu,mean_Cqu,
                std_plx,std_q,std_u,std_Cqq,std_Cuu,std_Cqu] if all 'Gauss'.

            **Note**: the inverted order of min and max for the plx in case of
            'Flat'.
            
        '''
        if inputTypes is not None:
            for mm in range(len(inputTypes)):
                n = len(inputTypes[mm])
                if n == 1:
                    values = self.OneLayer.getValueList()
                    self.OneLayer = ModelPriors('OneLayer',inputTypes[mm],\
                                                    values)
                elif n == 2:
                    values = self.TwoLayers.getValueList()
                    self.TwoLayers = ModelPriors('TwoLayers',inputTypes[mm],\
                                                    values)
                elif n == 3:
                    values = self.ThreeLayers.getValueList()
                    self.ThreeLayers = ModelPriors('ThreeLayers',\
                                                    inputTypes[mm],values)
                elif n == 4:
                    values = self.FourLayers.getValueList()
                    self.FourLayers = ModelPriors('FourLayers',inputTypes[mm],\
                                                    values)
                elif n == 5:
                    values = self.FiveLayers.getValueList()
                    self.FiveLayers = ModelPriors('FiveLayers',inputTypes[mm],\
                                                    values)
                else:
                    raise ('Invalid size of priorType inputs')
        if inputValues is not None:
            for mm in range(len(inputValues)):
                n = len(inputValues[mm])
                if n == 1:
                    types = self.OneLayer.getTypeList()
                    self.OneLayer = ModelPriors('OneLayer',types,\
                                                        inputValues[mm])
                elif n == 2:
                    types = self.TwoLayers.getTypeList()
                    self.TwoLayers = ModelPriors('TwoLayers',types,\
                                                        inputValues[mm])
                elif n == 3:
                    types = self.ThreeLayers.getTypeList()
                    self.ThreeLayers = ModelPriors('ThreeLayers',types,\
                                                        inputValues[mm])
                elif n == 4:
                    types = self.FourLayers.getTypeList()
                    self.FourLayers = ModelPriors('FourLayers',types,\
                                                        inputValues[mm])
                elif n == 5:
                    types = self.FiveLayers.getTypeList()
                    self.FiveLayers = ModelPriors('FiveLayers',types,\
                                                        inputValues[mm])
                else:
                    raise ('Invalid size of priorType inputs')
        return
    #

    def printPriors(self,modelName,numformat='%.4f'):
        '''
            Looks into the prior types and values of a given model and prints
            in the terminal a table with parameter names, types of priors and
            values used to specify the functional forms for each parameter and
            each layer.
            - modelName: str to specify the name of the model for which priors
                            are of interest
            - numformat: (optional), str to specific the numerical format to
                            display numerical values.
        '''
        if modelName == 'OneLayer':
            Nclouds = int(1)
            tab_type = np.asarray(self.OneLayer.getTypeList())
            tab_vals = np.asarray(self.OneLayer.getValueList())
            #
        elif modelName == 'TwoLayers':
            Nclouds = int(2)
            tab_type = np.asarray(self.TwoLayers.getTypeList())
            tab_vals = np.asarray(self.TwoLayers.getValueList())
            #
        elif modelName == 'ThreeLayers':
            Nclouds = int(3)
            tab_type = np.asarray(self.ThreeLayers.getTypeList())
            tab_vals = np.asarray(self.ThreeLayers.getValueList())
            #
        elif modelName == 'FourLayers':
            Nclouds = int(4)
            tab_type = np.asarray(self.FourLayers.getTypeList())
            tab_vals = np.asarray(self.FourLayers.getValueList())
            #
        elif modelName == 'FiveLayers':
            Nclouds = int(5)
            tab_type = np.asarray(self.FiveLayers.getTypeList())
            tab_vals = np.asarray(self.FiveLayers.getValueList())
            #
        else:
            message = ("Wrong model name. It should be one of: \n",\
                        "'OneLayer', 'TwoLayers', 'ThreeLayers', ",\
                            "'FourLayers', 'FiveLayers'.")
            raise ValueError(message)
        #
        rows = []
        [rows.append('Cloud '+str(nn)) for nn in range(Nclouds)]
        
        table_type = Table([rows,tab_type[:,0],tab_type[:,1],tab_type[:,2],\
                                tab_type[:,3],tab_type[:,4],tab_type[:,5]],\
                            names=['Clouds','plx','q','u','Cqq','Cuu','Cqu'],\
                            dtype=['U8','U8','U8','U8','U8','U8','U8'])



        table_vals = Table([rows,tab_vals[:,0],tab_vals[:,1],tab_vals[:,2],\
                                tab_vals[:,3],tab_vals[:,4],tab_vals[:,5],\
                                tab_vals[:,6],tab_vals[:,7],tab_vals[:,8],\
                                tab_vals[:,9],tab_vals[:,10],tab_vals[:,11]],\
                            names=['Clouds','plx_v1','q_v1','u_v1',\
                                    'Cqq_v1','Cuu_v1','Cqu_v1',\
                                    'plx_v2','q_v2','u_v2',\
                                    'Cqq_v2','Cuu_v2','Cqu_v2'],\
                            dtype=['U8','f8','f8','f8','f8','f8','f8',\
                                        'f8','f8','f8','f8','f8','f8'])
        #
        print('\n Prior Types: \n ------------ \n')
        print(table_type,'\n')
        
        for key in table_vals.keys()[1:]:
            table_vals[key].format = numformat
        print('\n Prior Values: \n ------------- \n')
        print(table_vals,'\n')
        print("For 'Flat' priors 'v1' and 'v2' refer to minimum and maximum",\
                "\n except for 'plx' where there are inverted as compared to",\
                    "plx_val1 and plx_val2.")
        print("For 'Gauss' priors 'v1' and 'v2' refer to mean and stddev. \n")
#


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all!                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
