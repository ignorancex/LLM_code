"""Contains Fitter objects which are used for scaling a representation for chemical tagging purposes. And functions for compressing a representation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import apoNN.src.vectors as vectors
from scipy.stats import median_absolute_deviation as mad
import abc
from pyppca import ppca



def compress_masked_spectra(x,x_occam,d,tol=1E-4):
    """
    Compresses high-dimensional spectra into a lower dimensional representation using PCA for missing values.

    INPUTS
    ------
    x: np.masked_array
        array containing dataset of spectra with flagged values masked.
    x_occam: np.masked_array
        array containing occam open-cluster spectra with flagged values masked.
    d: int
        number of dimensions used for compression.
    OUTPUTS
    z: np.array
        compressed array containing all spectra encoded
    z_occam: np.array
        compressed array containing occam spectra encoded
    ppca: PCA object containing all the information about the PCA transform"""
    masked_x = np.copy(x.data)
    masked_x_occam = np.copy(x_occam.data)
    masked_x[x.mask]=np.nan
    masked_x_occam[x_occam.mask]=np.nan
    masked_all = np.concatenate((masked_x_occam,masked_x))
    C, ss, M, X, Ye = ppca(masked_all,d,True,threshold=tol)
    n_occam = len(masked_x_occam)
    z_occam,z = X[:n_occam],X[n_occam:]
    return z,z_occam,C


class BaseFitter(abc.ABC):
    """Abstract base class for Fitters from which other fitters inherit"""
    def reparametrize(self, vector):
        """function doing all the preliminary transformations (typically a change of basis) before applying the scaling"""
        pass
    
    def transform(self,vector,scaling=True):
        """transform a vector using RSCA"""
        #transformed_vector  = np.dot(vector.whitened(self.whitener)(),self.pca.components_.T)
        transformed_vector = self.reparametrize(vector)
        if scaling is True:
            transformed_vector = transformed_vector/self.scaling_factor

        if vector.__class__ is vectors.OccamVector:
            return vector.__class__(cluster_names = vector.cluster_names, val = transformed_vector)
        else:
            return vector.__class__(val=transformed_vector)
        
    def pooled_std(self,z:vectors.OccamVector):
        """estimates intacluster standard deviation using a pooled variance estimator"""
        num_stars = []
        variances = []
        whitened_z = self.transform(self.z_occam.cluster_centered,scaling=False).val
        for cluster in sorted(z.registry):
            cluster_idx = z.registry[cluster]
            #print(f"{len(cluster_idx)} stars")
            if len(cluster_idx)>1:
                num_stars.append(len(cluster_idx))
                variances.append(self.std(whitened_z[cluster_idx],axis=0)**2)

        variances = np.array(variances)
        num_stars = np.array(num_stars)
        return (((np.dot(num_stars-1,variances)/(np.sum(num_stars)-len(num_stars))))**0.5)[None,:]
    
    
    def get_scaling(self,z:vectors.Vector,is_pooled=False):
        if is_pooled is True:
            return self.pooled_std(z) 
        else:
            return self.std(self.transform(self.z_occam.cluster_centered,scaling=False).val,axis=0)[None,:]
        
    @staticmethod
    def relative_modifier(sigma1,sigma2=1):
        return np.sqrt(np.abs(sigma1**2*sigma2**2/(sigma1**2-sigma2**2)))
    
    
    def calculate_scaling(self,z_occam,std,is_pooled,use_relative_scaling):
        """calculates the scaling factor for each dimension"""
        if use_relative_scaling is True:
            scaling_factor = self.get_scaling(z_occam,is_pooled)
            scaling_factor[scaling_factor>=1]=0.9999999 #put a very large value to ensure that dimensions randomly greater than 1 are zeroed out           
            scaling_factor = np.array([self.relative_modifier(std) for std in list(scaling_factor[0])])
        else:
            scaling_factor = self.get_scaling(z_occam,is_pooled)
         
        return scaling_factor
    
class StandardFitter(BaseFitter):
    def __init__(self,z:vectors.Vector,z_occam:vectors.OccamVector,use_relative_scaling=True, use_whitening=True,is_pooled=True,is_robust=True):
        """The StandardFitter, as used in the paper. Scales the representation through a 3 step procedure consisting of 1) whitening 2) change-of-basis 3) rescaling.
        INPUTS
        ------
        z: vector.Vector
            Vector containing large dataset of stellar spectra
        z_occam:vector.OccamVector
            Vector containing dataset of known occam cluster member stars
        use_relative_scaling: Boolean
            When True uses the relative scaling presented in the paper, otherwise use a simple scaling based only on intercluster standard deviation.
        is_pooled: Boolean
            If True use a pooled variance estimator, otherwise used standard variance estimator.
        is_robust: Boolean
            If True, replace the mean standard deviation with the more robust mean absolute deviation in the scaling calculations.
            """
        
        self.z = z
        self.z_occam = z_occam
        self.use_relative_scaling = use_relative_scaling
        self.is_pooled = is_pooled
        if use_whitening is True:
            self.whitener = PCA(n_components=self.z.val.shape[1],whiten=True)
        else:
            self.whitener = Normalizer()        
    
        if is_robust is True:
            self.std = mad #calculate std using meadian absolute deviation
        else:
            def std_ddfof1(x,axis=0,ddof=1):
                return np.std(x,axis=axis,ddof=ddof)
            self.std = std_ddfof1
            
            
        self.pca = PCA(n_components=self.z.val.shape[1]) #pca is used for performing a change of basis    
        
        self.whitener.fit(self.z.centered().val) #we learn a whitening transform of our dataset
        self.pca.fit(self.z_occam.cluster_centered.whitened(self.whitener).val) #we learn a change of basis
        self.scaling_factor = self.calculate_scaling(self.z_occam,self.std,self.is_pooled,self.use_relative_scaling)
            
            
    def reparametrize(self, vector):
        """We first whiten then perform a change of basis"""
        transformed_vector  = np.dot(vector.whitened(self.whitener).val,self.pca.components_.T)
        return transformed_vector
    
    
     
       
class SimpleFitter(BaseFitter):
    """This is a fitter that carries out the rescaling in the original basis of the representation (ie no change of basis)."""
    def __init__(self,z:vectors.Vector,z_occam:vectors.OccamVector,use_relative_scaling=True,is_pooled=True,is_robust=True):
        self.z = z
        self.z_occam = z_occam
        self.normalizer = Normalizer() #NOT AN ACTUAL WHITENING STEP ONLY STANDARDIZES
            
        if is_robust is True:
            self.std = mad #calculate std using meadian absolute deviation
        else:
            def std_ddfof1(x,axis=0,ddof=1):
                return np.std(x,axis=axis,ddof=ddof)
            self.std = std_ddfof1
            
        self.normalizer.fit(self.z.centered().val)
        if use_relative_scaling is True:
            self.scaling_factor = self.get_scaling(z_occam,is_pooled)
            self.scaling_factor[self.scaling_factor>=1]=0.9999 #ensure that dimensions randomly greater than 1 are zeroed out           
            self.scaling_factor = np.array([self.relative_modifier(std) for std in list(self.scaling_factor[0])])

        else:
            self.scaling_factor = self.std(self.transform(self.z_occam.cluster_centered,scaling=False)(),axis=0)[None,:]

    def reparametrize(self, vector):
        """Standardizes vectors"""
        transformed_vector  = vector.whitened(self.normalizer).val
        return transformed_vector


    
class Normalizer():
    """Standardizes the input data. Can be used as an alternative to whitening"""
    def fit(self,z):
        self.scaling_factor = np.std(z,axis=0)[None,:]
        
    def transform(self,z):
        return z/self.scaling_factor        
        

class EmptyFitter():
    def __init__(self,z:vectors.Vector,z_occam:vectors.OccamVector):
        """This fitter does nothing... But that's sometimes a good thing when you want to benchmark things"""
        self.z = z
        self.z_occam =z_occam

    def transform(self,vector):
        """transform a vector in a way that unit vector has variance one"""
        if vector.__class__ is vectors.OccamVector:
            return vector.__class__(cluster_names = vector.cluster_names, val = vector.val)
        else:
            return vector.__class__(val=vector.val)


class TopNFitter(StandardFitter):
    def __init__(self,z:vectors.Vector,z_occam:vectors.OccamVector,use_relative_scaling=True, use_whitening=True,is_pooled=True,is_robust=True,N=3):        
        """The StandardFitter, as used in the paper. Scales the representation through a 3 step procedure consisting of 1) whitening 2) change-of-basis 3) rescaling.
        INPUTS
        ------
        z: vector.Vector
            Vector containing large dataset of stellar spectra
        z_occam:vector.OccamVector
            Vector containing dataset of known occam cluster member stars
        use_relative_scaling: Boolean
            When True uses the relative scaling presented in the paper, otherwise use a simple scaling based only on intercluster standard deviation.
        is_pooled: Boolean
            If True use a pooled variance estimator, otherwise used standard variance estimator.
        is_robust: Boolean
            If True, replace the mean standard deviation with the more robust mean absolute deviation in the scaling calculations.
            """
        StandardFitter.__init__(self,z=z,z_occam=z_occam,use_relative_scaling=use_relative_scaling, use_whitening=use_whitening,is_pooled=is_pooled,is_robust=is_robust)
        not_smallest_3 = self.scaling_factor.argsort()[N:]
        self.scaling_factor[not_smallest_3]=10**5





