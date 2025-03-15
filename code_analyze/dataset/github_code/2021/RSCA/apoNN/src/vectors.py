"""Contains Vectors and OccamVector classes, which are manipulated by the Fitter and Evaluator classes, and serve as an extension of numpy arrays targetted for our experiments."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from astropy.io import fits
from scipy.stats import median_absolute_deviation as mad
import warnings

def project(data,direction):
    """obtain linear projection of data along direction"""
    return np.dot(direction,data.T).squeeze()

def get_vars(data,directions):
    return np.var(project(data,directions),axis=1)




class Vector():
    def __init__(self, val, order=1,interaction_only=True):
        self._val = val
        if order>1:
            poly = PolynomialFeatures(order,interaction_only,include_bias=False)
            self._val = poly.fit_transform(self._val)
            
            
            
    def __getitem__(self,i):
        return Vector(self.val[i])
        

    def whitened(self,whitener):
        """method that takes a whitening PCA instance and returned a whitened vector"""
        return Vector(whitener.transform(self._val))
        
    @property
    def val(self):
        return self._val
    
    def centered(self,relative_to=None):
        """
        Shift vector to have zero mean.
        
        relative_to: vector
            Vector to use for centering. If relative_to is None then vector is centered using its own mean.
        """
        if relative_to is None:
            return Vector(self._val -np.mean(self._val,axis=0))
        else:
            return Vector(self._val -np.mean(relative_to._val,axis=0))
    
    @property
    def normalized(self):
        return Vector(self.centered()/np.max(np.abs(self.centered()),0))
    



class LatentVector(Vector):
    def __init__(self,  dataset, autoencoder, n_data = 100, order=1,interaction_only=True):
        self.autoencoder = autoencoder
        self.dataset = dataset
        val = np.array([self.get_z(idx) for idx in range(n_data)]).squeeze()
        Vector.__init__(self,val,order,interaction_only)

    def get_z(self,idx):
        _,z = self.autoencoder(torch.tensor(self.dataset[idx][0]).to(device).unsqueeze(0))
        return z.detach().cpu().numpy()
    
    
    def get_x(self,idx):
        return self.dataset[idx][0]
   
    def get_mask(self,idx):
        return self.dataset[idx][1]
    
    def get_x_pred(self,idx):
        x_pred,_ = self.autoencoder(torch.tensor(self.dataset[idx][0]).to(device).unsqueeze(0))
        return x_pred.squeeze().detach().cpu().numpy()
    
    def plot(self,idx,limits=[4000,4200]):
        plt.plot(self.get_x_pred(idx),label="pred")
        plt.plot(self.get_x(idx),label="real")
        plt.legend()
        plt.xlim(limits)
        
      
        
    
    
class OccamVector(LatentVector,Vector):
    def __init__(self, cluster_names, dataset=None, autoencoder=None, val=None, n_data = 100, order=1,interaction_only=True):
        if val is None:     
            LatentVector.__init__(self,dataset,autoencoder,n_data,order,interaction_only)
        else:
            Vector.__init__(self,val,order,interaction_only)
        self.cluster_names = cluster_names
        self.registry = self.make_registry(self.cluster_names)

        
    def __getitem__(self,i):
        warnings.warn("slicing OccamVectors only modifies the val and not the cluster_names/registry. Proceed with caution.")
        return OccamVector(self.cluster_names, val = self.val[i])        

    @staticmethod
    def make_registry(cluster_names):
        clusters = list(set(cluster_names))
        cluster_registry = {}
        for cluster in clusters:
            cluster_idxs = np.where(cluster_names==cluster)
            cluster_registry[cluster] = cluster_idxs[0]
        return cluster_registry

   
    
    @property
    def cluster_centered(self):
        z = np.zeros(self.val.shape)
        for cluster in self.registry:
            cluster_idxs = self.registry[cluster]
            z[cluster_idxs]=self.val[cluster_idxs]-self.val[cluster_idxs].mean(axis=0)
        return Vector(z)

    def centered(self,relative_to=None):
        """
        Shift vector to have zero mean.
        
        relative_to: vector
            Vector to use for centering. If relative_to is None then vector is centered using its own mean.
        """
        if relative_to is None:
            return OccamVector(self.cluster_names, val = self._val -np.mean(self._val,axis=0))
        else:
            return OccamVector(self.cluster_names, val = self._val -np.mean(relative_to._val,axis=0))
    

    def whitened(self,whitener):
        """method that takes a whitening PCA instance and returned a whitened vector"""
        return OccamVector(self.cluster_names, val = whitener.transform(self._val))

    def only(self,cluster_name):
        """Return an OccamVector containing only the clusters of interest.
        
        INPUTS
        ------
        cluster_name: string or list
            if string returns an object containing only that cluster. If list return an object containing spectra associatd to all clusters in list."""
        if isinstance(cluster_name,list):
            idxs_kept = []
            for cluster in cluster_name:
                idxs_kept += list( self.registry[cluster])
        else: 
            idxs_kept = self.registry[cluster_name]
        return OccamVector(self.cluster_names[idxs_kept],val=self.val[idxs_kept])
   

    def without(self,cluster_name):
        """return an OccamVector containing all the clusters except one cluster"""
        idxs_cluster = self.registry[cluster_name]
        idxs_kept = np.delete(np.arange(len(self.val)),idxs_cluster)
        return OccamVector(self.cluster_names[idxs_kept],val=self.val[idxs_kept])

    
    def get_orphan_idxs(self,cluster_names):
        """return index of elements within list only appearing once"""
        #repeated code. I should probably merge the two functions.
        clusters_to_exclude = []
        registry = self.make_registry(cluster_names)
        for cluster_name in registry:
            if len(registry[cluster_name])==1:
                clusters_to_exclude.append(cluster_name)
        repeated_idxs = []
        for cluster in clusters_to_exclude:
            repeated_idxs.extend(registry[cluster])
        return repeated_idxs


    def get_nonorphan_idxs(self,cluster_names):
        """return indexes of all nonorphan stars"""
        return np.delete(np.arange(len(cluster_names)),self.get_orphan_idxs(cluster_names))    

    def remove_orphans(self):
        clusters_to_exclude = []
        for cluster_name in self.registry:
            if len(self.registry[cluster_name])==1:
                clusters_to_exclude.append(cluster_name)
        filtered_self = self
        for cluster in clusters_to_exclude:
            filtered_self = filtered_self.without(cluster)
        
        return filtered_self
    
    

class AstroNNVector(Vector):
    def __init__(self,allStar,params):
        self.astroNN_hdul = fits.open("/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-astronn/apogee_astroNN-DR16-v0.fits")
        self.allStar = allStar
        self.params = params
        ids = self.get_astroNN_ids(self.allStar)
        cut_astroNN = self.astroNN_hdul[1].data[ids]
        self._val = self.generate_abundances(cut_astroNN,self.params)
        
        
    def get_astroNN_ids(self,allStar):
        desired_ids= []
        astroNN_ids = list(self.astroNN_hdul[1].data["Apogee_id"])
        for apogee_id in allStar["APOGEE_ID"]:
            desired_ids.append(astroNN_ids.index(apogee_id))
        return desired_ids
    
    
    def generate_abundances(self,astroNN,params):
        values = []
        for i,p in enumerate(params):
            fe_h = astroNN["FE_H"]
            if p in ["Teff","logg","Fe_H"]:
                values.append(astroNN[p])
            else:
                p_h = astroNN[params[i].split("_")[0]+"_H"]
                values.append(p_h-fe_h)

        return np.array(values).T
    
    def remove_nan_cols(self,):
        idxs_deleted = list(set(np.where(np.isnan(self.val))[0]))
        idxs_kept = np.delete(np.arange(len(self.val)),idxs_deleted)
        return Vector(val=self.val[idxs_kept])


class AllStarVector(Vector):
    def __init__(self,allStar,params):
        self.allStar = allStar
        self.params = params
        self.FELEM_species =  ["C",  "CI",  "N",  "O",  "Na",  "Mg",  "Al",  "Si",  "P",  "S",  "K","Ca", "Ti", "TiII", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ge", "Rb", "Ce","Nd","Yb"]
        self._val = self.make_y(self.allStar,self.params)

    def make_y(self,allStar,params):
        values = []
        for i,p in enumerate(params):
            specie, relative_to = p.split("_")
            if relative_to == "H":
                p_values = allStar["X_H"][:,self.FELEM_species.index(specie)]
            elif relative_to == "M":
                p_values = allStar["X_M"][:,self.FELEM_species.index(specie)]
            else:
                print(p)
                raise Exception("Some labels in params were not found")
            values.append(p_values)
            
        return np.array(values).T
    
   
  
    
    
class LinearTransformation():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    @property
    def val(self):
        return np.dot(self.y.centered().val.T,np.linalg.pinv(self.x.centered().val.T))
    
    def predict(self,vector:Vector):
        #need to return a Vector. So ncenteredeed to make this take the correct shape
        uncentered = np.dot(self.val,vector.centered().val.T).T
        centered = uncentered+np.mean(self.y.val,axis=0)
        return Vector(centered)                
        #return np.dot(self.val,vector.centered.T)
        
        
        
class NonLinearTransformation():
    """Used for non-linear regression of y from x. Is currently depreciated as I have removed pytorch support."""
    def __init__(self,x,y):
        self.x = x
        self.y = y
        structure = [x.centered.shape[1],256,256,y.centered.shape[1]]
        self.network  = Feedforward(structure,activation=nn.SELU()).to(device)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=0.00001)
        self.idx_loader = torch.utils.data.DataLoader(torch.arange(y.centered.shape[0]),batch_size=100)

        
    def fit(self,n_epochs = 20):
        for epoch in range(n_epochs):
            for idx in self.idx_loader:
                self.optimizer.zero_grad()
                x = torch.tensor(self.x.centered[idx]).to(device)
                y = torch.tensor(self.y.normalized[idx]).to(device)
                y_pred = self.network(x)
                err = self.loss(y_pred,y)
                err.backward()
                self.optimizer.step()
                print(f"err:{err}")
        return
    
    def predict(self,vector:Vector):
        #need to return a Vector. So ncenteredeed to make this take the correct shape
        x = torch.tensor(vector.centered).to(device)
        y_unscaled_pred = self.network(x).detach().cpu().numpy()
        y_pred = y_unscaled_pred*np.max(np.abs(self.y.centered),0)+np.mean(self.y.val,axis=0)
        return Vector(y_pred)                
        #return np.dot(self.val,vector.centered.T)

