"""Contains files for handling allStar APOGEE files and converting them into numpy arrays of observed spectra"""

import apogee.tools.read as apread
import apogee.tools.path as apogee_path
from apogee.tools import bitmask
from apogee.spec import continuum
import numpy as np

filtered_bits = [bitmask.apogee_pixmask_int('BADPIX'),
 bitmask.apogee_pixmask_int('CRPIX'),
 bitmask.apogee_pixmask_int('SATPIX'),
 bitmask.apogee_pixmask_int('UNFIXABLE'),
 bitmask.apogee_pixmask_int('BADDARK'),
 bitmask.apogee_pixmask_int('BADFLAT'),
 bitmask.apogee_pixmask_int('BADFLAT'),
 bitmask.apogee_pixmask_int('BADERR')]

class Dataset():
    def __init__(self,allStar=None,filtered_bits=filtered_bits,filling_dataset=None,threshold=0.05):
        """
        allStar: 
            an allStar FITS file containg those APOGEE observations which should be included in the dataset.
        threshold: float
            A cut-off error above which pixels should be considered masked
        """
        self.threshold = threshold
        self.bad_pixels_spec = []
        self.bad_pixels_err = []
        self.allStar = allStar
        self.filtered_bits = filtered_bits
        self.filling_dataset = filling_dataset
        self.spectra = self.spectra_from_allStar(allStar)
        self.errs = self.errs_from_allStar(allStar)
        self.masked_spectra = self.make_masked_spectra(self.spectra,self.errs,self.threshold)
        #self.mask = self.mask_from_allStar(allStar)
        
        
      
    def filter_mask(self,mask,filtered_bits):
        """takes a bit mask and returns an array with those elements to be included and excluded from the representation."""
        mask_arrays = np.array([bitmask.bit_set(bit,mask).astype(bool) for bit in filtered_bits])
        filtered_mask = np.sum(mask_arrays,axis=0)==0
        return filtered_mask
    
    def idx_to_prop(self,idx):
        """Get the Apogee information associated to an index entry in the Allstar file"""
        return self.allStar[idx]["APOGEE_ID"],self.allStar[idx]["FIELD"], self.allStar[idx]["TELESCOPE"]
        
    def spectra_from_idx(self,idx):
        """Get the ASPCAP continium normalized spectra corresponding to an allStar entry from it's index in allStar"""
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)[0]
    
    def mask_from_idx(self,idx):
        """Get the APSTAR mask associated to an AllStar entry from its index in allStar"""
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        return apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=3)[0][0]
    
    def errs_from_idx(self,idx):
        """Get the ASPCAP errs associated to an ASPCAP continuum normalized spectra from its index in allStar"""
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=2)[0]
        
    def spectra_from_allStar(self,allStar):
        """Converts an AllStar file into an array containing the ASPCAP continuum-normalized spectra. Any spectra incapable of being retrieved is added to a bad_pixels_spec list"""
        spectras = []
        for idx in range(len(allStar)):
            try:
                spectras.append(self.spectra_from_idx(idx).astype(np.float32))
            except:
                self.bad_pixels_spec.append(idx)
        return np.array(spectras)       
         

    
    def errs_from_allStar(self,allStar):
        """Converts an AllStar file into an array containing the ASPCAP continuum-normalized errors associated to spectra. Any spectra incapable of being retrieved is added to a bad_pixels_spec list"""

        errs = []
        for idx in range(len(allStar)):
            try:
                errs.append(self.errs_from_idx(idx).astype(np.float32))
            except:
                self.bad_pixels_err.append(idx)
        return np.array(errs)
 
    def mask_from_allStar(self,allStar):
        """Converts an AllStar file into an array containing the APSTAR masks continuum-normalized spectra."""

        mask  = [self.mask_from_idx(idx).astype(np.float32) for idx in range(len(allStar))]
        return mask

    def add_mask(self,new_mask):
        self.masked_spectra.mask = np.logical_or(self.masked_spectra.mask,new_mask)
    
    def make_masked_spectra(self,spectra,errs,threshold=0.05):
        """set to zero all pixels for which the error is predicted to be greater than some threshold."""
        mask = errs>threshold
        empty_bins = ~(spectra.any(axis=0)[None,:].repeat(len(spectra),axis=0))
        mask = np.logical_or(empty_bins ,mask)
        masked_spectra = np.copy(spectra)
        masked_spectra[mask]= 0
        masked_spectra = np.ma.masked_array(masked_spectra, mask=mask)
        return masked_spectra


class FitDataset(Dataset):
    def __init__(self,allStar):
        self.allStar = allStar
        self.spectra = self.spectra_from_allStar(allStar)

    def spectra_from_idx(self,idx):
        """Get the ASPCAP continium normalized spectra corresponding to an allStar entry from it's index in allStar"""
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        return apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=3)[0]
 

    
class ApVisitDataset(Dataset):
    """Dataset containing continuum-normalized visits. This code is a bit hacky so may fail on some edgecases"""
    def __init__(self,allStar=None,threshold=0.05):
        self.allStar = allStar
        self.threshold = threshold
        self.bad_pixels_spec = []
        self.bad_pixels_err = []
        self.spectra = self.spectra_from_allStar(allStar)
        self.errs = self.errs_from_allStar(allStar)
        #self.masked_spectra = self.make_masked_spectra(self.spectra,self.errs,self.threshold)


    def make_masked_spectra(self,spectra,errs,threshold=0.05):
        """set to zero all pixels for which the error is predicted to be greater than some threshold."""
        mask = errs>threshold
        masked_spectra = []
        for i,spec in enumerate(self.spectra):
            ma_spec_data = np.copy(np.array(spec))
            ma_spec_data = np.nan_to_num(ma_spec_data,posinf=0,neginf=0)
            ma = mask[None,i].repeat(ma_spec_data.shape[0],axis=0)
            ma_spec_data[ma] = 0
            ma_spec = np.ma.masked_array(ma_spec_data, mask=ma)
            masked_spectra.append(ma_spec)
        return masked_spectra
 

    def update_masked_spectra(self,errs,threshold=0.05):
        """Feed an error array to use in masked spectra"""
        self.masked_spectra = self.make_masked_spectra(self.spectra,errs,threshold)
        
    def visit_from_idx(self,idx,visit_idx):
        spec,spec_err = self.get_apstar_visit(idx,visit_idx) #if nvisit=1 --> spec dim is 8575 else spec dim is nvist+2
        cont_spec = self.continium_normalize_visit(spec,spec_err)
        return cont_spec
    
    def continium_normalize_visit(self,spec,spec_err):
        spec= np.reshape(spec,(1,len(spec)))
        spec_err= np.reshape(spec_err,(1,len(spec_err)))
        cont= continuum.fit(spec,spec_err,type='aspcap',niter=0)
        return spec[0]/cont[0]
    
    def get_apstar_visit(self,idx,visit_idx):
        apogee_id,loc,telescope = self.idx_to_prop(idx)
        if visit_idx ==0: #visit_idx==0 spectra have different shape needing accomodating
            spec = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)[0]
            spec_err = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=2)[0]
        else:
            spec = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)[0][visit_idx]
            spec_err = apread.apStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=2)[0][visit_idx]
        return spec,spec_err
    
        
    def spectra_from_allStar(self,allStar):
        spectras = []
        for idx in range(len(allStar)):
            n_visits = allStar["NVISITS"][idx]
            if n_visits>1:
                
                visits = []
                for visit_idx in range(2,n_visits+2):
                    visits.append(self.visit_from_idx(idx,visit_idx).astype(np.float32))
                spectras.append(visits)

            else:
                visits = []
                visits.append(self.visit_from_idx(idx,0).astype(np.float32))
                spectras.append(visits)

        return np.array(spectras)       
         

    

    
def interpolate(spectra, filling_dataset):
    """
    Takes a spectra and a dataset and fills the missing values in the spectra with those from the most similar spectra in the dataset
    ---------------------
    spectra: numpy.array
            a spectra with missing values set to zero which we wish to fill
       filling_dataset: numpy.array
            dataset of spectra we would like to use for interpolation 
    """
    print("new spectrum interpolated...")
    well_behaved_bins = np.sum(filling_dataset,axis=0)!=0 #we are happy to leave at zero these bins
    missing_values = spectra.mask
    similarity = np.sum((filling_dataset - spectra)**2,axis=1)
    similarity_argsort = list(similarity.argsort()) #1 because 0 is the spectra itself
    
    inpainted_spectra = np.copy(spectra)
    zeroes_exist=True
    while zeroes_exist:
        most_similar_idx = similarity_argsort.pop(0)
        inpainted_spectra[missing_values] = filling_dataset[most_similar_idx][missing_values] #while loop makes replacing with flagged ok
        missing_values = inpainted_spectra==0
        if (missing_values[well_behaved_bins]==False).all(): #check whether some values are still zero. If none are break from loop
            zeroes_exist=False

    return inpainted_spectra  



def infill_masked_spectra(masked_dataset,masked_filling_dataset=None):
    infilled_dataset = [interpolate(spectra,masked_filling_dataset) for spectra in masked_dataset]     
    return np.array(infilled_dataset)
            
        

