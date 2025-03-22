"""Tools for cross-matching the occam catalogue with the APOGEE Allstar file. Returns a filtered allstar file containing only those stars which are part of occam and dataset cuts"""


from astropy.io import fits
import numpy as np
import apogee.tools.read as apread
import apogee.tools.path as apogee_path


def prepare_occam_allStar(occam_kept,allStar,excluded_apogee_id= ['2M19203303+3755558']):
    """
    Given a list of of occam_ids, generates a filtered APOGEE allStar containing only those OCCAM stars.
    Also filters out stars within allStar without spectra. 
    
    INPUTS
    ------
    occam_kept: array
        boolean mask of same size of occam dataset with True for used stars and False for discarded Stars
    allStar:
        allStar file
        
    OUTPUTS
    -------
    (1) filtered_allStar containing only open cluster stars
    (2) cluster_ids of every star in the filtered dataset
    """
    occam = Occam()
    filtered_occam_apogee_id = np.array(occam.apogee_id)[occam_kept]
    list_apogee_ids = list(allStar["Apogee_id"])
    apogee_idxs = []
    for idx in list(filtered_occam_apogee_id):
        try:
            apogee_idx = list_apogee_ids.index(idx)
            apogee_id,loc,telescope = allStar[apogee_idx]["APOGEE_ID"],allStar[apogee_idx]["FIELD"], allStar[apogee_idx]["TELESCOPE"]
            apread.aspcapStar(loc_id=str(loc),apogee_id=apogee_id,telescope=telescope,ext=1)
            if apogee_id not in excluded_apogee_id:
                apogee_idxs.append(apogee_idx)
            else:
                apogee_idxs.append(-1)
        except:
            apogee_idxs.append(-1)
    apogee_idxs = np.array(apogee_idxs)
    found_mask = apogee_idxs!=-1
    
    return allStar[apogee_idxs[found_mask]],np.array(occam.cluster_id)[occam_kept][found_mask]




class Occam():
    def __init__(self):  
        self.clusters  = self.load_cluster()
        self.members = self.load_members()
        self.cluster_id = list(self.members[1].data.field("CLUSTER"))
        self.apogee_id = list(self.members[1].data.field("APOGEE_ID"))
        self.rv_prob = self.members[1].data.field("rv_prob")
        self.feh_prob = self.members[1].data.field("feh_prob")
        self.pm_prob = self.members[1].data.field("PM_PROB")
        self.cg_prob = self.members[1].data.field("CG_prob")
        
    def load_cluster(self):
        clusters_path = "/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-occam/occam_cluster-DR16.fits"
        return fits.open(clusters_path)
    
    def load_members(self):
        members_path = "/share/splinter/ddm/modules/turbospectrum/spectra/dr16/apogee/vac/apogee-occam/occam_member-DR16.fits"
        return fits.open(members_path)


