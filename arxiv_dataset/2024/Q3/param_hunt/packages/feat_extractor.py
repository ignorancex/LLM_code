# this package extract features from a data file or the posterior from a posterior file
import numpy as np
import pandas as pd


# limit the observable to a range xmin, xmax    
def rangify(x, xmin, xmax):
    y = np.copy(x)
    y[x<xmin] = xmin
    y[x>xmax] = xmax
    return y

# just apply normal smearing to variables
# can be adapted to different uncertainty paradigms     
def smear(feat, err):
    x = np.random.normal(feat , np.abs(err))
    return x

# extract features from ext_file
# sig_ch, sig_E, sig_theta, sig_phi are the calorimeter hit resolution, energy relative uncertainty, polar and azimuthal angles resolution
# Eres is the minimum energy, lx, ly are the calo size and disply is the calo displacement
class feature_extract():
    def __init__(self, ext_file, sig_ch, sig_E, sig_theta, sig_phi, Eres, x_min, x_max, y_min, y_max):
        self.ext_file=ext_file
        self.df = pd.read_csv(self.ext_file)
        self.sig_ch = sig_ch
        self.sig_E = sig_E
        self.sig_theta = sig_theta
        self.sig_phi = sig_phi
        self.Eres = Eres
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
    def extract_model(self, iOBS):
        suffix = '_'+str(iOBS)
        malp=self.df['m_alp'+suffix].to_numpy()
        tmalp=self.df['ctau_alp'+suffix].to_numpy()/malp
        return malp, tmalp
        
    
    def extract_llo(self, iOBS):
        
        # this is for the naming convention I use for the three saved events per sampple
        suffix = '_'+str(iOBS)

        # read the variables
        self.p_gam_1 = np.vstack((self.df['E1'+suffix], self.df['px1'+suffix], self.df['py1'+suffix], self.df['pz1'+suffix])).T
        self.p_gam_2 = np.vstack((self.df['E2'+suffix], self.df['px2'+suffix], self.df['py2'+suffix], self.df['pz2'+suffix])).T

        self.calo_hit_1 = np.vstack((self.df['chx1'+suffix], self.df['chy1'+suffix], self.df['chz1'+suffix])).T
        self.calo_hit_2 = np.vstack((self.df['chx2'+suffix], self.df['chy2'+suffix], self.df['chz2'+suffix])).T
        
        self.theta1 = np.arccos(self.p_gam_1[:,3]/self.p_gam_1[:,0]) 
        self.theta2 = np.arccos(self.p_gam_2[:,3]/self.p_gam_2[:,0])
        
        self.phi1 = np.arctan2(self.p_gam_1[:,2], self.p_gam_1[:,1]) 
        self.phi2 = np.arctan2(self.p_gam_2[:,2], self.p_gam_2[:,1])
        
        # decay position is currently not used as input, but it can be used for debugging
        self.decay_pos  = np.vstack((self.df['Vx'+suffix], self.df['Vy'+suffix], self.df['Vz'+suffix] )).T
        
        # smear the variables, but force them in physical ranges
        
        # smear energy
        # energy forced to be betweeen Eres and an arbitrary upper bound (neither of these affect many events, at least for small energy uncertainty)
        self.p_gam_1[:,0], self.p_gam_2[:,0] = smear(self.p_gam_1[:,0], self.sig_E*self.p_gam_1[:,0]), smear(self.p_gam_2[:,0], self.sig_E*self.p_gam_2[:,0])
        self.p_gam_1[:,0], self.p_gam_2[:,0] = rangify(self.p_gam_1[:,0], self.Eres, 2.*self.p_gam_1[:,0].max()), rangify(self.p_gam_2[:,0], self.Eres, 2.*self.p_gam_2[:,0].max()) # min energy resolution Eres
        
        # smear calorimeter hits (z irrelevant as it is fixed)
        # smeared calorimeter hits should be within the calorimeter size (also this requirement affects very few events)
        self.calo_hit_1, self.calo_hit_2 = smear(self.calo_hit_1, self.sig_ch), smear(self.calo_hit_2, self.sig_ch)
        self.calo_hit_1[:,0], self.calo_hit_2[:,0] = rangify(self.calo_hit_1[:,0], self.x_min, self.x_max), rangify(self.calo_hit_2[:,0], self.x_min, self.x_max)
        self.calo_hit_1[:,1], self.calo_hit_2[:,1] = rangify(self.calo_hit_1[:,1], self.y_min, self.y_max), rangify(self.calo_hit_2[:,1], self.y_min, self.y_max)
        
        # smear angles
        # polar angle cannot be smaller than angle resolution or larger than pi/2, also implemented in order not to get negative theta
        # azimuthal angle in -pi, pi
        self.theta1, self.theta2 = rangify(smear(self.theta1, self.sig_theta), self.sig_theta, np.math.pi/2), rangify(smear(self.theta2, self.sig_theta), self.sig_theta, np.math.pi/2)
        self.phi1, self.phi2 = rangify(smear(self.phi1, self.sig_phi), -np.math.pi, np.math.pi), rangify(smear(self.phi2, self.sig_phi), -np.math.pi, np.math.pi)

        # we take the log of energies and theta since we span orders of magnitudes in their values
        # log of theta is safe since we have theta > angle resolution
        return np.vstack(( np.log(self.p_gam_1[:,0]), np.log(self.p_gam_2[:,0]), 
                           self.calo_hit_1[:, 0], self.calo_hit_1[:, 1], self.calo_hit_2[:, 0], self.calo_hit_2[:, 1],
                           np.log(self.theta1), np.log(self.theta2), self.phi1, self.phi2))

class post_extract():
    def __init__(self, ext_file):
        self.ext_file=ext_file
        self.df = pd.read_csv(self.ext_file)
        self.X= np.array(self.df)
        
    def extract_model(self, iobs):
        malp = self.X[:,0+202*(iobs)]
        tmalp=self.X[:,1+202*(iobs)]
        return malp, tmalp

    def extract_post(self, nobs): # extract the posteriors in the file for the first nobs events
        Xtemp = []
        for iobs in range(nobs):
            Xtemp.append(self.X[:,2+202*(iobs):202*(iobs+1)])
        return np.concatenate(Xtemp, axis=1)
