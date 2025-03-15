import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
from scipy.interpolate import interp1d

class growth_SN(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # needed arguments in order to get sigma_8(z) up to z=2 with correct precision
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': '1.'})
        self.need_cosmo_arguments(data, {'z_max_pk': '2.'})

        # are there conflicting experiments?
        if 'bao_fs_boss_dr12' in data.experiments:
            raise io_mp.LikelihoodError('conflicting bao measurments')

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.fsig8 = np.array([], 'float64')
        self.sfsig8 = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.fsig8 = np.append(self.fsig8, float(this_line[1]))
                    self.sfsig8 = np.append(self.sfsig8, float(this_line[2]))

        # positions of the data      
        self.Wigglez = [13, 14, 15];
        self.SDSS = [19, 20, 21, 22];
        # AP effect corrections
        self.HdAz=[5905.2, 5905.2, 5902.17, 27919.8, 40636.6, 45463.8, 47665.2, 90926.2, 63409.3, 88415.1, 78751., 132588., 102712., 132420., 155232.,134060., 179999., 263053., 200977., 242503., 289340., 352504.];
        # read covariance matrices
        self.CijWig=np.loadtxt(os.path.join(self.data_directory, self.cov_Wig_file),unpack=True)
        self.CijSDSS=np.loadtxt(os.path.join(self.data_directory, self.cov_SDSS_file),unpack=True)
        self.Cijfs8=np.diagflat(np.power(self.sfsig8,2))
        self.Cijfs8[(self.Wigglez[0]-1):self.Wigglez[-1],(self.Wigglez[0]-1):self.Wigglez[-1]]=self.CijWig
        self.Cijfs8[(self.SDSS[0]-1):self.SDSS[-1],(self.SDSS[0]-1):self.SDSS[-1]]=self.CijSDSS

        # number of bins
        self.num_points = np.shape(self.z)[0]

        # Scale dependent growth; some params: wavenumber k in Mpc etc. The params for the numerical derivative are a bit jerky, adjust at your own peril.
        self.k0=0.1
        self.dz=0.01
        self.hstep=0.001
		#Make a mock array; this will contain the redshifts z\in[0,2,0.01] we need for the interpolation
        self.zed = np.arange(0.0, 2.0,self.dz)
        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):
 		#Make a mock array; this will contain f(k,z) at z\in[0,2]
        scale_growthf=np.arange(0.0, 2.0,self.dz)
 		#Make a mock array; this will contain P(k,z) at z\in[0,2]
        Pkz=np.arange(0.0,2.0,self.dz)
        # Get the P(k,z) from CLASS and interpolate 
        for zz in range(len(self.zed)):
			Pkz[zz]=cosmo.pk(self.k0,self.zed[zz])
        Pkint=interp1d(self.zed,Pkz,kind='cubic')
        # Calculate the growth via f=-0.5*(1+z)*D[Log[P(k,z),z] with a simple difference and interpolate
        for zz in range(len(self.zed)-1):
            # One point numerical derivative (forward difference)
	        scale_growthf[zz]=-0.5*(1+self.zed[zz])*(Pkint(self.zed[zz]+self.hstep)-Pkint(self.zed[zz]))/self.hstep/Pkint(self.zed[zz])
            # Two point numerical derivative (central difference) // not working for now
	        #scale_growthf[zz]=-0.5*(1+self.zed[zz])*(Pkint(self.zed[zz]+self.hstep)-Pkint(self.zed[zz]-self.hstep))/2.0/self.hstep/Pkint(self.zed[zz])        
        scale_growthfint=interp1d(self.zed,scale_growthf,kind='cubic')
        data_array = np.array([], 'float64')
        chi2 = 0.
        for i in range(self.num_points):
            dA_at_z = cosmo.angular_distance(self.z[i])
            H_at_z = cosmo.Hubble(self.z[i])*conts.c /1000.0
            # This is the scale independent growth fs8!
            #theo_fsig8 = cosmo.scale_independent_growth_factor_f(self.z[i])*cosmo.sigma(8./cosmo.h(),self.z[i])
            #
            # This is the scale dependent growth fs8
            theo_fsig8 = scale_growthfint(self.z[i])*cosmo.sigma(8./cosmo.h(),self.z[i])
            ratio_at_z= H_at_z*dA_at_z/self.HdAz[i]
            #print 'f(',self.z[i],') =',cosmo.scale_independent_growth_factor_f(self.z[i])
            #print 'f(',self.z[i],') =',scale_growthfint(self.z[i])
            #print 'sigma8(',self.z[i],') =',cosmo.sigma(8./cosmo.h(),self.z[i])
            #print '(H, dA,f*sig8,ratio) =',H_at_z,dA_at_z,theo_fsig8,ratio_at_z
            # calculate difference between the sampled point and observations
            fsig8_diff = (self.fsig8[i]-theo_fsig8/ratio_at_z)
            data_array = np.append(data_array, fsig8_diff)
        # compute chi squared
        inv_cov_data = np.linalg.inv(self.Cijfs8)
        chi2 = np.dot(np.dot(data_array,inv_cov_data),data_array)
		
        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
