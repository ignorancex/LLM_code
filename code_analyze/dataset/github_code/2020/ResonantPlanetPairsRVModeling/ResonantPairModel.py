import numpy as np
import radvel 
from collections import OrderedDict

def acr_forward_model(t,pars,acr_fn,time_base):
    vel = np.zeros(len(t))

    k1 = pars['k1'].value
    P1 = pars['per1'].value
    tc1 = pars['tc1'].value
   

    m2_by_m1  = pars['m2_by_m1'].value
    stcosw = pars['stcosw'].value
    stsinw = pars['stsinw'].value
    omega1 = np.arctan2(stsinw,stcosw)
    omega2 = omega1 + np.pi
    _t = stcosw**2 + stsinw**2
    e1,e2 = acr_fn(m2_by_m1,_t)

    tp1 = radvel.basis.timetrans_to_timeperi(tc1,P1,e1,omega1)
    orbel1 = np.array([P1,tp1,e1,omega1,k1])
    vel+=radvel.kepler.rv_drive(t,orbel1)
   
    j = pars['jres'].value
    k = pars['kres'].value
    angle_n = pars['angle_n'].value
    P2_by_P1 = j / (j-k)
    P2 = P2_by_P1 * P1
    k2 = m2_by_m1 * P2_by_P1**(-1/3) * np.sqrt((1-e1*e1)/(1-e2*e2) ) * k1
    M1 = 2 * np.pi * (time_base-tp1) / P1
    M2 = (1-k/j) * M1 + ((1-j+k+2*angle_n) / j) * np.pi
    tp2 = time_base - P2 * M2 / 2 / np.pi
    orbel2=np.array([P2,tp2,e2,omega2,k2])
    vel+=radvel.kepler.rv_drive(t,orbel2)

    return vel
    
class ACRModel(radvel.GeneralRVModel):
    def __init__(self,j,k,acr_curves_fn,time_base=0):
        self.j = j
        self.k = k
        self._acr_curves_fn = acr_curves_fn
        params = radvel.Parameters(1,basis ='per tc secosw sesinw k')
        params.pop('secosw1')
        params.pop('sesinw1')
        alpha = ((j-k)/j)**(2/3.)
        params['jres']=radvel.Parameter(j,vary=False)
        params['kres']=radvel.Parameter(k,vary=False)
        params['m2_by_m1']=radvel.Parameter(1,vary=True)
        params['stcosw']=radvel.Parameter(0,vary=True)
        params['stsinw']=radvel.Parameter(0,vary=True)
        params['angle_n']=radvel.Parameter(0,vary=False)
        super(ACRModel,self).__init__(params,acr_forward_model,time_base)
    def __call__(self,t,*args,**kwargs):
        return super(ACRModel,self).__call__(t,self._acr_curves_fn,self.time_base,*args,**kwargs)
    def get_synthparams(self):
        pars = self.params
        spars = radvel.Parameters(2,basis='per tp e w k')

        for key,val in pars.items():
            if key in spars:
                spars[key] = val

        m2_by_m1  = pars['m2_by_m1'].value
        stcosw = pars['stcosw'].value
        stsinw = pars['stsinw'].value
        omega1 = np.arctan2(stsinw,stcosw)
        omega2 = omega1 + np.pi
        _t = stcosw**2 + stsinw**2
        e1,e2 = self._acr_curves_fn(m2_by_m1,_t)
        tc1 = pars['tc1'].value
        P1  = pars['per1'].value
        tp1 = radvel.basis.timetrans_to_timeperi(tc1,P1,e1,omega1)

        j = pars['jres'].value
        k = pars['kres'].value
        angle_n = pars['angle_n'].value
        P2_by_P1 = j / (j-k)
        P2 = P2_by_P1 * P1
        k1 = pars['k1'].value
        k2 = m2_by_m1 * P2_by_P1**(-1/3) * np.sqrt((1-e1*e1)/(1-e2*e2) ) * k1
        M1 = 2 * np.pi * (self.time_base-tp1) / P1
        M2 = (1-k/j) * M1 + ((1-j+k+2*angle_n) / j) * np.pi
        tp2 = self.time_base-P2 * M2 / 2 / np.pi

        
        spars['e1'].value = e1
        spars['w1'].value = omega1
        spars['tp1'].value = tp1
        
        spars['per2'].value = P2
        spars['tp2'].value = tp2
        spars['e2'].value = e2
        spars['w2'].value = omega2
        spars['k2'].value = k2 
        return spars

class ACRModelPrior(radvel.prior.Prior):
    """
    Prior to keep things physical for the ACR model
    """
    def __init__(self,upperlim=0.99):
        self.upperlim = upperlim

    def __repr__(self):
        return "Eccentricities constrained to be < {}\n".format(self.upperlim)

    def __call__(self, params):
        stc = params['stcosw'].value
        sts = params['stsinw'].value
        _t = stc*stc + sts * sts
        if _t>1:
            return -np.inf
        return 0

class ACRModelPriorTransform():
    """
    A class providing a prior transformation function from 
    the unit hyper-cube to parameters of the ACR radvel model.
    This transformation is used by the dynesty nested sampler
    algorithms.
    
    Attributes
    ----------
    mindict : collections.OrderedDict
        Dictionary defining maximum bounds on model parameters.
        Used to set bounds on model parameters 'per1', 'k1', 
        plus all 'jitter' and 'gamma' parameters.
        
    maxdict : collections.OrderedDict
        Dictionary defining minimum bounds on model parameters.
        Used to set bounds on model parameters 'per1', 'k1', 
        plus all 'jitter' and 'gamma' parameters.
        
    Npars : int
        Dimension of parameter space.
        
    suffixes: list of str
        Suffixes of instrument-specific jitter and gamma terms.
     
    """
    def __init__(self, observations_df,like):
        """    
        Parameters
        ----------
        observations_df : pandas.core.frame.DataFrame
            A pandas DataFrame containing RV observation data.
            The dataframe must of 'instrument', 'uncertainty' and
            'velocity' columns. These are used when defining
            priors on jitter and gamma (RV offset) parameters.

        like : radvel.likelihood.CompositeLikelihood
            Radvel likelihood object.

        """
        self.maxdict=OrderedDict({col:np.inf for col in like.list_vary_params()})
        self.mindict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        self.meddict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        
        self.meddict['tc1'] = like.params['tc1'].value
        
        self.maxdict['per1'] = 1.25 * like.params['per1'].value
        self.mindict['per1'] = 0.75 * like.params['per1'].value
        
        self.maxdict['k1'] = 5 * like.params['k1'].value
        self.mindict['k1'] = 0.2 * like.params['k1'].value

        
        self.Npars = len(like.list_vary_params())
        self.suffixes = np.atleast_1d(like.suffixes)
        
        for inst in self.suffixes:
            inst_obs = observations_df.query('instrument==@inst')
            gammastr="gamma{}".format(inst)
            jitstr="jit{}".format(inst)
            self.mindict[jitstr] = 1e-3 * inst_obs['uncertainty'].median()
            self.maxdict[jitstr] = 30   * inst_obs['uncertainty'].median()
            self.mindict[gammastr] = inst_obs['velocity'].min()
            self.maxdict[gammastr] = inst_obs['velocity'].max()



    def __call__(self,u):
        """
        Transformation from unit hyper-cube to parameter space
        used by an ACR likelihood object.
        
        Arguments
        ---------
        u : array-like (Npars,)
            Uniform variables between 0 and 1
        Returns
        -------
        array-like
            Arguments for likelihood function. Argument order is:
                per1,tc1,k1,m2_by_m1,stcosw,stsinw,gamma_i,jit_i,...            
        """
        ###############
        ### planets ###
        ###############
        i=0
        per1 = self.mindict['per1'] * (1-u[i]) + self.maxdict['per1'] * u[i]  # within range p1min to p2max

        i+=1
        tc1 = self.meddict['tc1'] + (u[i]-0.5) * per1

        i+=1
        logk1_min = np.log(self.mindict['k1'])
        logk1_max = np.log(self.maxdict['k1'])
        logk1 = logk1_min * (1-u[i]) + logk1_max * u[i] # K log-uniform between Kmin and Kmax
        k1 = np.exp(logk1)

        i+=1
        log10_m2_by_m1 = 2 * (u[i] - 0.5)
        m2_by_m1 = 10**log10_m2_by_m1

        i+=1
        t = u[i] # t from 0 to 1

        i+=1
        w1 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

        stcosw,stsinw = np.sqrt(t) * np.array((np.cos(w1),np.sin(w1)))
        #########################
        ### instrument params ###
        #########################

        Ninst = len(np.atleast_1d(self.suffixes))
        gamma = np.zeros(Ninst)
        jit = np.zeros(Ninst)

        for k,sfx in enumerate(np.atleast_1d(self.suffixes)):

            i+=1
            gamma[k] =  self.mindict['gamma{}'.format(sfx)] * (1-u[i]) + self.maxdict['gamma{}'.format(sfx)] * u[i]

            i+=1
            logjit_min = np.log(self.mindict['jit{}'.format(sfx)])
            logjit_max = np.log(self.maxdict['jit{}'.format(sfx)])
            logjit = logjit_min * (1-u[i]) + logjit_max * u[i]
            jit[k] = np.exp(logjit)

        return np.append(
            np.array([per1,tc1,k1,m2_by_m1,stcosw,stsinw]),
            np.vstack((gamma,jit)).T.reshape(-1)
        )
class RadvelModelPriorTransform():
    """
    A class providing a prior transformation function from 
    the unit hyper-cube to parameters of a standard two-planet radvel model.
    This transformation is used by the dynesty nested sampler
    algorithms.
    
    Attributes
    ----------
    mindict : collections.OrderedDict
        Dictionary defining maximum bounds on model parameters.
        Used to set bounds on model parameters 'per1', 'k1', 
        'per2', 'k2' plus all 'jitter' and 'gamma' parameters.
        
    maxdict : collections.OrderedDict
        Dictionary defining minimum bounds on model parameters.
        Used to set bounds on model parameters 'peri', 'k1', 
        'per2', 'k2' plus all 'jitter' and 'gamma' parameters.
        
    Npars : int
        Dimension of parameter space.
        
    suffixes: list of str
        Suffixes of instrument-specific jitter and gamma terms.
     
    """
    def __init__(self, observations_df,like):
        """    
        Parameters
        ----------
        observations_df : pandas.core.frame.DataFrame
            A pandas DataFrame containing RV observation data.
            The dataframe must of 'instrument', 'uncertainty' and
            'velocity' columns. These are used when defining
            priors on jitter and gamma (RV offset) parameters.

        like : radvel.likelihood.CompositeLikelihood
            Radvel likelihood object.

        """
        self.maxdict=OrderedDict({col:np.inf for col in like.list_vary_params()})
        self.mindict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        self.meddict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        
        for i in range(1,3):
            tc = 'tc{}'.format(i)
            per='per{}'.format(i)
            k='k{}'.format(i)
            self.meddict[tc] = like.params[tc].value        
            self.maxdict[per] = 1.25 * like.params[per].value
            self.mindict[per] = 0.75 * like.params[per].value
            self.maxdict[k] = 5 * like.params[k].value
            self.mindict[k] = 0.2 * like.params[k].value
        
        self.Npars = len(like.list_vary_params())
        self.suffixes = np.atleast_1d(like.suffixes)
        
        for inst in self.suffixes:
            inst_obs = observations_df.query('instrument==@inst')
            gammastr="gamma{}".format(inst)
            jitstr="jit{}".format(inst)
            self.mindict[jitstr] = 1e-3 * inst_obs['uncertainty'].median()
            self.maxdict[jitstr] = 30   * inst_obs['uncertainty'].median()
            self.mindict[gammastr] = inst_obs['velocity'].min()
            self.maxdict[gammastr] = inst_obs['velocity'].max()



    def __call__(self,u):
        """
        Transformation from unit hyper-cube to parameter space
        used by an ACR likelihood object.
        
        Arguments
        ---------
        u : array-like (Npars,)
            Uniform variables between 0 and 1
        Returns
        -------
        array-like
            Arguments for likelihood function. Argument order is:
                per1,tc1,e1,w1,k1,per2,tc2,e2,w2,k2,gamma_i,jit_i,...            
        """
        ####################
        ### first planet ###
        ####################
        i=0
        per1 = self.mindict['per1'] * (1-u[i]) + self.maxdict['per1'] * u[i]  # within range p1min to p2max

        i+=1
        tc1 = self.meddict['tc1'] + (u[i]-0.5) * per1

        i+=1
        e1 = u[i] # eccentricity from 0 to 1

        i+=1
        w1 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

        i+=1
        logk1_min = np.log(self.mindict['k1'])
        logk1_max = np.log(self.maxdict['k1'])
        logk1 = logk1_min * (1-u[i]) + logk1_max * u[i] # K log-uniform between Kmin and Kmax
        k1 = np.exp(logk1)

        #####################
        ### second planet ###
        #####################
        i+=1
        per2 = self.mindict['per2']  * (1-u[i]) + self.maxdict['per2'] * u[i] # within range p2min to p2max

        i+=1
        tc2 = self.meddict['tc2'] + (u[i]-0.5) * per2

        i+=1
        e2 = u[i] # eccentricity from 0 to 1

        i+=1
        w2 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

        i+=1
        logk2_min = np.log(self.mindict['k2'])
        logk2_max = np.log(self.maxdict['k2'])
        logk2 = logk2_min * (1-u[i]) + logk2_max * u[i] # K log-uniform between Kmin and Kmax
        k2 = np.exp(logk2)

        #########################
        ### instrument params ###
        #########################

        Ninst = len(self.suffixes)
        gamma = np.zeros(Ninst)
        jit = np.zeros(Ninst)

        for k,sfx in enumerate(self.suffixes):

            i+=1
            gamma[k] =  self.mindict['gamma{}'.format(sfx)] * (1-u[i]) + self.maxdict['gamma{}'.format(sfx)] * u[i]

            i+=1
            logjit_min = np.log(self.mindict['jit{}'.format(sfx)])
            logjit_max = np.log(self.maxdict['jit{}'.format(sfx)])
            logjit = logjit_min * (1-u[i]) + logjit_max * u[i]
            jit[k] = np.exp(logjit)

        return np.append(
            np.array([per1,tc1,e1,w1,k1,per2,tc2,e2,w2,k2]),
            np.vstack((gamma,jit)).T.reshape(-1)
        )
class NbodyModelPriorTransform():
    """
    A class providing a prior transformation function from 
    the unit hyper-cube to parameters of an N-body radvel model.
    This transformation is used by the dynesty nested sampler
    algorithms.
    
    Attributes
    ----------
    mindict : collections.OrderedDict
        Dictionary defining maximum bounds on model parameters.
        Used to set bounds on model parameters 'per1', 'm1', 
        'per2', 'm2' plus all 'jitter' and 'gamma' parameters.
        
    maxdict : collections.OrderedDict
        Dictionary defining minimum bounds on model parameters.
        Used to set bounds on model parameters 'peri', 'm1', 
        'per2', 'm2' plus all 'jitter' and 'gamma' parameters.
        
    Npars : int
        Dimension of parameter space.
        
    suffixes: list of str
        Suffixes of instrument-specific jitter and gamma terms.
     
    """
    def __init__(self, observations_df,like):
        """    
        Parameters
        ----------
        observations_df : pandas.core.frame.DataFrame
            A pandas DataFrame containing RV observation data.
            The dataframe must of 'instrument', 'uncertainty' and
            'velocity' columns. These are used when defining
            priors on jitter and gamma (RV offset) parameters.

        like : radvel.likelihood.CompositeLikelihood
            Radvel likelihood object.

        """
        self.maxdict=OrderedDict({col:np.inf for col in like.list_vary_params()})
        self.mindict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        self.meddict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
        
        self.periodic_indicies = [1,3,6,8]
        for i in range(1,3):
            per='per{}'.format(i)
            m='m{}'.format(i)
            self.maxdict[per] = 1.25 * like.params[per].value
            self.mindict[per] = 0.75 * like.params[per].value
            self.maxdict[m] = 5 * like.params[m].value
            self.mindict[m] = 0.2 * like.params[m].value
        
        self.Npars = len(like.list_vary_params())
        self.suffixes = np.atleast_1d(like.suffixes)
        
        for inst in self.suffixes:
            inst_obs = observations_df.query('instrument==@inst')
            gammastr="gamma{}".format(inst)
            jitstr="jit{}".format(inst)
            self.mindict[jitstr] = 1e-3 * inst_obs['uncertainty'].median()
            self.maxdict[jitstr] = 30   * inst_obs['uncertainty'].median()
            self.mindict[gammastr] = inst_obs['velocity'].min()
            self.maxdict[gammastr] = inst_obs['velocity'].max()



    def __call__(self,u):
        """
        Transformation from unit hyper-cube to parameter space
        used by an ACR likelihood object.
        
        Arguments
        ---------
        u : array-like (Npars,)
            Uniform variables between 0 and 1
        Returns
        -------
        array-like
            Arguments for likelihood function. Argument order is:
                per1,tc1,e1,w1,k1,per2,tc2,e2,w2,k2,gamma_i,jit_i,...            
        """
        ####################
        ### first planet ###
        ####################
        i=0
        per1 = self.mindict['per1'] * (1-u[i]) + self.maxdict['per1'] * u[i]  # within range p1min to p2max

        i+=1
        M1 = np.mod(2 * np.pi * u[i], 2 * np.pi)

        i+=1
        e1 = u[i] # eccentricity from 0 to 1

        i+=1
        w1 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

        i+=1
        logm1_min = np.log(self.mindict['m1'])
        logm1_max = np.log(self.maxdict['m1'])
        logm1 = logm1_min * (1-u[i]) + logm1_max * u[i] # m log-uniform between mmin and mmax
        m1 = np.exp(logm1)

        #####################
        ### second planet ###
        #####################
        i+=1
        per2 = self.mindict['per2']  * (1-u[i]) + self.maxdict['per2'] * u[i] # within range p2min to p2max

        i+=1
        M2 = np.mod(2 * np.pi * u[i], 2 * np.pi)

        i+=1
        e2 = u[i] # eccentricity from 0 to 1

        i+=1
        w2 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

        i+=1
        logm2_min = np.log(self.mindict['m2'])
        logm2_max = np.log(self.maxdict['m2'])
        logm2 = logm2_min * (1-u[i]) + logm2_max * u[i] # K log-uniform between Kmin and Kmax
        m2 = np.exp(logm2)

        #########################
        ### instrument params ###
        #########################

        Ninst = len(self.suffixes)
        gamma = np.zeros(Ninst)
        jit = np.zeros(Ninst)

        for k,sfx in enumerate(self.suffixes):

            i+=1
            gamma[k] =  self.mindict['gamma{}'.format(sfx)] * (1-u[i]) + self.maxdict['gamma{}'.format(sfx)] * u[i]

            i+=1
            logjit_min = np.log(self.mindict['jit{}'.format(sfx)])
            logjit_max = np.log(self.maxdict['jit{}'.format(sfx)])
            logjit = logjit_min * (1-u[i]) + logjit_max * u[i]
            jit[k] = np.exp(logjit)

        secosw1 = np.sqrt(e1) * np.cos(w1)
        sesinw1 = np.sqrt(e1) * np.sin(w1)
        secosw2 = np.sqrt(e2) * np.cos(w2)
        sesinw2 = np.sqrt(e2) * np.sin(w2)
        return np.append(
            np.array([per1,secosw1,sesinw1,per2,secosw2,sesinw2,m1,M1,m2,M2]),
            np.vstack((gamma,jit)).T.reshape(-1)
        )
