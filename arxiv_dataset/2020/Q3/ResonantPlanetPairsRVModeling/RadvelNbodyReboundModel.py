import radvel
import rebound as rb
import numpy as np
from radvel.orbit import timeperi_to_timetrans,timetrans_to_timeperi


_AU_PER_DAY_IN_KM_PER_SECOND=1731.456836805555556

def get_planet_mass_from_mstar_K_P_e(mstar,K,P,e):
    """
    Compute planet mass from input stellar mass, semi-amplitude, period and eccentricity.

    Parameters
    ----------
    mstar : real
        Stellar mass in solar masses.
    K : real
        RV semi-amplitude in m/s.
    P : real
        Orbital period in days.
    e : real
        eccentricity

    Returns
    -------
    real
        Planet mass in solar masses
    """

    G = 0.0002959107392494323
    k = K / 1731456.836805555556
    Kcubed= k*k*k
    x = np.sqrt(1-e*e)
    x3 = x*x*x
    Y = x3 * P * Kcubed / (2*np.pi*G)
    return np.real(np.roots([-1,Y,Y*2*mstar,Y*mstar**2])[0])
def _nb_params_to_sim(params,time_base):
    sim = rb.Simulation()
    sim.units=('Msun','days','AU')
    sim.t = time_base
    mstar = params['Mstar'].value
    sim.add(m=mstar,hash='star')
    star = sim.particles['star']
    for i in range(1,params.num_planets + 1):
        mass = params['m{}'.format(i)].value
        per = params['per{}'.format(i)].value
        mean_anom = params['M{}'.format(i)].value
        secosw = params['secosw{}'.format(i)].value
        sesinw = params['sesinw{}'.format(i)].value
        e = secosw * secosw + sesinw * sesinw
        pomega =np.arctan2(sesinw,secosw)
        sim.add(primary = star, m=mass,P=per,e=e,M=mean_anom,pomega=pomega,inc=np.pi/2,hash=i)
    sim.move_to_com()
    sim.exit_min_distance = 3 * np.max([p.rhill for p in sim.particles[1:]])
    return sim

def _nb_rv_calc(time,params,time_base,meters_per_second =True):
    vel = np.zeros(len(time))
    sim = _nb_params_to_sim(params,time_base)
    star = sim.particles['star']
    for i,t in enumerate(time):
        try:
            sim.integrate(t)
        except rb.Encounter:
            return np.inf * np.ones(len(vel))
        vel[i] = -1 * star.vz
    vel *= _AU_PER_DAY_IN_KM_PER_SECOND
    if meters_per_second:
        vel *=1e3
    return vel
def get_planet_K_from_mstar_mplanet_P_e(mstar,mplanet,P,e):
    """
    Compute RV semi-amplitude in units [m/s] from input 
    stellar mass, mplanet, period and eccentricity.

    Parameters
    ----------
    mstar : real
        Stellar mass in solar masses.
    mplanet : real
        Planet mass in solar masses.
    P : real
        Orbital period in days.
    e : real
        eccentricity

    Returns
    -------
    real
        Planet mass in solar masses
    """

    G = 0.0002959107392494323
    
    k = (2 * np.pi * G / P)**(1/3) * mplanet / np.sqrt(1-e*e) / (mstar + mplanet)**(2/3)
    return 1731456.836805555556 * k
     
class NbodyModel(radvel.GeneralRVModel):
    def __init__(self,Nplanets,Mstar = 1.0, time_base=0,meters_per_second=True):
        params = radvel.Parameters(Nplanets,basis ='per tc secosw sesinw k')
        self.num_planets = params.num_planets
        self.meters_per_second = meters_per_second
       
        for i in range(1,Nplanets+1):
            params.pop("k{}".format(i))
            params.pop("tc{}".format(i))
            params.update(
             {"m{}".format(i):radvel.Parameter()}
            )
            params.update(
             {"M{}".format(i):radvel.Parameter()}
            )

        params.update(
            {"Mstar":radvel.Parameter(value=Mstar,vary=False)}
        )
        super(NbodyModel,self).__init__(params,_nb_rv_calc,time_base)

    def __call__(self,t,*args,**kwargs):
        kwargs = {'meters_per_second':self.meters_per_second}
        return super(NbodyModel,self).__call__(t,self.time_base,*args,**kwargs)
    def get_sim(self):
        return _nb_params_to_sim(self.params,self.time_base)

    def set_pars_from_synth_pars(self,synth_pars):
        mstar = self.params['Mstar'].value

        for i in range(1,self.num_planets+1):
            
            per = synth_pars['per{}'.format(i)].value
            K = synth_pars['k{}'.format(i)].value
            e = synth_pars['e{}'.format(i)].value
            w = synth_pars['w{}'.format(i)].value
            tp = synth_pars['tp{}'.format(i)].value
            if self.meters_per_second:
                mpl = get_planet_mass_from_mstar_K_P_e(mstar,K,per,e)
            else:
                mpl = get_planet_mass_from_mstar_K_P_e(mstar,1e3 * K,per,e)
            mean_anom = np.mod( 2*np.pi * (self.time_base - tp) / per,2*np.pi)

            
            self.params['m{}'.format(i)].value = mpl
            self.params['per{}'.format(i)].value = per
            self.params['secosw{}'.format(i)].value = np.sqrt(e) * np.cos(w)
            self.params['sesinw{}'.format(i)].value = np.sqrt(e) * np.sin(w)
            self.params['M{}'.format(i)].value = mean_anom
            

