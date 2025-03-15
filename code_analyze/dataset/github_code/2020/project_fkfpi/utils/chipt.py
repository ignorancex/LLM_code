#!/usr/bin/env python3
import sys
import numpy as np
import functools # for LRU cache
lru_cache_size=400 #increase this if need be to store all bessel function evaluations in cache memory
import warnings # supress divide by zero warning in fakeData determination of terms
import scipy.special as spsp # Bessel functions
import gvar as gv

sys.path.append('py_chiron')
import chiron

@functools.lru_cache(maxsize=lru_cache_size)
def chironFF(aFloat):
    return chiron.FF(aFloat)

def FF(x):
    if isinstance(x, gv.GVar):
        f = chironFF(x.mean)
        stepSize = 1e-7# * chiron.FF(x.mean)
        dfdx = 0.5*(chiron.FF(x.mean+stepSize) - chiron.FF(x.mean-stepSize))/stepSize
        return gv.gvar_function(x, f, dfdx)
    else:
        return chiron.FF(x)

pi = np.pi

'''
lazily evaluated dictionary that will compute convenience observables
as they are needed by calling into the parent model's convenience
functions (with a prepended underscore)
'''
class ConvenienceDict(dict):

    def __init__(self, parent_model, x, p, *args, **kwargs):
        self.p_model = parent_model
        self.x = x
        self.p = p
        dict.__init__(self,*args, **kwargs)

    def __getitem__(self, key):
        if key not in self.keys():
            dict.__setitem__(self, key, getattr(FitModel, "_"+key)(self.p_model,self.x,self.p,self))
        return dict.__getitem__(self,key)


'''
This class defines the functions that go into the various fit models
'''
class FitModel:

    def __init__(self, _term_list, _fv, _FF):
        self.term_list       = _term_list
        self.fv              = _fv
        self.FF              = _FF
        self.required_params = self._get_used_params()

    def __call__(self, x, p):
        if len(self.term_list)==0: return 0. # convenience value for plotting purposes
        convenience_p = ConvenienceDict(self, x, p)
        return sum(getattr(FitModel, term)(self, x, p, convenience_p) for term in self.term_list)

    def get_required_parameters(self):
        return self.required_params[1]

    ''' this function self-reflects to find out
        which x, p and convenience_p (cp) are going to be required '''
    def _get_used_params(self):
        class FakeDict(dict):
            def __init__(self):
                self.param_list = set()
            def __getitem__(self, key):
                self.param_list.add(key)
                return 1.
        fake_x  = FakeDict()
        fake_p  = FakeDict()
        fake_cp = ConvenienceDict(self, fake_x, fake_p)
        # the regular function calls (which automatically recurse into the
        # convenience functions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dummy = sum(getattr(FitModel, term)(self,fake_x,fake_p,fake_cp)
                            for term in self.term_list)
            except:
                pass
        return fake_x.param_list, fake_p.param_list, set(fake_cp.keys())

    ''' define all the convenience functions; whichever attribute is accessed of
        `cP` in the physics functions above needs to have a corresponding
        function defined here with the name and an underscore prepended '''
    # xpt masses
    def _p2(self, x, p, cP):  return (p['mpi'] / p['Lchi_'+self.FF])**2
    def _k2(self, x, p, cP):  return (p['mk']  / p['Lchi_'+self.FF])**2
    def _e2(self, x, p, cP):  return 4./3.*cP['k2']-1./3.*cP['p2']
    def _k2p2(self, x, p, cP): return cP['k2'] - cP['p2']
    # logs
    def _lp(self, x, p, cP): return np.log(cP['p2'])
    def _lk(self, x, p, cP): return np.log(cP['k2'])
    def _le(self, x, p, cP): return np.log(cP['e2'])
    # convenience NLO functions - so we can evaluate without FV corrections
    # when we stick them in NNLO terms
    def _dFPnlo(self, x, p, cP):
        a_result  = -cP['p2']*cP['lp'] -0.5*cP['k2']*cP['lk']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['p2']
        a_result += p['L4'] * (4*pi)**2 * 4 * (cP['p2'] + 2*cP['k2'])
        return a_result
    def _dFKnlo(self, x, p, cP):
        a_result  = -3./8 * cP['p2']*cP['lp'] -3./4 *cP['k2']*cP['lk'] -3./8*cP['e2']*cP['le']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['k2']
        a_result += p['L4'] * (4*pi)**2 * 4 * (cP['p2'] + 2*cP['k2'])
        return a_result
    # eps_a**2
    def _a2(self, x, p, cP):  return (p['aw0'] / 2)**2
    # mixed action params
    def _ju2(self, x, p, cP): return (p['mju'] / p['Lchi_'+self.FF])**2
    def _ru2(self, x, p, cP): return (p['mru'] / p['Lchi_'+self.FF])**2
    def _js2(self, x, p, cP): return (p['mjs'] / p['Lchi_'+self.FF])**2
    def _rs2(self, x, p, cP): return (p['mrs'] / p['Lchi_'+self.FF])**2
    def _ss2(self, x, p, cP): return (p['mss'] / p['Lchi_'+self.FF])**2
    def _xx2(self, x, p, cP): return cP['e2'] + p['a2DI'] / p['Lchi_'+self.FF]**2
    def _dju2(self, x, p, cP): return p['a2DI'] / p['Lchi_'+self.FF]**2
    def _drs2(self, x, p, cP): return p['a2DI'] / p['Lchi_'+self.FF]**2

    # Finite Volume Corrections to Tadpole Integral
    @functools.lru_cache(maxsize=lru_cache_size)
    def k0k1k2(self, mL):
        cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24])
        n_mag = np.sqrt(np.arange(1,len(cn)+1,1))
        k1 = np.sum(cn * spsp.kn(1,n_mag * mL) / mL / n_mag)
        k2 = np.sum(cn * spsp.kn(2,n_mag * mL) )
        k0 = np.sum(cn * spsp.kn(0,n_mag * mL) )
        return k0,k1,k2

    # Tadpole Integrals
    def _I(self, eps_sq, mL):
        return eps_sq*np.log(eps_sq) + (4.*eps_sq*self.k0k1k2(mL)[1] if self.fv else 0.)
    def _Ip(self, x, p, cP): return self._I(cP['p2'], (x['mpiL'] if self.fv else None))
    def _Ik(self, x, p, cP): return self._I(cP['k2'], (x['mkL']  if self.fv else None))
    def _Ie(self, x, p, cP): return self._I(cP['e2'], (x['meL']  if self.fv else None))
    # mock Taylor Expansion FV correction
    def _IT(self, mL):
        return (4*self.k0k1k2(mL)[1] if self.fv else 0.)
    def _ITp(self, x, p, cP): return p['t_fv']*self._IT( x['mpiL'] if self.fv else None)
    # mixed action
    def _Iju(self, x, p, cP): return self._I(cP['ju2'], (x['mjuL'] if self.fv else None))
    def _Iru(self, x, p, cP): return self._I(cP['ru2'], (x['mruL'] if self.fv else None))
    def _Ijs(self, x, p, cP): return self._I(cP['js2'], (x['mjsL'] if self.fv else None))
    def _Irs(self, x, p, cP): return self._I(cP['rs2'], (x['mrsL'] if self.fv else None))
    def _Iss(self, x, p, cP): return self._I(cP['ss2'], (x['mssL'] if self.fv else None))
    def _Ixx(self, x, p, cP): return self._I(cP['xx2'], (x['mxL']  if self.fv else None))

    def _dI(self, eps_sq, mL):
        return 1 + np.log(eps_sq) + (2*self.k0k1k2(mL)[1] -self.k0k1k2(mL)[0] -self.k0k1k2(mL)[2] if self.fv else 0.)
    def _dIp(self, x, p, cP): return self._dI(cP['p2'], (x['mpiL'] if self.fv else None))
    def _dIss(self, x, p, cP): return self._dI(cP['ss2'], (x['mssL'] if self.fv else None))

    def _Kpx(self, x, p, cP): return (cP['Ixx'] - cP['Ip'])  / (cP['xx2'] - cP['p2'])
    def _Ksx(self, x, p, cP): return (cP['Ixx'] - cP['Iss']) / (cP['xx2'] - cP['ss2'])
    def _K21px(self, x, p, cP): return (cP['Kpx'] - cP['dIp'])  / (cP['xx2'] - cP['p2'])
    def _K21sx(self, x, p, cP): return (cP['Ksx'] - cP['dIss']) / (cP['xx2'] - cP['ss2'])
    def _K123psx(self, x, p, cP):
        a_result  = cP['Ip']  / (cP['p2'] - cP['ss2']) / (cP['p2'] - cP['xx2'])
        a_result += cP['Iss'] / (cP['ss2'] - cP['p2']) / (cP['ss2'] - cP['xx2'])
        a_result += cP['Ixx'] / (cP['xx2'] - cP['p2']) / (cP['xx2'] - cP['ss2'])
        return a_result


    ''' Define all the fit functions to be used in the analysis.  We describe them
        in pieces, which are assembled based upon the term_list, to form a given
        fit function.
    '''
    # Fit functions
    def xpt_nlo(self,x,p,cP):
        a_result  = 1.
        a_result +=  self.FK_xpt_nlo(x,p,cP)
        a_result += -self.Fpi_xpt_nlo(x,p,cP)
        return a_result

    def xpt_ratio_nlo(self,x,p,cP):
        num = 1 + self.FK_xpt_nlo(x,p,cP)  + p['L4'] * (4*pi)**2 *(4*cP['p2'] +8*cP['k2'])
        den = 1 + self.Fpi_xpt_nlo(x,p,cP) + p['L4'] * (4*pi)**2 *(4*cP['p2'] +8*cP['k2'])
        return num / den

    def Fpi_xpt_nlo(self,x,p,cP):
        a_result  = -cP['Ip']
        a_result += -0.5 * cP['Ik']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['p2']
        return a_result

    def FK_xpt_nlo(self,x,p,cP):
        a_result  = -3./8 * cP['Ip']
        a_result += -3./4 * cP['Ik']
        a_result += -3./8 * cP['Ie']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['k2']
        return a_result

    def ma_nlo(self,x,p,cP):
        a_result  = 1.
        a_result +=  self.FK_ma_nlo(x,p,cP)
        a_result += -self.Fpi_ma_nlo(x,p,cP)
        return a_result

    def ma_ratio_nlo(self,x,p,cP):
        num = 1 + self.FK_ma_nlo(x,p,cP)  + p['L4'] * (4*pi)**2 *(4*cP['p2'] +8*cP['k2'])
        den = 1 + self.Fpi_ma_nlo(x,p,cP) + p['L4'] * (4*pi)**2 *(4*cP['p2'] +8*cP['k2'])
        return num / den

    def Fpi_ma_nlo(self,x,p,cP):
        a_result  = -cP['Iju']
        a_result += -1./2 * cP['Iru']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['p2']
        return a_result

    def FK_ma_nlo(self,x,p,cP):
        a_result  = -1./2 * cP['Iju']
        a_result +=  1./8 * cP['Ip']
        a_result += -1./4 * cP['Iru']
        a_result += -1./2 * cP['Ijs']
        a_result += -1./4 * cP['Irs']
        a_result +=  1./4 * cP['Iss']
        a_result += -3./8 * cP['Ixx']
        a_result += p['L5'] * (4*pi)**2 * 4 * cP['k2']

        a_result += cP['dju2']              * -1./8  * cP['dIp']
        a_result += cP['dju2']              *  1./4  * cP['Kpx']
        a_result += cP['dju2']**2           * -1./24 * cP['K21px']
        a_result += cP['dju2'] * cP['drs2'] *  1./12 * cP['K21sx']
        a_result += cP['dju2'] * cP['drs2'] * -1./6  * cP['K123psx']
        a_result += cP['drs2']              *  1./4  * cP['Ksx']
        a_result += cP['drs2'] * cP['k2']   * -1./6  * cP['K21sx']
        a_result += cP['drs2'] * cP['p2']   *  1./6  * cP['K21sx']

        return a_result

    def taylor_nlo(self,x,p,cP):
        ''' in order to keep the LECs of the same order as XPT, we multiply
            by powers of (4pi)**2 just as in XPT
        '''
        return 1. + (4*pi)**2 * p['L5'] * cP['k2p2'] * (1 + p['t_fv'] * cP['ITp'])

    # NNLO terms
    def nnlo_ct(self, x, p, cP):
        a_result  = cP['k2p2'] * cP['k2'] * p['k_4']
        a_result += cP['k2p2'] * cP['p2'] * p['p_4']
        a_result += cP['k2p2'] * cP['a2'] * p['s_4']
        return a_result

    def nnlo_alphaS(self, x, p, cP):
        return cP['k2p2'] * cP['a2'] * x['alphaS'] * p['saS_4']

    def xpt_nnlo_logSq(self, x, p, cP):
        a_result  = cP['lp']*cP['lp'] * ( 11./24   * cP['p2']*cP['k2'] -131./192 * cP['p2']*cP['p2'])
        a_result += cP['lp']*cP['lk'] * (-41./96   * cP['p2']*cP['k2'] -3./32    * cP['p2']*cP['p2'])
        a_result += cP['lp']*cP['le'] * ( 13./24   * cP['p2']*cP['k2'] +59./96   * cP['p2']*cP['p2'])
        a_result += cP['lk']*cP['lk'] * ( 17./36   * cP['k2']*cP['k2'] +7./144   * cP['p2']*cP['k2'])
        a_result += cP['lk']*cP['le'] * (-163./144 * cP['k2']*cP['k2'] -67./288  * cP['p2']*cP['k2'] +3./32   * cP['p2']*cP['p2'])
        a_result += cP['le']*cP['le'] * ( 241./288 * cP['k2']*cP['k2'] -13./72   * cP['p2']*cP['k2'] -61./192 * cP['p2']*cP['p2'])
        a_result += cP['k2']**2 * FF(cP['p2'] / cP['k2'])

        return a_result

    def xpt_nnlo_log(self, x, p, cP):
        tk  =  8*(4*pi)**2 *p['L5'] *(8*p['L4'] +3*p['L5'] -16*p['L6'] -8*p['L8'])
        tk += -2*p['L1'] -p['L2'] -1./18*p['L3'] +4./3*p['L5'] -16*p['L7'] -8*p['L8']

        tp  =  8*(4*pi)**2 *p['L5'] *(4*p['L4'] +5*p['L5']  -8*p['L6'] -8*p['L8'])
        tp += -2*p['L1'] -p['L2'] -5./18*p['L3'] -4./3*p['L5'] +16*p['L7'] +8*p['L8']

        ct  = (4*pi)**2 * cP['k2p2'] * cP['k2'] * tk
        ct += (4*pi)**2 * cP['k2p2'] * cP['p2'] * tp


        C1  = -cP['p2']*cP['k2'] *(7./9     + (4*pi)**2 * 11./2 *p['L5'])
        C1 += -cP['p2']*cP['p2'] *(113./72  + (4*pi)**2 *(4*p['L1'] +10*p['L2'] +13./2*p['L3'] -21./2*p['L5']))

        C2  =  cP['p2']*cP['k2'] *(209./144 + (4*pi)**2 *3*p['L5'])
        C2 +=  cP['k2']*cP['k2'] *(53./96   + (4*pi)**2 *(4*p['L1'] +10*p['L2'] +5*p['L3'] -5*p['L5']))

        C3  =  cP['k2']*cP['k2'] *(13./18   + (4*pi)**2 *(8./3*p['L3'] -2./3 *p['L5'] -16*p['L7']  -8*p['L8']))
        C3 += -cP['p2']*cP['k2'] *(4./9     + (4*pi)**2 *(4./3*p['L3'] +25./6*p['L5'] -32*p['L7'] -16*p['L8']))
        C3 +=  cP['p2']*cP['p2'] *(19./288  + (4*pi)**2 *(1./6*p['L3'] +11./6*p['L5'] -16*p['L7']  -8*p['L8']))

        return ct + C1*cP['lp'] +C2*cP['lk'] +C3*cP['le']

    def xpt_nnlo_ratio(self, x, p, cP):
        # Note: we used predefined NLO terms without FV to be consistent at NNLO
        return cP['dFKnlo']*cP['dFPnlo'] - cP['dFPnlo']**2

    def xpt_nnlo_FF_PP(self, x, p, cP):
        # Note: we used predefined NLO terms without FV to be consistent at NNLO
        return -3./2*cP['k2p2'] * cP['dFPnlo']

    def xpt_nnlo_FF_PK(self, x, p, cP):
        # Note: we used predefined NLO terms without FV to be consistent at NNLO
        return -3./4*cP['k2p2'] *(cP['dFKnlo'] +cP['dFPnlo']) +(cP['dFKnlo'] -cP['dFPnlo'])**2

    def xpt_nnlo_FF_KK(self, x, p, cP):
        # Note: we used predefined NLO terms without FV to be consistent at NNLO
        return -3./2*cP['k2p2'] *cP['dFKnlo'] +2*(cP['dFKnlo'] -cP['dFPnlo'])**2

    # NNNLO terms
    def nnnlo_a4(self, x, p, cP):
        return cP['k2p2'] * cP['a2']**2 * p['s_6']

    def nnnlo_ct(self, x, p, cP):
        # a^4
        a_result  = cP['k2p2'] * cP['a2'] * cP['a2'] * p['s_6']
        # a^2
        a_result += cP['k2p2'] * cP['k2'] * cP['a2'] * p['sk_6']
        a_result += cP['k2p2'] * cP['p2'] * cP['a2'] * p['sp_6']
        # xpt
        a_result += cP['k2p2'] * cP['k2'] * cP['p2'] * p['kp_6']
        a_result += cP['k2p2']**2         * cP['k2'] * p['k_6']
        a_result += cP['k2p2']**2         * cP['p2'] * p['p_6']

        return a_result

    def ma_longform_nlo(self,x,p,cP):
        a_result  = 1.
        # L5 counter term
        a_result += p['L5'] * (4*pi)**2 * 4 * (cP['k2'] - cP['p2'])
        # - dFPi
        a_result += cP['Iju']
        a_result += 0.5 * cP['Iru']
        # + dFK
        a_result += -0.5  * cP['Iju']
        a_result += -1./4 * cP['Iru']
        a_result += -0.5  * cP['Ijs']
        a_result += -1./4 * cP['Irs']
        a_result += -1./8 * cP['dju2']
        a_result += cP['dju2']**2 / 24 / (cP['xx2'] - cP['p2'])
        a_result += cP['drs2'] * (cP['k2']-cP['p2']) / 6 / (cP['xx2']-cP['ss2'])
        a_result += -cP['dju2'] * cP['drs2'] / 12 / (cP['xx2'] - cP['ss2'])
        a_result += cP['lp']/24 * (3*cP['p2'] \
                - 3*cP['dju2']*(cP['xx2']+cP['p2'])/(cP['xx2']-cP['p2']) \
                + cP['dju2']**2 * cP['xx2']/(cP['xx2']-cP['p2'])**2\
                -4*cP['dju2']*cP['drs2']*cP['p2']/(cP['xx2']-cP['p2'])/(cP['ss2']-cP['p2'])\
                )
        a_result += -cP['xx2']/24 * np.log(cP['xx2'])*(9 \
                -6*cP['dju2']/(cP['xx2']-cP['p2']) \
                + cP['dju2']**2/(cP['xx2']-cP['p2'])**2\
                +cP['drs2']*(4*(cP['k2']-cP['p2'])+6*(cP['ss2']-cP['xx2']))/(cP['xx2']- cP['ss2'])**2 \
                -2*cP['dju2']*cP['drs2']*(2*cP['ss2']-cP['p2']-cP['xx2'])/(cP['xx2']-cP['ss2'])**2/(cP['xx2']-cP['p2'])\
                )
        a_result += np.log(cP['ss2'])/12 * (3*cP['ss2'] \
                +cP['drs2']*(3*cP['ss2']**2 + 2*(cP['k2']-cP['p2'])*cP['xx2'] -3*cP['ss2']*cP['xx2'])/(cP['xx2']-cP['ss2'])**2\
                -cP['dju2']*cP['drs2']*(2*cP['ss2']**2 - cP['xx2']*(cP['ss2']+cP['p2']))/(cP['xx2']-cP['ss2'])**2 / (cP['ss2']-cP['p2'])\
                )

        return a_result

def dFKFpi_iso(phys_point,FF):
    p   = phys_point['p']
    k2  = (p['mk']/p['Lchi_'+FF])**2
    k2p = (p['mk+']/p['Lchi_'+FF])**2
    k20 = (p['mk0']/p['Lchi_'+FF])**2
    p2  = (p['mpi']/p['Lchi_'+FF])**2
    e2  = 4./3*k2 - 1./3*p2

    result  = (k2p - k2) * 4 * (4*np.pi)**2 * p['L5']
    result += -1./4*( k2p*np.log(k2p) - k2*np.log(k2) )
    result +=  1./4 *(k20 -k2p) * (e2*np.log(e2) - p2*np.log(p2)) / (e2 - p2)

    return result

def dFKFpi_iso_2(phys_point,FF, FKFpi):
    p   = phys_point['p']
    k2  = (p['mk']/p['Lchi_'+FF])**2
    k2p = (p['mk+']/p['Lchi_'+FF])**2
    k20 = (p['mk0']/p['Lchi_'+FF])**2
    p2  = (p['mpi']/p['Lchi_'+FF])**2
    e2  = 4./3*k2 - 1./3*p2

    prefac = -1./6 * ( k20 - k2p ) / ( e2 - p2 )
    result = prefac * ( 4 *(FKFpi - 1) +p2 * np.log(k2/p2) -k2 +p2 )

    return result


def dFKFpi_vincenzo(phys_point,FF):
    p   = phys_point['p']
    k2  = (p['mk']/p['Lchi_'+FF])**2
    k2p = (p['mk+']/p['Lchi_'+FF])**2
    k20 = (p['mk0']/p['Lchi_'+FF])**2
    p2  = (p['mpi']/p['Lchi_'+FF])**2
    e2  = 4./3*k2 - 1./3*p2
    eps = np.sqrt(3)/4 * (k20-k2p)/(k2-p2)

    result  = -16*np.sqrt(3) * eps / 3 * p['L5'] * (4*np.pi)**2 * (k2 - p2)
    result += -np.sqrt(3)*eps/2*(p2*np.log(p2) -e2*np.log(e2) -2./3*(k2-p2)*(np.log(k2) +1))

    return result / 2

def dFKFpi_vincenzo_2(phys_point,FF, FKFpi):
    p   = phys_point['p']
    k2  = (p['mk']/p['Lchi_'+FF])**2
    k2p = (p['mk+']/p['Lchi_'+FF])**2
    k20 = (p['mk0']/p['Lchi_'+FF])**2
    p2  = (p['mpi']/p['Lchi_'+FF])**2
    e2  = 4./3*k2 - 1./3*p2
    eps = np.sqrt(3)/4 * (k20-k2p)/(k2-p2)

    result  = np.sqrt(3) * eps * (-4./3) * (FKFpi -1)
    result += np.sqrt(3) * eps * 1./3 * (k2 -p2 -p2*np.log(k2/p2))

    return result / 2
