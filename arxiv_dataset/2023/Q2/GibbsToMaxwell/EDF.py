import scipy.constants as const
from scipy.constants import physical_constants
import numpy as np
from sympy import symbols, diff,lambdify,log#,limit,factor
from solver_equations import solve_equations
from unitconvert import toMevfm,toMev4,m_e_MeV,m_muon_MeV,unitMeVfm#,m_p_MeV,m_n_MeV

ns=0.16
m_p_MeV=939.5
m_n_MeV=939.5
m_u_MeV=5
m_d_MeV=7
m_s_MeV=150

class Fermions(object):
    ns=0.16
    def __init__(self,args):
        self.name, self.m, self.g=args #m in unit MeV, g is degenracy in spin or isospin...
    def set_mass(self,mass):
        self.m=mass
    def chi(self,x):
        return self.g*(x*(1+x**2)**0.5*(2*x**2+1)-np.log(x+(1+x**2)**0.5))/(16*np.pi**2)
    def phi(self,x): #x=kF/m demensionless
        return self.g*(x*(1+x**2)**0.5*(2*x**2-3)+3*np.log(x+(1+x**2)**0.5))/(48*np.pi**2)
    def psi(self,x):
        return self.g*(4*x**5/(1+x**2)**0.5-3*x*(1+x**2)**0.5*(2*x**2-3)-9*np.log(x+(1+x**2)**0.5))/(72*np.pi**2)
    def eosDensity_from_x(self,x,x0=0):
        return toMevfm(self.m**4*(self.chi(x)-self.chi(x0)),'mev4')
    def eosPressure_from_x(self,x,x0=0):
        return toMevfm(self.m**4*(self.phi(x)-self.phi(x0)),'mev4')
    def eosN3d2Edn2_from_x(self,x,x0=0):
        return toMevfm(self.m**4*(self.psi(x)-self.psi(x0)),'mev4')
    def eosCs2(self,x):
        return (2*self.eosPressure_from_x(x)+self.eosN3d2Edn2_from_x(x))/(self.eosDensity_from_x(x)+self.eosPressure_from_x(x))
    def eosBaryonDensity_from_x(self,x,x0=0):
        return toMevfm(self.g*((x*self.m)**3-(x0*self.m)**3)/(6*np.pi**2),'mev4')
    def eosChempo_from_x(self,x):
        return self.m*(x**2+1)**0.5
    def eos_array_from_x(self,x,x0=0):
        return np.array([self.eosBaryonDensity_from_x(x,x0),self.eosDensity_from_x(x,x0),self.eosPressure_from_x(x,x0)])
    def eosX_from_chempo(self,chempo):
        return (np.where(chempo>self.m,chempo/self.m,1)**2-1)**0.5
    def eosX_from_n(self,n):
        return np.sign(n)*np.abs(toMev4(n,'mevfm')*(6*np.pi**2/(self.g*self.m**3)))**(1/3)

electron=Fermions(['electron',m_e_MeV,2])
muon=Fermions(['muon',m_muon_MeV,2])
def get_eos_array_lepton(chempo_lep,leptons=[electron,muon],ad_ratio=1):
    baryondensity_lepton = [lepton.eosBaryonDensity_from_x(ad_ratio*lepton.eosX_from_chempo(chempo_lep)) for lepton in leptons]
    density_lepton       =       [lepton.eosDensity_from_x(ad_ratio*lepton.eosX_from_chempo(chempo_lep)) for lepton in leptons]
    pressure_lepton      =      [lepton.eosPressure_from_x(ad_ratio*lepton.eosX_from_chempo(chempo_lep)) for lepton in leptons]
    return np.array([baryondensity_lepton,density_lepton,pressure_lepton])

def get_chempo_lepton(n_lep, m_e, m_muon):
#n_lep(fm^{-3}); m_e(MeV); m_muon(MeV)
#n_lep is assumed to be positive.
    a=np.sqrt(m_muon**2-m_e**2)
    b=3*np.pi**2*toMev4(n_lep,'mevfm')
    a2,a3,a4,a6,a8,a12=[a**2,a**3,a**4,a**6,a**8,a**12]
    b2,b4=[b**2,b**4]
    with_muon = b>a3 #
    c_cubic=-11*a12 + 14*a6*b2 - 2*b4 + 2*np.sqrt(np.where(with_muon,b-a3,0)*(a3 + b)*(a6 + b2)**3)
    c=np.sign(c_cubic)*np.abs(c_cubic)**(1/3)
    d=(5*a8 - 4*a2*b2 + c**2)/(3.*a2*c)
    f=(-6*a6 + b2)/(9.*a4)
    g=(2*b*(9 - b2/a6))/27.
    d_plus_f_sqr = np.sqrt(np.where(with_muon,d + f,1))
    #there are four roots, two complex, two real. only one real root.
    #there is a switch from fourth root to second root according to the sign changing at b=3a^3
    k_F_muon=(-(b/a2) - np.sign(b-3*a3)*3*d_plus_f_sqr + 3*np.sqrt(np.where(with_muon,-d + 2*f - np.sign(b-3*a3)*g/d_plus_f_sqr,0)))/6.
    k_F_muon=np.where(with_muon,k_F_muon,0)
    chempo_lep_squre=np.where(with_muon,k_F_muon**2+m_muon**2,b**(2/3)+m_e**2)
    k_F_e=np.sqrt(chempo_lep_squre-m_e**2)
    return np.sqrt(chempo_lep_squre),k_F_e,k_F_muon

class EDF_Bag(object):
    mp=m_p_MeV
    mn=m_n_MeV
    m_in_MeVfm2=unitMeVfm**(2/3)*0.5*(mp+mn)
    u_quark=Fermions(['u',m_u_MeV,6])
    d_quark=Fermions(['d',m_d_MeV,6])
    s_quark=Fermions(['s',m_s_MeV,6])
    quarks=[u_quark,d_quark,s_quark]
    def __init__(self,args):
        self.args=args
        self.B,self.a4,self.av=args #av=0.2 fm^2 or av=0.2*197.3 MeV*fm^3
        self.av=self.av*unitMeVfm**(-1/3)
    def eosX_from_n(self,x,nB):
        n_u=(1+x)*nB/self.a4
        n_ds=(2-x)*nB/self.a4
        chempo_ds,kF_d,kF_s=get_chempo_lepton(n_ds/3,self.d_quark.m,self.s_quark.m)
        x_u=(np.pi**2*toMev4(n_u,'mevfm'))**(1/3)/self.u_quark.m
        x_d=kF_d/self.d_quark.m
        x_s=kF_s/self.s_quark.m
        return x_u,x_d,x_s
    def eosDensity_from_n(self,x,nB):
        x_uds=self.eosX_from_n(x,nB)
        density=[quark_i.eosDensity_from_x(x_i) for x_i,quark_i in zip(x_uds, self.quarks)]
        return self.a4*np.array(density).sum(axis=0)+self.B+self.av*(3*nB/self.a4)**2/2
    def eosPressure_from_n(self,x,nB):
        x_uds=self.eosX_from_n(x,nB)
        pressure=[quark_i.eosPressure_from_x(x_i) for x_i,quark_i in zip(x_uds, self.quarks)]
        return self.a4*np.array(pressure).sum(axis=0)-self.B+self.av*(3*nB/self.a4)**2/2
    def eos_array_from_n(self,x,nB):
        x_uds=self.eosX_from_n(x,nB)
        density  = [quark_i.eosDensity_from_x(x_i) for x_i,quark_i in zip(x_uds, self.quarks)]
        pressure = [quark_i.eosPressure_from_x(x_i) for x_i,quark_i in zip(x_uds, self.quarks)]
        chempo=self.eosChempo_from_n(x,nB)
        chempo_lep=chempo[1]-chempo[0]
        return [nB,self.a4*np.array(density).sum(axis=0)+self.B+self.av*(3*nB/self.a4)**2/2,self.a4*np.array(pressure).sum(axis=0)-self.B+self.av*(3*nB/self.a4)**2/2,chempo_lep]
    def eosChempo_from_n(self,x,nB):
        n_u=(1+x)*nB/self.a4
        n_ds=(2-x)*nB/self.a4
        chempo_ds,kF_d,kF_s=get_chempo_lepton(n_ds/3,self.d_quark.m,self.s_quark.m)
        kF_u=(np.pi**2*toMev4(n_u,'mevfm'))**(1/3)
        chempo_u=(kF_u**2+self.u_quark.m**2)**0.5
        return [chempo_u+self.av*(n_u+n_ds),chempo_ds+self.av*(n_u+n_ds)]
    def equation_Ylep(self,Ylep_tanh,nB,extra_args):
        Ylep=0.5*(np.tanh(Ylep_tanh)+1)
        n_p=Ylep*nB
        chempo=self.eosChempo_from_n(Ylep,nB)
        chempo_lep=chempo[1]-chempo[0]
        k_F_e=np.where(chempo_lep>m_e_MeV,chempo_lep**2-m_e_MeV**2,0)**0.5
        n_e=toMevfm(k_F_e**3/(3*np.pi**2),'mev4')
        k_F_muon=np.where(chempo_lep>m_muon_MeV,chempo_lep**2-m_muon_MeV**2,0)**0.5
        n_muon=toMevfm(k_F_muon**3/(3*np.pi**2),'mev4')
        return n_p-n_e-n_muon
    def eosBeta_eq_from(self,nB):
        solve_success,solve_result=solve_equations(self.equation_Ylep,([-1],),[nB],vary_list=np.linspace(1.,1.,1),tol=1e-12)
        return 0.5*(np.tanh(solve_result[0])+1)
    
    
    
def free_no_potential(*anything):
    return 0
class Potential_single(object):
    ns=ns
    def __init__(self,args,sym_list,mean_potential_expr,additional_E_function=[free_no_potential]*3):
        self.args=args
        args_sym_list=sym_list[:-1]
        mean_potential_expr_subs=mean_potential_expr.subs(zip(args_sym_list,args))
        self.mean_potential_E=lambdify(sym_list[-1],mean_potential_expr_subs)
        self.mean_potential_dEdn=lambdify(sym_list[-1],diff(mean_potential_expr_subs,sym_list[-1]))
        self.mean_potential_d2Edn2=lambdify(sym_list[-1],diff(mean_potential_expr_subs,sym_list[-1],2))
        self.additional_e_function,self.additional_p_function,self.additional_n3d2Edn2_function=additional_E_function
        self.E=self.mean_potential_E(self.ns)+self.additional_e_function(self.ns)/self.ns
        self.L=3*self.ns*self.mean_potential_dEdn(self.ns)+3*self.additional_p_function(self.ns)/self.ns
        self.K=9*self.ns**2*self.mean_potential_d2Edn2(self.ns)+9*self.additional_n3d2Edn2_function(self.ns)/self.ns
    def eosDensity_from_n(self,n):
        return n*self.mean_potential_E(n)+self.additional_e_function(n)
    def eosPressure_from_n(self,n):
        return n**2*self.mean_potential_dEdn(n)+self.additional_p_function(n)
    def eosChempo_from_n(self,n):
        return (self.eosDensity_from_n(n)+self.eosPressure_from_n(n))/n
class Potential(object):
    mp=m_p_MeV
    mn=m_n_MeV
    def __init__(self,Potential_single_sym,Potential_single_pnm):
        self.mean_potential_e_sym=Potential_single_sym.eosDensity_from_n
        self.mean_potential_p_sym=Potential_single_sym.eosPressure_from_n
        self.mean_potential_e_pnm=Potential_single_pnm.eosDensity_from_n
        self.mean_potential_p_pnm=Potential_single_pnm.eosPressure_from_n
    def eosDensity_from_n(self,n_p,n_n):
        n=n_p+n_n
        beta=(n_p-n_n)/np.where(n>0,n,np.infty)
        return self.mean_potential_e_sym(n)+beta**2*(self.mean_potential_e_pnm(n)-self.mean_potential_e_sym(n))
    def eosChempo_from_n(self,n_p,n_n):
        n=n_p+n_n
        ni=np.array([n_p,n_n])
        part1=self.eosDensity_from_n(n_p,n_n)/n
        part2=(self.mean_potential_e_pnm(n)-self.mean_potential_e_sym(n))*4*(n-ni)*(2*ni-n)/n**3
        part3=(n**2*self.mean_potential_p_sym(n)+(self.mean_potential_p_pnm(n)-self.mean_potential_p_sym(n))*(2*ni-n)**2)/n**3
        return part1+part2+part3
    def eosPressure_from_n(self,n_p,n_n):
        chempo_p,chempo_n=self.eosChempo_from_n(n_p,n_n)
        return chempo_p*n_p+chempo_n*n_n-self.eosDensity_from_n(n_p,n_n)
class EDF_Potential(object):
    mp=m_p_MeV
    mn=m_n_MeV
    proton=Fermions(['proton',m_p_MeV,2])
    neutron=Fermions(['neutron',m_n_MeV,2])
    def __init__(self,EDF_Potential):
        self.EDF_Potential=EDF_Potential
    def eosDensity_from_n(self,n_p,n_n):
        xp=self.proton.eosX_from_n(n_p)
        xn=self.neutron.eosX_from_n(n_n)
        density_kinetic=self.proton.eosDensity_from_x(xp)+self.neutron.eosDensity_from_x(xn)
        density_potential=self.EDF_Potential.eosDensity_from_n(n_p,n_n)
        return density_kinetic+density_potential
    def eosChempo_from_n(self,n_p,n_n):
        xp=self.proton.eosX_from_n(n_p)
        xn=self.neutron.eosX_from_n(n_n)
        chempo_kinetic=np.array([self.proton.eosChempo_from_x(xp),self.neutron.eosChempo_from_x(xn)])
        chempo_potential=self.EDF_Potential.eosChempo_from_n(n_p,n_n)
        return chempo_kinetic+chempo_potential
    def eosPressure_from_n(self,n_p,n_n):
        chempo_p,chempo_n=self.eosChempo_from_n(n_p,n_n)
        return chempo_p*n_p+chempo_n*n_n-self.eosDensity_from_n(n_p,n_n)
    def eos_array_from_n(self,n_p,n_n):
        density=self.eosDensity_from_n(n_p,n_n)
        chempo_p,chempo_n=self.eosChempo_from_n(n_p,n_n)
        chempo_lep=chempo_n-chempo_p
        eos_array_hadronic=[n_p+n_n,density,chempo_p*n_p+chempo_n*n_n-self.eosDensity_from_n(n_p,n_n),chempo_lep]
        return eos_array_hadronic
    def equation_Ylep(self,Ylep_tanh,nB,extra_args):
        Ylep=0.5*(np.tanh(Ylep_tanh)+1)
        n_p=Ylep*nB
        chempo=self.eosChempo_from_n(n_p,nB-n_p)
        chempo_lep=chempo[1]-chempo[0]
        k_F_e=np.where(chempo_lep>m_e_MeV,chempo_lep**2-m_e_MeV**2,0)**0.5
        n_e=toMevfm(k_F_e**3/(3*np.pi**2),'mev4')
        k_F_muon=np.where(chempo_lep>m_muon_MeV,chempo_lep**2-m_muon_MeV**2,0)**0.5
        n_muon=toMevfm(k_F_muon**3/(3*np.pi**2),'mev4')
        return n_p-n_e-n_muon
    def eosBeta_eq_from(self,nB):
        solve_success,solve_result=solve_equations(self.equation_Ylep,([-1],),[nB],vary_list=np.linspace(1.,1.,1),tol=1e-12)
        return 0.5*(np.tanh(solve_result[0])+1)

def V_Lattimer(n_s,a,b,gamma,n):
    return a*(n/n_s)+b*(n/n_s)**gamma
sym_a, sym_b, sym_d, sym_gamma, sym_n= symbols('a b d gamma n', real=True)
syms_Lattimer=[sym_a, sym_b, sym_gamma, sym_n]
V_Lattimer_expr=V_Lattimer(ns, sym_a, sym_b, sym_gamma, sym_n)


def equation_transition_left(y,g,extra=None):
    skyrme_sly4,bag=extra
    nB,nB_,Ylep_tanh=y
    Ylep_=(3*np.tanh(Ylep_tanh)+1)/2
    Ylep_sign=np.sign(Ylep_)
    Ylep_=np.abs(Ylep_)
    Ylep=skyrme_sly4.eosBeta_eq_from(nB)
    chempo_pn=skyrme_sly4.eosChempo_from_n(Ylep*nB,(1-Ylep)*nB)
    chempo_tmp=bag.eosChempo_from_n(Ylep_sign*Ylep_,nB_)
    chempo_uud=2*chempo_tmp[0]+chempo_tmp[1]
    chempo_udd=chempo_tmp[0]+2*chempo_tmp[1]
    chempo_lep_hadronic=get_chempo_lepton(Ylep*nB,   m_e_MeV, m_muon_MeV)[0]
    chempo_lep_quark   =get_chempo_lepton(Ylep_*nB_, m_e_MeV, m_muon_MeV)[0]
    eos_array_lepton_hadronic=get_eos_array_lepton(chempo_lep_hadronic).sum(axis=1)
    eos_array_lepton_quark   =get_eos_array_lepton(chempo_lep_quark).sum(axis=1)
    eos_array_hadronic =skyrme_sly4.eos_array_from_n(Ylep*nB,(1-Ylep)*nB)
    eos_array_quark    =bag.eos_array_from_n(Ylep_sign*Ylep_,nB_)
    return [chempo_pn[0]-chempo_uud+g*(chempo_lep_hadronic-Ylep_sign*chempo_lep_quark),chempo_pn[1]-chempo_udd,eos_array_hadronic[2]-eos_array_quark[2]+g*(eos_array_lepton_hadronic[2]-eos_array_lepton_quark[2])]#,eos_array_hadronic[2]/eos_array_quark[2]-1]

def equation_transition_right(y,g,extra=None):
    skyrme_sly4,bag=extra
    nB,nB_,Yleptanh=y
    Ylep=(np.tanh(Yleptanh)+1)/2
    Ylep_=bag.eosBeta_eq_from(nB_)
    Ylep_sign=np.sign(Ylep_)
    Ylep_=np.abs(Ylep_)
    chempo_pn=skyrme_sly4.eosChempo_from_n(Ylep*nB,(1-Ylep)*nB)
    chempo_tmp=bag.eosChempo_from_n(Ylep_sign*Ylep_,nB_)
    chempo_uud=2*chempo_tmp[0]+chempo_tmp[1]
    chempo_udd=chempo_tmp[0]+2*chempo_tmp[1]
    chempo_lep_hadronic=get_chempo_lepton(Ylep*nB,   m_e_MeV, m_muon_MeV)[0]
    chempo_lep_quark   =get_chempo_lepton(Ylep_*nB_, m_e_MeV, m_muon_MeV)[0]
    eos_array_lepton_hadronic=get_eos_array_lepton(chempo_lep_hadronic).sum(axis=1)
    eos_array_lepton_quark   =get_eos_array_lepton(chempo_lep_quark).sum(axis=1)
    eos_array_hadronic =skyrme_sly4.eos_array_from_n(Ylep*nB,(1-Ylep)*nB)
    eos_array_quark    =bag.eos_array_from_n(Ylep_sign*Ylep_,nB_)
    return [chempo_pn[0]-chempo_uud+g*(chempo_lep_hadronic-Ylep_sign*chempo_lep_quark),chempo_pn[1]-chempo_udd,eos_array_hadronic[2]-eos_array_quark[2]+g*(eos_array_lepton_hadronic[2]-eos_array_lepton_quark[2])]#,eos_array_hadronic[2]/eos_array_quark[2]-1]

def equation_transition(y,nB,extra=None):
    g,skyrme_sly4,bag=extra
    Yleptanh,nB_,Ylep_tanh=y
    Ylep=(np.tanh(Yleptanh)+1)/2
    Ylep_=(3*np.tanh(Ylep_tanh)+1)/2
    Ylep_sign=np.sign(Ylep_)
    Ylep_=np.abs(Ylep_)
    chempo_pn=skyrme_sly4.eosChempo_from_n(Ylep*nB,(1-Ylep)*nB)
    chempo_tmp=bag.eosChempo_from_n(Ylep_sign*Ylep_,nB_)
    chempo_uud=2*chempo_tmp[0]+chempo_tmp[1]
    chempo_udd=chempo_tmp[0]+2*chempo_tmp[1]
    chempo_lep_hadronic=get_chempo_lepton(Ylep*nB,   m_e_MeV, m_muon_MeV)[0]
    chempo_lep_quark   =get_chempo_lepton(Ylep_*nB_, m_e_MeV, m_muon_MeV)[0]
    eos_array_lepton_hadronic=get_eos_array_lepton(chempo_lep_hadronic).sum(axis=1)
    eos_array_lepton_quark   =get_eos_array_lepton(chempo_lep_quark).sum(axis=1)
    eos_array_hadronic =skyrme_sly4.eos_array_from_n(Ylep*nB,(1-Ylep)*nB)
    eos_array_quark    =bag.eos_array_from_n(Ylep_sign*Ylep_,nB_)
    return [chempo_pn[0]-chempo_uud+g*(chempo_lep_hadronic-Ylep_sign*chempo_lep_quark),chempo_pn[1]-chempo_udd,eos_array_hadronic[2]-eos_array_quark[2]+g*(eos_array_lepton_hadronic[2]-eos_array_lepton_quark[2])]#,eos_array_hadronic[2]/eos_array_quark[2]-1]

def sol_to_all(g,nB,sol_g06,skyrme_sly4,bag):
    Yleptanh,nB_,Ylep_tanh=sol_g06.transpose()
    Ylep=(np.tanh(Yleptanh)+1)/2
    Ylep_=(3*np.tanh(Ylep_tanh)+1)/2
    Ylep_sign=np.sign(Ylep_)
    Ylep_=np.abs(Ylep_)
    chempo_pn=np.array([skyrme_sly4.eosChempo_from_n(Ylep_i*nB_i,(1-Ylep_i)*nB_i) for Ylep_i,nB_i in zip(Ylep,nB)])
    chempo_tmp=np.array([bag.eosChempo_from_n(Ylep__i,nB__i)  for Ylep__i,nB__i in zip(Ylep_,nB_)])
    chempo_uud=2*chempo_tmp[:,0]+chempo_tmp[:,1]
    chempo_udd=chempo_tmp[:,0]+2*chempo_tmp[:,1]
    chempo_lep_hadronic=np.array([get_chempo_lepton(Ylep_i*nB_i,   m_e_MeV, m_muon_MeV)[0] for Ylep_i,nB_i in zip(Ylep,nB)])
    chempo_lep_quark   =np.array([get_chempo_lepton(Ylep__i*nB__i, m_e_MeV, m_muon_MeV)[0] for Ylep__i,nB__i in zip(Ylep_,nB_)])
    chempo_lep_ghost=(chempo_pn[:,1]-chempo_pn[:,0]-chempo_lep_hadronic)/np.where(g<1,1-g,np.infty)+chempo_lep_hadronic

    
    eos_array_emu_ghost   =np.array([get_eos_array_lepton(chempo_lep_ghost_i)    for chempo_lep_ghost_i    in chempo_lep_ghost]   )
    eos_array_emu_hadronic=np.array([get_eos_array_lepton(chempo_lep_hadronic_i) for chempo_lep_hadronic_i in chempo_lep_hadronic])
    eos_array_emu_quark   =np.array([get_eos_array_lepton(chempo_lep_quark_i)    for chempo_lep_quark_i    in chempo_lep_quark]   )
    
    eos_array_lepton_ghost   =eos_array_emu_ghost.sum(axis=2)
    eos_array_lepton_hadronic=eos_array_emu_hadronic.sum(axis=2)
    eos_array_lepton_quark   =eos_array_emu_quark.sum(axis=2)
    #eos_array_lepton_quark   =Ylep_sign[:,np.newaxis]*eos_array_lepton_quark

    eos_array_hadronic=np.array([skyrme_sly4.eos_array_from_n(Ylep_i*nB_i,(1-Ylep_i)*nB_i) for Ylep_i,nB_i in zip(Ylep,nB)])[:,:3]
    eos_array_quark   =np.array([bag.eos_array_from_n(Ylep__i,nB__i)  for Ylep__i,nB__i in zip(Ylep_,nB_)])[:,:3]

    
    #chargedensity_hadronic=eos_array_hadronic[:,0]*Ylep           - g*eos_array_lepton_hadronic[:,0]       -(1-g)*eos_array_lepton_ghost[:,0]
    #chargedensity_quark   =eos_array_quark[:,0]*(Ylep_sign*Ylep_) - g*Ylep_sign*eos_array_lepton_quark[:,0]-(1-g)*eos_array_lepton_ghost[:,0]
    #quark_ratio=chargedensity_hadronic/(chargedensity_hadronic-chargedensity_quark)
    if(g==1):
        f=np.linspace(1,0,len(nB))
    else:
        f=(eos_array_lepton_ghost[:,0]-eos_array_quark[:,0]*(Ylep_sign*Ylep_))/(eos_array_hadronic[:,0]*Ylep-eos_array_quark[:,0]*(Ylep_sign*Ylep_))
    
    eos_array_gibbs=f[:,np.newaxis]*eos_array_hadronic+(1-f[:,np.newaxis])*eos_array_quark
    eos_array_gibbs_with_lepton=np.copy(eos_array_gibbs)
    eos_array_gibbs_with_lepton[:,1]+=f * (g*eos_array_lepton_hadronic[:,1]+(1-g)*eos_array_lepton_ghost[:,1])+(1-f)*(g*eos_array_lepton_quark[:,1]+(1-g)*eos_array_lepton_ghost[:,1])
    eos_array_gibbs_with_lepton[:,2]+=f * (g*eos_array_lepton_hadronic[:,2]+(1-g)*eos_array_lepton_ghost[:,2])+(1-f)*(g*eos_array_lepton_quark[:,2]+(1-g)*eos_array_lepton_ghost[:,2])
    
    chempo_all=np.concatenate((chempo_pn.transpose(),chempo_tmp.transpose(),[chempo_lep_hadronic,Ylep_sign*chempo_lep_quark,chempo_lep_ghost]),axis=0)

    p_n_u_ds_eN_muN_eQ_muN_eG_muG =[]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[Ylep*nB]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[(1-Ylep)*nB]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[(1+Ylep_sign*Ylep_)*nB_]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[(2-Ylep_sign*Ylep_)*nB_]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[eos_array_emu_hadronic[:,0,0],eos_array_emu_hadronic[:,0,1]]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[Ylep_sign*eos_array_emu_quark[:,0,0],Ylep_sign*eos_array_emu_quark[:,0,1]]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG+=[eos_array_emu_ghost[:,0,0],eos_array_emu_ghost[:,0,1]]
    p_n_u_ds_eN_muN_eQ_muN_eG_muG =np.array(p_n_u_ds_eN_muN_eQ_muN_eG_muG)
    
    #chargedensity_hadronic_=p_n_u_ds_eN_muN_eQ_muN_eG_muG[0]-g*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[4]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[5])-(1-g)*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[8]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[9])
    #chargedensity_quark_   =p_n_u_ds_eN_muN_eQ_muN_eG_muG[2]*2/3-p_n_u_ds_eN_muN_eQ_muN_eG_muG[3]/3-g*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[6]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[7])-(1-g)*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[8]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[9])
    #print((f*chargedensity_hadronic_+(1-f)*chargedensity_quark_))
    #print((f*p_n_u_ds_eN_muN_eQ_muN_eG_muG[0]+(1-f)*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[2]*2/3-p_n_u_ds_eN_muN_eQ_muN_eG_muG[3]/3))/(p_n_u_ds_eN_muN_eQ_muN_eG_muG[8]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[9]))
    #print((f*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[4]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[5])+(1-f)*(p_n_u_ds_eN_muN_eQ_muN_eG_muG[6]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[7]))/(p_n_u_ds_eN_muN_eQ_muN_eG_muG[8]+p_n_u_ds_eN_muN_eQ_muN_eG_muG[9]))

    #print(chargedensity_hadronic_/chargedensity_hadronic)
    #print(chargedensity_quark_/chargedensity_quark)
    #print(chargedensity_quark)
    #p_n_u_ds_eN_muN_eQ_muN_eG_muG*=np.array([f,f,1-f,1-f,f*g,f*g,(1-f)*g,(1-f)*g,1-g+0*f,1-g+0*f])
    #return eos_array_gibbs_with_lepton,quark_ratio,f,np.array([nB,nB_,Ylep,Ylep_sign*Ylep_,eos_array_lepton_hadronic[:,0],Ylep_sign*eos_array_lepton_quark[:,0],eos_array_lepton_ghost[:,0]])
    #return np.concatenate((eos_array_gibbs_with_lepton.transpose(),np.array([f,nB,nB_,Ylep,Ylep_sign*Ylep_,eos_array_lepton_hadronic[:,0],Ylep_sign*eos_array_lepton_quark[:,0],eos_array_lepton_ghost[:,0]])),axis=0)
    return np.concatenate((np.array([0*f+g,f]),eos_array_gibbs_with_lepton.transpose(),chempo_all,p_n_u_ds_eN_muN_eQ_muN_eG_muG,np.array([nB,nB_,Ylep,Ylep_sign*Ylep_,eos_array_lepton_hadronic[:,0],Ylep_sign*eos_array_lepton_quark[:,0],eos_array_lepton_ghost[:,0]])),axis=0)

def eos_array_adiabatic(data_MIX,skyrme_sly4,bag,ad_ratio=1):
    g,f,nB,density,p,chempo_p,chempo_n,chempo_u,chempo_d,chempo_lep_N,chempo_lep_Q,chempo_lep_G,n_p,n_n,n_u,n_ds,n_eN,n_muN,n_eQ,n_muQ,n_eG,n_muG=data_MIX
    Ylep=n_p/(n_p+n_n)
    Ylep_=(2*n_u-n_ds)/(3*(n_u+n_ds))
    Ylep_sign=np.sign(Ylep_)
    Ylep_=np.abs(Ylep_)
    
    eos_array_hadronic=np.array([skyrme_sly4.eos_array_from_n(Ylep_i*nB_i,(1-Ylep_i)*nB_i) for Ylep_i,nB_i in zip(Ylep,ad_ratio*(n_p+n_n))])[:,:3]
    eos_array_quark   =np.array([bag.eos_array_from_n(Ylep__i,nB__i)  for Ylep__i,nB__i in zip(Ylep_,ad_ratio*(n_u+n_ds)/3)])[:,:3]
    eos_array_eN=electron.eos_array_from_x(electron.eosX_from_n(ad_ratio*n_eN))
    eos_array_eQ=electron.eos_array_from_x(electron.eosX_from_n(np.abs(ad_ratio*n_eQ)))
    eos_array_eQ[0]=n_eQ*ad_ratio
    eos_array_eG=electron.eos_array_from_x(electron.eosX_from_n(ad_ratio*n_eG))
    eos_array_muN=muon.eos_array_from_x(muon.eosX_from_n(ad_ratio*n_muN))
    eos_array_muQ=muon.eos_array_from_x(muon.eosX_from_n(np.abs(ad_ratio*n_muQ)))
    eos_array_muQ[0]=n_muQ*ad_ratio
    eos_array_muG=muon.eos_array_from_x(muon.eosX_from_n(ad_ratio*n_muG))
    eos_array_lepN=eos_array_eN+eos_array_muN
    eos_array_lepQ=eos_array_eQ+eos_array_muQ
    eos_array_lepG=eos_array_eG+eos_array_muG
    eos_array_lep=g* (f * eos_array_lepN + (1-f) * eos_array_lepQ) + (1-g)*eos_array_lepG
    
    eos_array_gibbs=f[:,np.newaxis]*eos_array_hadronic+(1-f[:,np.newaxis])*eos_array_quark
    eos_array_gibbs_with_lepton=np.copy(eos_array_gibbs)
    eos_array_gibbs_with_lepton[:,1]+=eos_array_lep[1]
    eos_array_gibbs_with_lepton[:,2]+=eos_array_lep[2]
    return eos_array_gibbs_with_lepton