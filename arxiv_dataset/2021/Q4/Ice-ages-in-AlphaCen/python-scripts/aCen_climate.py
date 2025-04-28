import numpy as np
import os
import sys
import shutil
import rebound
from rebound.interruptible_pool import InterruptiblePool
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
import subprocess as sb
import threading


def get_dist_sqr(p1,p2):
    dist_sqr = (p2.x-p1.x)**2 + (p2.y-p1.y)**2 + (p2.z-p1.z)**2
    return dist_sqr
def get_Spectral_Weight(T,atm_idx):
    runaway_green = [1.2456e-4,1.4612e-8,-7.6345e-12,-1.1950e-15]
    max_green = [5.9578e-5, 1.6707e-9,-3.0058e-12,-5.1925e-16]
    recent_venus = [1.4335e-4,3.3954e-9,-7.6364e-12,-1.1950e-15]
    early_mars = [5.4471e-5,1.5275e-9,-2.1709e-12,-3.8282e-16]
    atm_coeff = [runaway_green,max_green,recent_venus,early_mars]

    lx_sun = [0.97,1.67,0.75,1.77]
    T_scale = T - 5780
    alpha = 0.
    for i in range(1,5):
        alpha += atm_coeff[atm_idx][i-1]*T_scale**i
    return 1./(1. + alpha*lx_sun[atm_idx]**2)


def write2file(f_str,head,output_fldr,fn):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
        f = open(output_fldr+fn,'w')
    else:
        f = open(output_fldr+fn,'a')
    f.write(f_str)
    f.close()
    lock.release()


def get_avg_Flux(M_cent,semi,ecc):
    norm = 0
    phi_step = 2.*np.pi/10.
    phi_rng = np.arange(0,2.*np.pi,phi_step)
    avg_Flux = 0
    for phi in phi_rng :
        r = semi*(1.-ecc**2)/(1.+ecc*np.cos(phi))
        S_p = L_A/r**2*1361.
        fact = (1.-ecc**2)**1.5/(1.+ecc*np.cos(phi))**2
        norm += fact
        avg_Flux += fact*S_p
    avg_Flux = avg_Flux/norm
    return avg_Flux

def X2eps(x):
    return np.degrees(np.arccos(x))

def unfold_2pi(t,x):
    x_unfold = np.zeros(len(t))
    NN = 0
    for j in xrange(0,len(t)-1):
        diff = x[j+1] - x[j]
        if diff > 1.:
            NN -= 1
        if diff < -np.pi:
            NN += 1
        x_unfold[j+1] = x[j+1] + 2.*NN*np.pi
    return x_unfold


def deriv(y,t,alphat,ecct,CAt,CBt,CCt):
    Xt, psit = y
    if np.abs(Xt) < 1:
        term = Xt/np.sqrt(1.-Xt**2)
        root = np.sqrt(1.-Xt**2)
    else:
        term = -1.
        root = 0.
        
    dHdpsi = root*(CAt*np.cos(psit) - CBt*np.sin(psit))
    dHdX = alphat*Xt/(1.-ecct**2)**1.5-term*(CAt*np.sin(psit)+CBt*np.cos(psit)) - 2.*CCt
    return [-dHdpsi,dHdX]

def run_obl(days,semi_t,ecc_t,inc_t,argP_t,RA_t,eps,alpha,dest):
    #semi_t,ecc_t,inc_t,argP_t,RA_t are arrays from run_sim()
    time_arr = np.copy(days)/365.25

    inc = np.radians(inc_t)
    RA = np.radians(RA_t)
    RA_cont = unfold_2pi(time_arr,RA)

    q = np.sin(inc/2.)*np.cos(RA)
    p = np.sin(inc/2.)*np.sin(RA)

    qt = UnivariateSpline(time_arr, q, s=0)
    pt = UnivariateSpline(time_arr, p, s=0)

    pdot = pt.derivative()(time_arr)
    qdot = qt.derivative()(time_arr)

    CC = qt(time_arr)*pdot-pt(time_arr)*qdot
    CA = 2./np.sqrt(1.-pt(time_arr)**2-qt(time_arr)**2)*(qdot + pt(time_arr)*CC)
    CB = 2./np.sqrt(1.-pt(time_arr)**2-qt(time_arr)**2)*(pdot - qt(time_arr)*CC)
    
    
    Chi,psi = np.zeros(len(time_arr)),np.zeros(len(time_arr))
    i_spin = np.zeros(len(time_arr))
    if eps == 0:
        eps = 1e-6
    Chi[0] = np.cos(np.radians(eps))
    psi[0] = np.radians(23.761) #const_psi
    
    #dest = "%s/aCenB_Climate_%s/Obl[%03i]/Alpha[%03i]/" % (home,run_type,eps,alpha)
    #write2file("#time,semi,ecc,argP,LongA,Obl,PrecA\n",'head',sub_dir,'obl_data.txt')
    write2file("%1.6e %1.6f %1.6f %3.5f %3.5f %3.5f %3.5f\n" % (0,semi_t[0],ecc_t[0],argP_t[0],RA_t[0],eps,np.degrees(psi[0])),'head',dest,'obl_data.txt')
    tstep = time_arr[1]/10.
    
    for t in range(1,len(time_arr)):
        delta_t = np.arange(time_arr[t-1],time_arr[t]+tstep,tstep)
        y0 = [Chi[t-1],psi[t-1]]
        alp = alpha/3600.*np.pi/180.
        temp = odeint(deriv,y0,delta_t,args=(alp,ecc_t[t-1],CA[t-1],CB[t-1],CC[t-1]),atol=1e-10,rtol=1e-10)
        Chi[t],psi[t] = temp[-1,0],temp[-1,1]
        if not np.isfinite(Chi[t]):
            Chi[t] = 1.

        ArgP = argP_t[t] % 360.
        LongP = RA_t[t] % 360.
        Obl = X2eps(Chi[t])
        PrecA = np.degrees(psi[t])%360.
        write2file("%1.6e %1.6f %1.6f %3.5f %3.5f %3.5f %3.5f\n" % (time_arr[t],semi_t[t],ecc_t[t],ArgP,LongP,Obl,PrecA),'foot',dest,'obl_data.txt')

    return

def run_sim(i_p):
    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.units = ('days', 'AU', 'Msun')
    if run_type == 'multi':
        T_min = np.sqrt(SS_orb[0,1]**3/M_A)*365.25
    else:
        T_min = np.sqrt(a_p**3/M_A)*365.25
    sim.dt = 0.05*T_min

    #add the binary first
    sim.add(m=M_A)
    #add the planet
    if run_type == 'single':
        sim.add(m=M_p,a=a_p,e=e_p,omega=omg_bin,inc=np.radians(i_p),M=MA_p) #defaults for other values are zero
    else:
        #add the SS planets
        for i in range(0,4):
            sim.add(m=SS_orb[i,0],a=SS_orb[i,1],e=SS_orb[i,2],inc=np.radians(SS_orb[i,3]),omega=np.radians(SS_orb[i,4]),Omega=np.radians(SS_orb[i,5]),M=np.radians(SS_orb[i,6]))

    sim.move_to_com()
    sim.add(m=M_B,a=a_bin,e=e_bin,omega=omg_bin,M=MA_bin)

    ps = sim.particles
    


    orb_dir = data_dir + "aCenA_orb_%s/" % run_type
    if not os.path.exists(orb_dir):
        os.makedirs(orb_dir)
    
    #F_t = np.zeros(len(days))
    fname = "Orb_data_%i.txt" % i_p
    write2file("",'head',orb_dir,fname) ##time,flux,ecc,varpi,inc,Omg\n
    Earth_idx = 1
    if run_type == 'multi':
        Earth_idx = 3
    for d in range(0,len(days)):
        sim.integrate(days[d])
        #F_t[d] = (L_A/get_dist_sqr(ps[1],ps[2]) + L_B/get_dist_sqr(ps[1],ps[0]))*L_sun/AU**2
        #F_t[d] = get_avg_Flux(M_A,ps[1].a,ps[1].e)
        #F_t[d] = (L_B/get_dist_sqr(ps[1],ps[0]))*L_sun/AU**2
        ecc_t = ps[Earth_idx].e
        #varpi_t = np.degrees(ps[Earth_idx].pomega)
        omg_t = np.degrees(ps[Earth_idx].omega) % 360.
        inc_t = np.degrees(ps[Earth_idx].inc)
        RA_t = np.degrees(ps[Earth_idx].Omega) % 360.
        MA_t = np.degrees(ps[Earth_idx].M) % 360.
        if d == 0 and run_type == 'multi':
            omg_t = SS_orb[Earth_idx,4] % 360.
            RA_t = SS_orb[Earth_idx,5] % 360.
            MA_t = SS_orb[Earth_idx,6] % 360.

        out_stg = "%1.6e %1.6f %1.6f %2.3f %2.3f %2.3f %2.3f\n" % (days[d]/365.25,ps[Earth_idx].a,ecc_t,inc_t,omg_t,RA_t,MA_t)
        write2file(out_stg,'foot',orb_dir,fname)

    return 

def copy_Sun_in(d,Mstar,Rstar,Lstar):
    indir = home + "VPlanet_files/"
    infile = 'sun.in'
    with open(indir+infile,'r') as f:
        paramin = f.readlines()
    out = open(d+infile,'w')
    for i in xrange(0,len(paramin)):
        if i == 2:
            out.write("dMass        %1.3f\n" % Mstar)
        elif i==5:
            out.write("dRadius      %1.5f\n" % (Rstar*R_sun))
        elif i==6:
            out.write("dLuminosity  %1.4e\n" % (Lstar*L_sun) )
        else:
            out.write(paramin[i])
    

def copy_Earth_in(d,eps,alpha_idx):
    indir = home + "VPlanet_files/"
    infile = 'earth.in'
    with open(indir+infile,'r') as f:
        paramin = f.readlines()
    out = open(d+'earth.in','w')
    for i in xrange(0,len(paramin)):
        if i == 5:
            out.write("dObliquity   %3.2f\n" % eps)
        elif i==4:
            out.write("dRotPeriod   -%1.8f\n" % Spin_params[alpha_idx,2])
        elif i==8:
            out.write("dSemi        %1.8f\n" % a_p)
        elif i == 12:
            out.write("#dPrecRate    %1.8f\n" % Spin_params[alpha_idx,1])
        elif i == 10:
            out.write("dDynEllip    %1.8f\n" % Spin_params[alpha_idx,3])
        else:
            out.write(paramin[i])

def copy_vpl_in(d,tscale):
    indir = home + "VPlanet_files/"
    infile = 'vpl.in'
    with open(indir+infile,'r') as f:
        paramin = f.readlines()
    out = open(d+infile,'w')
    for i in xrange(0,len(paramin)):
        if i == 15:
            out.write("dStopTime     %1.1e\n" % tscale)
        else:
            out.write(paramin[i])

def Run_VPlanet(par_idx):
    eps,alpha,i_p = params[par_idx]
    orb_dir = data_dir + "aCenA_orb_%s/" % run_type
    dest = "%s/aCenA_Climate_%s_%i/Obl[%03i]/Alpha[%03i]/" % (home,run_type,i_p,eps,alpha)
    if not os.path.exists(dest):
        os.makedirs(dest)
    #shutil.copy2(orb_dir+"Orb_data.txt",dest+"Orb_data.txt")
    #run obliquity evolution
    a_t,e_t,i_t,w_t,O_t = np.genfromtxt(orb_dir+"Orb_data_%i.txt" % i_p,usecols=(1,2,3,4,5),unpack=True)
    run_obl(days,a_t,e_t,i_t,w_t,O_t,eps,alpha,dest)
    #create VPlanet files
    #shutil.copy2(home+ "VPlanet_files/vpl.in",dest+"vpl.in") #copy vpl file
    copy_vpl_in(dest,tscale)
    copy_Sun_in(dest,M_A,R_B,L_A)

    alpha_idx = np.where(np.abs(alpha-Spin_params[:,0])<1e-2)[0]
    copy_Earth_in(dest,eps,alpha_idx)

    os.chdir(dest)
    sb.call("vplanet vpl.in >& output.txt",shell=True)
    if os.path.exists(dest+"Orb_data.txt"):
        os.remove(dest+"Orb_data.txt")
    #sb.call("tar -czf aCenB_[%03i,%03i].tar.gz *.*" % (eps,alpha),shell=True)
    '''file_list = [f for f in os.listdir('.') if not f.endswith('tar.gz') or not os.path.isdir(f)]
    if len(file_list) > 0:
        for fl in file_list:
            os.remove(fl)'''



lock = threading.Lock()
L_sun = 3.828e26 #Solar Luminosity in Watts
AU = 1.495978707e11 #Astronomical Unit in m
R_sun = 0.00465047 #Solar Radius in AU
sigma = 5.67e-8 #Boltzmann constant
G = 4.*np.pi**2/365.25**2
F_pl = 1. #in Earth Flux

aspect = 4./3.
width = 10.
lw = 4
fs = 'x-large'
home = os.getcwd() + "/"
data_dir = home + "aCenA_orb/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

lat = np.arange( -90., 91., 1.)

M_A = 1.133
M_B = 0.972
L_A = 1.512 #M_A**4
L_B = 0.5 #M_B**4
R_A = 1.2234
R_B = 0.8632

T_A = M_A**0.73*5780
T_B = M_B**0.73*5780
a_bin = 23.78
e_bin = 0.524
omg_bin = np.radians(77.05)
MA_bin = np.radians(209.6901)
T_bin = np.sqrt(a_bin**3/(M_A+M_B))*365.25

F_A = get_Spectral_Weight(T_A,0)*L_A
F_B = get_Spectral_Weight(T_B,0)*L_B

M_p = 3.0035e-6

a_p = np.sqrt(L_A/F_pl)
e_p = 0.0#257
T_p = np.sqrt(a_p**3/(M_A))*365.25
T_p = 365.25
#days = np.arange(0, 1e5*365.25+10.*T_p, 10.*T_p)
SS_orb = np.zeros((4,7))
SS_orb[0,:] = [1.66011415305e-07,2.71138252e-01,0.205630292982,7.005014141,29.12428281,48.33053734,174.795883]
SS_orb[1,:] = [2.44783828778e-06,5.09871562e-01,0.00675563660057,3.394589829,55.1834597,76.67838505,50.11725433]
SS_orb[2,:] = [3.04043264802e-06,0.707106781,0.0166990469644,0.00011013,-38.30828091,141.222759,-2.451572364]
SS_orb[3,:] = [3.22715603755e-07,1.08187420,0.0933197193532,1.849878459,-73.46216375,49.56189469,19.35591626]

'''SS_orb[0,1] = 4.96808711e-01
SS_orb[1,1] = 9.03536382e-01
SS_orb[2,1] = 1.23247718
SS_orb[3,1] = 1.84507184'''

#a_p = np.sqrt(L_B/F_pl)
#e_p = 0.0257 # for planets orbiting star B
e_p = -0.007*a_p**2 + 0.044*a_p - 0.002 #for star A
#T_p = np.sqrt(SS_orb[2,1]**3/(M_A))*365.25
MA_p = np.radians(2.22492236e+02)
n_orb = 10
tscale = 5e5
days = np.arange(0, (tscale+n_orb)*365.25, n_orb*365.25)
run_type = sys.argv[3]
i_p = int(sys.argv[4])
#Run N-body simulation
#run_sim(i_p)
#sys.exit(0)

#long_peri = -45
#obl_var = 45.+45.*np.sin(2.*np.pi/(5*T_bin)*days + np.pi)

obl_st = int(sys.argv[1])
alph_st = 50*int(sys.argv[2])

params = []
for o in np.arange(obl_st,obl_st+1):
#for o in np.arange(0,91,1):
    for alf in np.arange(alph_st,alph_st+51):
    #for alf in np.arange(0,101,1):
        if o == 0:
            params.append((0.001,alf,i_p))
        else:
            params.append((o,alf,i_p))

Spin_params = np.genfromtxt("aCenA_spin_params.txt",delimiter=',',comments='#')

#Run_VPlanet(46)
#sys.exit(0)


pool = InterruptiblePool(processes=20)

pool.map(Run_VPlanet,range(0,len(params)))
pool.close()
sys.exit(0)

#fire off workers
jobs = []

for p in range(0,len(params)):
    job = pool.apply_async(Run_VPlanet, (params[p],))
    jobs.append(job)

# collect results from the workers through the pool result queue
for job in jobs: 
    job.get()

