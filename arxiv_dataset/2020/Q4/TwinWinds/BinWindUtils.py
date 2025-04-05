import numpy as np
from astropy.io import ascii
from astropy.table import Table
#from Constants import Constants
import athena_read as ar



def get_slice_val(fn,level,slice_val=0.0):
    d = ar.athdf(fn,quantities=[],level=level,subsample=True)
    return d['x3v'][ np.argmin(np.abs(d['x3v'] - slice_val) ) ]


def read_data(fn,level,
              get_slice=True,slice_dir='z',slice_val=0.0,
              lim=48.):
    print("retrieving data with level =", level," within in limit=",lim)
    
    if(get_slice):
        sv = get_slice_val(fn,level,slice_val=slice_val)
        print("slicing at ",slice_dir,"=",sv)
        
        if(slice_dir=='z'):
            d = ar.athdf(fn,
                 x1_min=-lim,x1_max=lim,
                 x2_min=-lim,x2_max=lim,
                 x3_min=sv,x3_max=sv,
                 subsample=True,
                 level=level)
        elif(slice_dir=='y'):
            d = ar.athdf(fn,
                 x1_min=-lim,x1_max=lim,
                 x2_min=sv,x2_max=sv,
                 x3_min=-lim,x3_max=lim,
                 subsample=True,
                 level=level)    
        elif(slice_dir=='x'):
            d = ar.athdf(fn,
                 x1_min=sv,x1_max=sv,
                 x2_min=-lim,x2_max=lim,
                 x3_min=-lim,x3_max=lim,
                 subsample=True,
                 level=level)
            
    else:
        d = ar.athdf(fn,
                 x1_min=-lim,x1_max=lim,
                 x2_min=-lim,x2_max=lim,
                 x3_min=-lim,x3_max=lim,
                 subsample=True,
                 level=level)
    
    x1 = -0.5
    x2 = 0.5
    #dx = d['x1v'][1]-d['x1v'][0]
    #dA = dx**2
    #dV = dx**3
    
    d['x']=np.broadcast_to(d['x1v'],(len(d['x3v']),len(d['x2v']),len(d['x1v'])) )
    d['y']=np.swapaxes(np.broadcast_to(d['x2v'],(len(d['x3v']),len(d['x1v']),len(d['x2v'])) ),1,2)
    d['z']=np.swapaxes(np.broadcast_to(d['x3v'],(len(d['x1v']),len(d['x2v']),len(d['x3v'])) ) ,0,2 )
    
    d1 = np.sqrt((d['x']-x1)**2 +d['y']**2 + d['z']**2 )
    d2 = np.sqrt((d['x']-x2)**2 +d['y']**2 + d['z']**2 )
    d['PhiEff'] = -0.5/d1 -0.5/d2 - 0.5*(d['x']**2 + d['y']**2)
    
    # torque on binary due to wind
    d['torque_dens_z'] = x1*(0.5*d['rho']/d1**3 * d['y'])   +  x2*(0.5*d['rho']/d2**3 * d['y'])
    
    d['vx'] = d['vel1'] - 1.0*d['y'] # rot frame "+ Omega x r"
    d['vy'] = d['vel2'] + 1.0*d['x']
    d['vz'] = d['vel3']
    
    print("data has shape",d['rho'].shape)
    
    return d


# mdot = A * rho * v (cm^2 * g/cm^3 * cm/s)
def get_fluxes(d,offset=0):
    
    flux = {}
    
    istart = offset
    iend   = len(d['x1v'])-1-offset
    
    print("box limits = ",d['x1v'][istart],d['x1v'][iend] )

    dx = d['x1v'][1]-d['x1v'][0]
    dA = dx**2

    flux_mdot = np.zeros(6)
    flux_jdot_z = np.zeros(6)
   
    
    # -z
    flux_mdot[0] = np.sum( d['rho'][istart,istart:iend,istart:iend]*(-d['vz'][istart,istart:iend,istart:iend])*dA )
    flux_jdot_z[0] = np.sum( d['rho'][istart,istart:iend,istart:iend]*
                            (d['x'][istart,istart:iend,istart:iend]*d['vy'][istart,istart:iend,istart:iend] 
                             - d['y'][istart,istart:iend,istart:iend]*d['vx'][istart,istart:iend,istart:iend])*
                            (-d['vz'][istart,istart:iend,istart:iend])*dA)
    # +z 
    flux_mdot[1] = np.sum( d['rho'][iend,istart:iend,istart:iend]*d['vz'][iend,istart:iend,istart:iend]*dA )
    flux_jdot_z[1] = np.sum( d['rho'][iend,istart:iend,istart:iend]*
                            (d['x'][iend,istart:iend,istart:iend]*d['vy'][iend,istart:iend,istart:iend] 
                             - d['y'][iend,istart:iend,istart:iend]*d['vx'][iend,istart:iend,istart:iend])*
                             d['vz'][iend,istart:iend,istart:iend]*dA)
    
    # -y
    flux_mdot[2] = np.sum( d['rho'][istart:iend,istart,istart:iend]*(-d['vy'][istart:iend,istart,istart:iend])*dA )
    flux_jdot_z[2] = np.sum( d['rho'][istart:iend,istart,istart:iend]*
                            (d['x'][istart:iend,istart,istart:iend]*d['vy'][istart:iend,istart,istart:iend] 
                             - d['y'][istart:iend,istart,istart:iend]*d['vx'][istart:iend,istart,istart:iend])*
                            (-d['vy'][istart:iend,istart,istart:iend])*dA)

    # +y
    flux_mdot[3] = np.sum( d['rho'][istart:iend,iend,istart:iend]*d['vy'][istart:iend,iend,istart:iend]*dA )
    flux_jdot_z[3] = np.sum( d['rho'][istart:iend,iend,istart:iend]*
                            (d['x'][istart:iend,iend,istart:iend]*d['vy'][istart:iend,iend,istart:iend] 
                             - d['y'][istart:iend,iend,istart:iend]*d['vx'][istart:iend,iend,istart:iend])*
                             d['vy'][istart:iend,iend,istart:iend]*dA)


    
    # -x 
    flux_mdot[4] = np.sum( d['rho'][istart:iend,istart:iend,istart]*(-d['vx'][istart:iend,istart:iend,istart])*dA )
    flux_jdot_z[4] = np.sum( d['rho'][istart:iend,istart:iend,istart]*
                            (d['x'][istart:iend,istart:iend,istart]*d['vy'][istart:iend,istart:iend,istart] 
                             - d['y'][istart:iend,istart:iend,istart]*d['vx'][istart:iend,istart:iend,istart])
                            *(-d['vx'][istart:iend,istart:iend,istart])*dA )


    # +x 
    flux_mdot[5] = np.sum( d['rho'][istart:iend,istart:iend,iend]*d['vx'][istart:iend,istart:iend,iend]*dA )
    flux_jdot_z[5] = np.sum( d['rho'][istart:iend,istart:iend,iend]*
                          (d['x'][istart:iend,istart:iend,iend]*d['vy'][istart:iend,istart:iend,iend] 
                           - d['y'][istart:iend,istart:iend,iend]*d['vx'][istart:iend,istart:iend,iend])
                          *d['vx'][istart:iend,istart:iend,iend]*dA )



    flux['mdot'] = np.sum(flux_mdot)
    flux['mdot_fluxes'] = flux_mdot
    flux['Ldotz'] = np.sum(flux_jdot_z)
    flux['Ldotz_fluxes'] = flux_jdot_z
    
    return flux


def get_fluxes_sphere(d,radius):
    flux={}
    
    flux_mdot = np.zeros(6)
    flux_jdot_z = np.zeros(6)
    
    rin = np.where( np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)<radius, True, False)

    cond_mz = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[2:,1:-1,1:-1]==True))
    cond_pz = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[0:-2,1:-1,1:-1]==True))

    cond_my = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[1:-1,2:,1:-1]==True))
    cond_py = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[1:-1,0:-2,1:-1]==True))

    cond_mx = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[1:-1,1:-1,2:]==True))
    cond_px = ((rin[1:-1,1:-1,1:-1]==False)  & (rin[1:-1,1:-1,0:-2]==True))

    dx = d['x1v'][1]-d['x1v'][0]
    dA = dx**2
    d['l'] = d['x']*d['vy'] - d['y']*d['vx']

    flux_mdot[0] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_mz]*(-d['vz'][1:-1,1:-1,1:-1][cond_mz])*dA)
    flux_mdot[1] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_pz]*(d['vz'][1:-1,1:-1,1:-1][cond_pz])*dA)
    flux_mdot[2] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_my]*(-d['vy'][1:-1,1:-1,1:-1][cond_my])*dA)
    flux_mdot[3] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_py]*(d['vy'][1:-1,1:-1,1:-1][cond_py])*dA)
    flux_mdot[4] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_mx]*(-d['vx'][1:-1,1:-1,1:-1][cond_mx])*dA)
    flux_mdot[5] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_px]*(d['vx'][1:-1,1:-1,1:-1][cond_px])*dA)

    flux_jdot_z[0] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_mz]*(-d['vz'][1:-1,1:-1,1:-1][cond_mz])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_mz])
    flux_jdot_z[1] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_pz]*(d['vz'][1:-1,1:-1,1:-1][cond_pz])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_pz])
    flux_jdot_z[2] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_my]*(-d['vy'][1:-1,1:-1,1:-1][cond_my])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_my])
    flux_jdot_z[3] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_py]*(d['vy'][1:-1,1:-1,1:-1][cond_py])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_py])
    flux_jdot_z[4] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_mx]*(-d['vx'][1:-1,1:-1,1:-1][cond_mx])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_mx])
    flux_jdot_z[5] = np.sum(d['rho'][1:-1,1:-1,1:-1][cond_px]*(d['vx'][1:-1,1:-1,1:-1][cond_px])*dA
                            *d['l'][1:-1,1:-1,1:-1][cond_px])

    flux['mdot'] = np.sum(flux_mdot)
    flux['mdot_fluxes'] = flux_mdot
    flux['Ldotz'] = np.sum(flux_jdot_z)
    flux['Ldotz_fluxes'] = flux_jdot_z
    
    return flux





def sum_torque(d,fphi,offset=0):
    
    istart = offset
    iend   = len(d['x1v'])-offset
    
    # sum torque = Ldot_grav
    dx = d['x1v'][1]-d['x1v'][0]
    dV = dx**3 
    cond = ((d['PhiEff']<-2*fphi) & (np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)<1.2))[istart:iend,istart:iend,istart:iend]
    Ldot_grav = np.sum( np.ma.masked_where(cond,
                           (d['torque_dens_z'][istart:iend,istart:iend,istart:iend])*dV) )
    return Ldot_grav


def sum_torque_sphere(d,fphi,radius):
    
    # sum torque = Ldot_grav
    dx = d['x1v'][1]-d['x1v'][0]
    dV = dx**3 
    cond = (((d['PhiEff']<-2*fphi) &  (np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)<1.2)) |
            (np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)>radius))
    Ldot_grav = np.sum( np.ma.masked_where(cond,d['torque_dens_z']*dV) )
    return Ldot_grav





def get_mdot_gammas(d,fphi,offset=0):
    f = get_fluxes(d,offset)
  
    # l_loss 
    lbin = 0.25
    lloss = -f['Ldotz']/-f['mdot']
    gamma_loss = lloss/lbin 

    Ldot_grav = sum_torque(d,fphi,offset)
    
    # wind 
    # (Lgrav > 0 accelerates binary, decelerates wind)
    # (Lgrav < 0 decelarates binary, accelerates wind)
    lgrav = - Ldot_grav / f['mdot']  
    gamma_grav = lgrav/lbin
    
    lwind = lloss - lgrav
    gamma_wind = lwind/lbin    
    
    return -f['mdot'], -f['Ldotz'], Ldot_grav, gamma_wind, gamma_grav, gamma_loss


def get_mdot_gammas_sphere(d,fphi,radius):
    f = get_fluxes_sphere(d,radius)
  
    # l_loss 
    lbin = 0.25
    lloss = -f['Ldotz']/-f['mdot']
    gamma_loss = lloss/lbin 

    Ldot_grav = sum_torque_sphere(d,fphi,radius)
    
    # wind 
    # (Lgrav > 0 accelerates binary, decelerates wind)
    # (Lgrav < 0 decelarates binary, accelerates wind)
    lgrav = - Ldot_grav / f['mdot']  
    gamma_grav = lgrav/lbin
    
    lwind = lloss - lgrav
    gamma_wind = lwind/lbin    
    
    return -f['mdot'], -f['Ldotz'], Ldot_grav, gamma_wind, gamma_grav, gamma_loss


def get_vr10(d):
    dV=(d['x1v'][1]-d['x1v'][0])**3
    d['r'] = np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)
    d['vr'] =  d['vx']*d['x']/d['r'] + d['vy']*d['y']/d['r'] + + d['vz']*d['z']/d['r']
    sel10 = (d['r']>9.5) & (d['r']<10.5)
    vr10=np.sum(d['vr'][sel10]*d['rho'][sel10]*dV)/np.sum(d['rho'][sel10]*dV)
    return vr10