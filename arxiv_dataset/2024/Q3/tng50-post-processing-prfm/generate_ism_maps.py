import sys
import illustris_python as il
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import utils
import pandas as pd
import gc
import matplotlib.pyplot as plt


# ====== constants ===========

kB      = 1.3807e-16 # Boltzmann's constan in cm^2 g s^-2 K^-1 
Msun_per_gram   = 1.989e+33
pc_per_cm       = 3.086e+18
Proton_mass_cgs = 1.6726e-24


def GetMap(gal_id):

    # ================================================================================================================================== #
    # =========================== GAS BLOCK ============================================================================================ #
    # ================================================================================================================================== #
    
    # Reading gas particle properties

    gas_coord    = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='Coordinates')/Hubble/(1.+Redshift) #coordinates in pkpc
    gas_vel      = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='Velocities')*np.sqrt(Scalefactor)  #velocities in km/s
    gas_sfr      = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='StarFormationRate') # SFR in  Msun/yr 
    gas_mass     = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='Masses')*1e10/Hubble #masses in Msun
    gas_fNH      = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='NeutralHydrogenAbundance') # fraction of neutral hydrogen 
    gas_fH       = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='GFM_Metals')[:,0] # fraction of hydrogen
    gas_rho      = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='Density')*(1.+Redshift)*(1.+Redshift)*(1.+Redshift)*Hubble*Hubble*1.e10 # density in Msun/kpc^3
    gas_dm_rho   = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='SubfindDMDensity')*(1.+Redshift)*(1.+Redshift)*(1.+Redshift)*Hubble*Hubble*1.e10 # dark matter density in Msun/kpc^3
    gas_B        = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='MagneticField')*Hubble*Hubble*2.6e-6*(1.+Redshift)*(1.+Redshift) # magenetic field in Gauss  =  1 cm^−1/2 g^1/2 s^−1
    gas_E        = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='InternalEnergy') # internal energy in (km/s)^2 
    gas_Z        = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'gas', fields='GFM_Metallicity')/0.0127 # Metallicity in solar units
    

    # Getting the subhaloe (galaxy) properties
    
    star_radius    = gal_star_radius[gal_id] # kpc
    gal_pos        = gal_pos_all[gal_id] # kpc
    gal_vel        = gal_vel_all[gal_id] # km/s


    # Check boundary conditions (some galaxies are in the edges!)                                                                                            
    gas_coord = utils.apply_boundary_conditions(gas_coord,gal_pos, BoxSize,0)
    

    # Centering gas particles coordinates and velocities    
    gas_coord -= gal_pos
    gas_vel   -= gal_vel

    # Find face-on rotation matrix along angular momentum direction
    
    radius              = np.sqrt(np.sum(gas_coord**2.0, axis=1)) # compute radius using all gas coordinates

    mask_rad_sfr        = (radius <  2.0*star_radius) & ( gas_sfr > 0.0  ) # consider only star-forming particles within twice stellar radius

    masked_gas_coord    = gas_coord[mask_rad_sfr] # applying mask_rad_sfr
    masked_gas_vel      = gas_vel[mask_rad_sfr]
    masked_gas_mass     = gas_mass[mask_rad_sfr]
    
    angular_momentum    = (masked_gas_mass.reshape((len(masked_gas_mass), 1)) * np.cross(masked_gas_coord, masked_gas_vel)).sum(axis=0) # find angular momentum
    rotation_matrix     = utils.rotation_matrix(angular_momentum, reference_axis=[0.0, 0.0, 1.0])   # find rotation matrix 

        
    # Rotating coordinates and velocities
    
    rotated_gas_coord       = np.dot(gas_coord,rotation_matrix)
    rotated_gas_vel         = np.dot(gas_vel,rotation_matrix)
    rotated_radius          = np.sqrt(np.sum(rotated_gas_coord**2.0, axis=1))

    # Define grid (maps) parameters using either twise the stellar radius or 95% of the star forming region

    mask_sfr                = gas_sfr > 0.0  # again consider only star forming particles.
    R_95                    = np.max( [2*star_radius,np.quantile(rotated_radius[mask_sfr], 0.95)] )
    if np.isnan(R_95): R_95 = 2*star_radius
    bins_xy                 = np.arange(-np.ceil(R_95),np.ceil(R_95)+cell_size, cell_size) # same bins in x and y directions
    ndim                    = bins_xy.size
    #print ("There is ", ndim, "cells for this subhalos", gal_id, "and R95", star_radius, np.quantile(rotated_radius[mask_sfr], 0.95))

    #if ndim > 200:
    #    print ("............................. Report a problem!!!!!")
    
    # Consider only the neutral fraction mass for non star forming cells.
    mask_non_sfr            = gas_sfr <= 0.0
    gas_mass[mask_non_sfr]  = gas_mass[mask_non_sfr]*gas_fNH[mask_non_sfr]


    # Mask to identify particles with coordinates x, y within +/- R_95 and z within +/- int_scale
    mask_R95_int_scale_g = (rotated_gas_coord[:,0] <= R_95)  & (rotated_gas_coord[:,1] <= R_95)  & (rotated_gas_coord[:,2] <= int_scale) & (rotated_gas_coord[:,0] >= -R_95) & (rotated_gas_coord[:,1] >= -R_95) & (rotated_gas_coord[:,2] >= -int_scale)

    # Apply this mask to all gas properites
    
    rotated_gas_coord    = rotated_gas_coord[mask_R95_int_scale_g]
    rotated_gas_vel      = rotated_gas_vel[mask_R95_int_scale_g]
    gas_mass             = gas_mass[mask_R95_int_scale_g]
    gas_sfr              = gas_sfr[mask_R95_int_scale_g]
    gas_fH               = gas_fH[mask_R95_int_scale_g]
    gas_rho              = gas_rho[mask_R95_int_scale_g]
    gas_dm_rho           = gas_dm_rho[mask_R95_int_scale_g]
    gas_B                = gas_B[mask_R95_int_scale_g]
    gas_E                = gas_E[mask_R95_int_scale_g]
    gas_Z                = gas_Z[mask_R95_int_scale_g]
    

    # Compute integrated quantities such as surface density of gas, SFR
    
    Grid_Sigma_gas       = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights=gas_mass)[0]/(cell_size*cell_size)
    Grid_Sigma_SFR       = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights=gas_sfr)[0]/(cell_size*cell_size)

    # Compute mass weighted averagred metallicity
    Grid_Z_gas           = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights=gas_Z*gas_mass)[0]/Grid_Sigma_gas
    
    # Mask to identify particles within the mid-plane (z coordinates within +/- mid_scale) 

    mask_mid_scale_g       = (rotated_gas_coord[:,2] <= mid_scale) & (rotated_gas_coord[:,2] >= -mid_scale)

    # Apply this mask

    rotated_gas_coord = rotated_gas_coord[mask_mid_scale_g]
    rotated_gas_vel   = rotated_gas_vel[mask_mid_scale_g]
    gas_mass          = gas_mass[mask_mid_scale_g]
    gas_fH            = gas_fH[mask_mid_scale_g]
    gas_rho           = gas_rho[mask_mid_scale_g]
    gas_dm_rho        = gas_dm_rho[mask_mid_scale_g]
    gas_B             = gas_B[mask_mid_scale_g]
    gas_E             = gas_E[mask_mid_scale_g]


    # Compute mid-plane mass averaged quantities such as densities and various pressure components (thermal, magentic, turbulent)
    
    Grid_total_mass = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights=gas_mass)[0]    
    Grid_rho_gas    = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= gas_rho*gas_mass)[0]/Grid_total_mass     
    Grid_DMrho_gas  = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= gas_dm_rho*gas_mass)[0]/Grid_total_mass 

    Grid_nH_gas     = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= gas_fH*gas_rho*gas_mass)[0]/Grid_total_mass
    Grid_nH_gas    *= (Msun_per_gram/Proton_mass_cgs)/(pc_per_cm*pc_per_cm*pc_per_cm*1e9) # convert from mass density to number density (e.g. dividing by proton mass) and convert units from Msun/kpc^3/g to cm^-3
    
   # Msun/kpc^3/ Proton_mass_cgs
    
    Grid_P_th       = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= (2./3.)*gas_mass*gas_rho*gas_E)[0]/Grid_total_mass
    Grid_P_th      *= Msun_per_gram / (kB *pc_per_cm *pc_per_cm *pc_per_cm/10) # convert from Msun/kpc^3 * (km/s)^2 to K/cm^3
    
    Grid_P_mag      = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= gas_mass*(gas_B[:,0]*gas_B[:,0] + gas_B[:,1]*gas_B[:,1] - gas_B[:,2]*gas_B[:,2])/ (8*np.pi) )[0]/Grid_total_mass
    Grid_P_mag     /= kB # change units from Gauss^2 to K/ cm^3 
    
    Grid_P_turb     = np.histogram2d(rotated_gas_coord[:,0], rotated_gas_coord[:,1], bins=[bins_xy,bins_xy],weights= gas_mass*gas_rho*rotated_gas_vel[:,2]*rotated_gas_vel[:,2] )[0]/Grid_total_mass
    Grid_P_turb    *= Msun_per_gram / (kB *pc_per_cm *pc_per_cm *pc_per_cm/10) # convert from Msun/kpc^3 * (km/s)^2 to K/cm^3 
    
    Grid_H_gas      = Grid_Sigma_gas/(2.0*Grid_rho_gas)

    # =================================================================================================================================== #
    # =========================== STAR BLOCK ============================================================================================ #
    # =================================================================================================================================== #
    
    # Reading star particle properties    
    star_coord       = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'stars', fields='Coordinates')/Hubble/(1.+Redshift) #coordinates in pkpc                       
    star_mass        = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'stars', fields='Masses')*1e10/Hubble # mass in  Msun    
    star_tf          = il.snapshot.loadSubhalo(basePath, snap, gal_id, 'stars', fields='GFM_StellarFormationTime') #  real stars have tf>0 otherwise wind
    star_coord       = star_coord[star_tf>0.0]
    star_mass        = star_mass[star_tf>0.0]


    # Check boundary conditions (some galaxies are in the edges!)    
    star_coord = utils.apply_boundary_conditions(star_coord,gal_pos, BoxSize,1)
    
    # Centering star particles coordinates
    star_coord -= gal_pos

    # Rotating star coordinates
    
    rotated_star_coord           = np.dot(star_coord,rotation_matrix)

    # Mask to identify particles with coordinates x, y within +/- R_95 and z within +/- int_scale
    
    mask_R95_int_scale_s      = (rotated_star_coord[:,0] <= R_95)  & (rotated_star_coord[:,1] <= R_95)  & (rotated_star_coord[:,2] <= int_scale) & (rotated_star_coord[:,0] >= -R_95) & (rotated_star_coord[:,1] >= -R_95) & (rotated_star_coord[:,2] >= -int_scale)

    # Apply this mask
    
    rotated_star_coord   = rotated_star_coord[mask_R95_int_scale_s]
    star_mass            = star_mass[mask_R95_int_scale_s]

    # Compute Surface density of stars (integrated over int_scale)
    
    Grid_Sigma_star      = np.histogram2d(rotated_star_coord[:,0], rotated_star_coord[:,1], bins=[bins_xy,bins_xy],weights=star_mass)[0]/(cell_size*cell_size)

    # Mask to identify particles within the mid-plane (z coordinates within +/- mid_scale)                                                                                              
    mask_mid_scale_s       = (rotated_star_coord[:,2] <= mid_scale) & (rotated_star_coord[:,2] >= -mid_scale)

    # Apply this mask

    rotated_star_coord   = rotated_star_coord[mask_mid_scale_s]
    star_mass            = star_mass[mask_mid_scale_s]
    
    # find the volumetic density of stars
    
    Grid_rho_star  = utils.compute_density_grid(rotated_star_coord[:,0], rotated_star_coord[:,1], rotated_star_coord[:,2], star_mass, bins_xy, ndim, mid_scale)
    Grid_H_star    = Grid_Sigma_star/(2.0*Grid_rho_star)
    

    
    return np.array([gal_id, Grid_Sigma_gas, Grid_rho_gas,Grid_H_gas, Grid_nH_gas, Grid_Z_gas, Grid_P_th, Grid_P_mag, Grid_P_turb, Grid_Sigma_star, Grid_rho_star, Grid_H_star, Grid_Sigma_SFR,  Grid_DMrho_gas], dtype=object)


if __name__ == "__main__":

    start   = time.time()
    
    snap    = int(sys.argv[1]) # from https://www.tng-project.org/, see Snap tables, e.g. 99 for redshift=0                                                     
    
    # define map/grid resolution (cell_size in proper kpc), height for integrated quantities e.g. surface density (int_scale in proper kpc), height to define midplane (mid_scale in proper kpc)
    cell_size, int_scale, mid_scale  = 1.0, 10.0, 0.1

    # path to TNG50 simulation directory
    basePath='/Illustris_IllustrisTNG_public_data_release/L35n2160TNG/output/'
    

    #  Reading header 
    Redshift            = il.groupcat.loadHeader(basePath, snap)['Redshift']
    Hubble              = il.groupcat.loadHeader(basePath, snap)['HubbleParam']
    BoxSize             = il.groupcat.loadHeader(basePath, snap)['BoxSize']/Hubble/(1.+Redshift) # kpc
    Scalefactor         = il.groupcat.loadHeader(basePath, snap)['Time']

    print ("Redshift = %.2f" % Redshift)
    #print ("Box Size = %.2f kpc" % BoxSize)
    
    #  Reading subhaloes catalogs

    gal_star_radius = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloHalfmassRadType')[:,4]/Hubble/(1.+Redshift) # radius containing half of stellar mass in kpc 
    gal_pos_all     = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloPos')/Hubble/(1.+Redshift) # coordinates in kpc                                             
    gal_vel_all     = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloVel') # velocities in km/s

    gal_flag        = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloFlag') # real galaxy = 1 otherwise 0
    gal_M_star      = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloMassType')[:,4]*1e10/Hubble # Msun
    gal_SFR         = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloSFRinRad') # Msun/yr
    gal_part_num    = il.groupcat.loadSubhalos(basePath,snap,fields='SubhaloLenType') # number of particles of each type [N,6] 
    gal_gas_num     = gal_part_num[:,0] # number of gas particles
    gal_dm_num      = gal_part_num[:,1] # number of dark matter particles
    gal_star_num    = gal_part_num[:,4] # number of star particles

    # generate galaxies IDs
    gal_ids  = np.arange(gal_flag.size)

    # Apply sample selection criteria (stellar mass -> 1e7-1e11 Msun, min of 100 particles of each type, SFR > 5e-4 Msun/yr, flag=1)
    
    mask_sample = (gal_M_star >= 1e7) & (gal_M_star <= 1e11) & (gal_gas_num >= 100) & (gal_star_num >= 100) & (gal_dm_num >= 100) & (gal_SFR >= 5e-4)  & (gal_flag==1)
    gal_ids     = gal_ids[mask_sample]


    # freeing memory and force garbage collection
    del gal_flag
    del gal_M_star
    del gal_SFR
    del gal_part_num
    del gal_gas_num
    del gal_dm_num
    del gal_star_num
    del mask_sample
    gc.collect()
    
    print ("Working on ", gal_ids.size, "selected subhalos ....")
    
    # Multiprocessing 
    pool    = mp.Pool(mp.cpu_count())

    print (".... using", mp.cpu_count(), "available cpus ...." )
    results = pool.map_async(GetMap, [gal_id for gal_id in gal_ids]).get()
    pool.close() 

    column_names = ["subhalo_id", "Sigma_g", "rho_g","H_g", "n_H","Z_g","P_th", "P_mag", "P_turb", "Sigma_*", "rho_*", "H_*", "Sigma_SFR", "rho_dm"]
                      
    
    # saving all maps    
    df  = pd.DataFrame(data=results, columns=column_names)
    df.to_pickle("./data/all_data_z"+"%i" % Redshift +".pkl")

    print ('Time spent = ', ( time.time()  - start)/60.0, 'mins')

    
