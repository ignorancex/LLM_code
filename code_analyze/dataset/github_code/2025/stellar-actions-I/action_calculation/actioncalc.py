#imports

import numpy as np
import h5py

import functions as f
import mid_res_info as md
import potential_der as pd
import kappa as k
import nu as n


##combining all these functions into action function that uses kappa and nu to calculate the actions of the stars in ID_list

def action_func(i,j,filepath,ID_list):
    """
    takes i and the ID_list/age as input
    gives ID,actions,velocities,positions, masses, kappa and nu of stars at i-th snapshot.
    """
    C_cyl,V_cyl,M,ID,A = md.mid_res_info(i,j,ID_list=ID_list)
    if C_cyl is None or len(C_cyl) == 0:  
        print(f'Particle not in {i} snapshot ({i} Myr)')
        return None  # Early exit to skip to the next iteration
    else:
        fit_grid,fit_pot,RPhi = pd.potential_der(i,filepath)  
        kappa_result = k.kappa(i,C_cyl,V_cyl,RPhi)
        if kappa_result is None:
            print('ERROR: kappa not calculated')
            return None #early exit (nu calculation not inititated if R_g 
            #couldn't be calculated)
        else:
            kappa_stars, R_g, L_z, L_z_galaxy, mask = kappa_result
            nu_low,nu_middle = n.nu(i,fit_grid,fit_pot,R_g)
            print('Calculating actions now!')
          
            #actions: only calculating if R_g is between 2kpc and 16kpc
            
            #radial action     #M_dot * km/s *kpc
            J_R = M[mask]*((V_cyl[mask][:,1])**2 + kappa_stars**2 * ((C_cyl[mask][:,1]-R_g)/1000)**2)/(2*kappa_stars)

            #vertical action
            J_z = M[mask]*((V_cyl[mask][:,2])**2 + np.array(nu_low)**2 * (C_cyl[mask][:,2]/1000)**2)/(2*np.array(nu_low))

            #azimuthal action 
            J_phi = M[mask]*L_z

            print(f'Returning ID,actions (in order R,phi,z),velocities,positions, masses, kappa and nu of stars at snapshot {i} Myr')
            return ID[mask], np.array((J_R,J_phi,J_z)), V_cyl[mask], C_cyl[mask], M[mask], A[mask],kappa_stars, nu_low, nu_middle, R_g, L_z, L_z_galaxy



def save_star_data_hdf5(output_file, snapshot_data, snapshot_name):
    """
    Saves star data for a single snapshot into an HDF5 file.
    
    :param output_file: Path to the HDF5 file.
    :param snapshot_data: A dictionary containing star data (e.g., 'ID', 'J', etc.).
    :param snapshot_name: The name of the snapshot (e.g., 'snapshot_0').
    """
    print(f'Saving {snapshot_name} to {output_file}')
    
    # Open the HDF5 file in append mode
    with h5py.File(output_file, 'a') as hdf:
        # Create a group for this snapshot
        grp = hdf.create_group(f'snapshots/{snapshot_name}')
        
        # Store star data for this snapshot
        grp.create_dataset('ID', data=snapshot_data['ID'], compression='gzip')
        grp.create_dataset('actions', data=snapshot_data['J'], compression='gzip')
        grp.create_dataset('velocities', data=snapshot_data['V'], compression='gzip')
        grp.create_dataset('coordinates', data=snapshot_data['C'], compression='gzip')
        grp.create_dataset('mass', data=snapshot_data['M'], compression='gzip')
        grp.create_dataset('kappa', data=snapshot_data['Kappa'], compression='gzip')
        grp.create_dataset('nu_low', data=snapshot_data['Nu_low'], compression='gzip')
        grp.create_dataset('nu_middle', data=snapshot_data['Nu_middle'], compression='gzip')
        grp.create_dataset('R_g', data=snapshot_data['R_g'], compression='gzip')
        grp.create_dataset('L_z', data=snapshot_data['L_z'], compression='gzip')
        grp.create_dataset('L_z_gal', data=snapshot_data['L_z_gal'], compression='gzip')
        grp.create_dataset('age', data=snapshot_data['A'], compression='gzip')


#########################################################################

