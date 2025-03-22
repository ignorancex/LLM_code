import h5py
import glob
import os
import numpy as np
import csv

#function to read hdf5 files and give coordinates, ID, masses, velocities, ages and potential arrays for star particles
def read_stars(fname):
    os.chdir(fname)
    coord_array=vel_array=np.zeros((1,3))
    ID_array=m_array=age_array=pot_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        coord_array = np.append(coord_array,np.array(file0['PartType4']['Coordinates'])*1000 ,axis=0) #pc
        ID_array = np.append(ID_array, np.array(file0['PartType4']['ParticleIDs']),axis=0)
        m_array = np.append(m_array, np.array(file0['PartType4']['Masses'])*1e10 ,axis=0) #M_sun
        vel_array = np.append(vel_array, np.array(file0['PartType4']['Velocities']), axis=0) #km/s
        age_array = np.append(age_array, (curTime-np.array(file0['PartType4']['StellarFormationTime']))*1000, axis =0)    #Myr
        pot_array =np.append(pot_array,np.array(file0['PartType4']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:], m_array[1:], vel_array[1:], age_array[1:], pot_array[1:]

#function to read hdf5 files and give coordinates and potential arrays for gas particles
def read_gas(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=int_energy=m_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        coord_array = np.append(coord_array,np.array(file0['PartType0']['Coordinates'])*1000 ,axis=0) #pc
        ID_array = np.append(ID_array, np.array(file0['PartType0']['ParticleIDs']),axis=0)
        V_array = np.append(V_array, np.array(file0['PartType0']['Velocities']),axis=0)
        m_array = np.append(m_array, np.array(file0['PartType0']['Masses'])*1e10 ,axis=0) #M_sun
        pot_array =np.append(pot_array,np.array(file0['PartType0']['Potential']),axis=0)  #(km/s)^2
        int_energy = np.append(int_energy, np.array(file0['PartType0']['InternalEnergy']),axis=0)
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],V_array[1:], m_array[1:],pot_array[1:],int_energy[1:]

####function to read initial stars (disk only,not bulge- bulge is parttype2')
def read_oldstars(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=m_array=age_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']
        age_array = np.append(age_array, (curTime-np.array(file0['PartType2']['StellarFormationTime']))*1000, axis =0)    #Myr
        # age_array = np.append(age_array, (curTime-np.array(file0['PartType3']['StellarFormationTime']))*1000, axis =0)
        
        coord_array = np.append(coord_array,np.array(file0['PartType2']['Coordinates'])*1000 ,axis=0) #pc
        # coord_array = np.append(coord_array,np.array(file0['PartType3']['Coordinates'])*1000 ,axis=0) #pc

        ID_array = np.append(ID_array, np.array(file0['PartType2']['ParticleIDs']),axis=0)
        # ID_array = np.append(ID_array, np.array(file0['PartType3']['ParticleIDs']),axis=0)

        V_array = np.append(V_array, np.array(file0['PartType2']['Velocities']),axis=0)
        # V_array = np.append(V_array, np.array(file0['PartType3']['Velocities']),axis=0)

        m_array = np.append(m_array, np.array(file0['PartType2']['Masses'])*1e10 ,axis=0) #M_sun
        # m_array = np.append(m_array, np.array(file0['PartType3']['Masses'])*1e10 ,axis=0) #M_sun

        pot_array =np.append(pot_array,np.array(file0['PartType2']['Potential']),axis=0)  #(km/s)^2
        # pot_array =np.append(pot_array,np.array(file0['PartType3']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],m_array[1:],V_array[1:],age_array[1:], pot_array[1:]

#function to read dark matter particles
def read_darkmatter(fname):
    os.chdir(fname)
    coord_array=V_array=np.zeros((1,3))
    ID_array= pot_array=m_array=age_array=np.zeros(1)
    for h5name in glob.glob('*.hdf5'):
        file0 = h5py.File(fname+h5name,'r')
        curTime = file0['Header'].attrs['Time']        
        coord_array = np.append(coord_array,np.array(file0['PartType1']['Coordinates'])*1000 ,axis=0) #pc

        ID_array = np.append(ID_array, np.array(file0['PartType1']['ParticleIDs']),axis=0)

        V_array = np.append(V_array, np.array(file0['PartType1']['Velocities']),axis=0)

        m_array = np.append(m_array, np.array(file0['PartType1']['Masses'])*1e10 ,axis=0) #M_sun

        pot_array =np.append(pot_array,np.array(file0['PartType1']['Potential']),axis=0)  #(km/s)^2
        file0.close()
    os.chdir('/g/data/jh2/ax8338')
    return coord_array[1:] , ID_array[1:],V_array[1:], m_array[1:],pot_array[1:]

#function to read the fitgrid 'potential' file
def read_fitgridhdf5(snapshot_number,filepath):
    '''
    function to read the the hdf5 file that contains the fit_grid and fit_pot for all snapshots.
    takes in number of the snapshot and returns the fit_grid and fit_pot arrays
    '''
    snapshot_name = f'snapshot_{snapshot_number}'
    
    try:
        with h5py.File(filepath, 'r') as hdf:
            if f'{snapshot_name}' in hdf:
                grp = hdf[f'{snapshot_name}']
                try:
                # Extracting relevant data from the snapshot
                    fit_grid = grp['fit_grid'][:]
                    fit_pot = grp['fit_pot'][:]

                except Exception as e:
                    if "Unable to synchronously open object (object 'fit_grid' doesn't exist)" in str(e):
                        fit_grid = grp['grid'][:]
                        fit_pot = grp['potential'][:]
                
                print(f"Successfully read data for {snapshot_name}")
                return fit_grid,fit_pot
            else:
                raise ValueError(f"Snapshot {snapshot_name} does not exist in the HDF5 file.")
    
    except Exception as e:
        print(f"Error reading snapshot {snapshot_number}: {e}")
        return None

   
# Define the function to read shift values from a CSV file
def read_shift_values_from_csv(file_path):
    shifts = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            shifts.append([float(x) for x in row])
    return shifts 
