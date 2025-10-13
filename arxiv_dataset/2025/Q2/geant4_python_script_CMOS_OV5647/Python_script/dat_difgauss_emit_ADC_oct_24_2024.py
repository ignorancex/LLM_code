#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 07:17:51 2024

@author: mikb

"""
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

###############################################################################
# Definir función Gaussiana 3D
def gaussian3D(x, y, z, sigma):
    """Función gaussiana 3D."""
    normalization = 1.0 / ((2.0 * np.pi) ** (3/2) * sigma ** 3)
    return normalization * np.exp(- (x * x + y * y + z * z) / (2.0 * sigma * sigma))

# Define a 2D Gaussian function
def gaussian2D(x, y, sigma):
    """Función gaussiana 2D."""
    return (1.0 / (2.0 * np.pi * sigma * sigma)) * np.exp(- (x * x + y * y) / (2.0 * sigma * sigma))

# OV5647 sensor parameters
width = 2592  # Number of pixels in width
height = 1944 # Number of pixels in height
pixel_size = 1.4  # Pixel size in micrometers
z_si = 2  # Sensor thickness in micrometers


# Constants
pair_eh = 3.6  # Energy to create an electron-hole pair in silicon (eV)
e_h_pair_energy = 3.6e-3  # Energy to create an electron-hole pair in silicon (keV)
emax = 4300  # Maximum well capacity
emin = 5     # Minimum energy threshold
#k = 0.0062  # Micrometers/keV
#k =  0.002
#alpha = 1.75  # Exponent in energy-range relationship
#zff = 0.25  # Field-free region thickness in micrometers
#nsigm = 1  # Number of sigma for Gaussian spread

# Simulation parameters
nframes = 1000  # Number of frames
z_count = 1  # Count of Z levels
source = 'Sr90'  # Radiation source (can be 'Sr90' or 'Cs137')
source = 'Cs137'

if(source == 'Sr90'):
    winback = '254'
    fecha = '2023_Oct_24_23h'
    ev_actv = 1514
if(source == 'Cs137'):
    winback = '635'
    fecha = '2023_Oct_23_23h'
    ev_actv = 3811
'''  
if(source == 'Sr90'):
    winback = '254'
    fecha = '2021_Nov_09'
    ev_actv = 1587
if(source == 'Cs137'):
    winback = '635'
    fecha = '2021_Nov_23'
    ev_actv = 3991
'''

evt = winback+'_ev'+str(ev_actv)

level_z = list(range(z_count))
#level_z = [0]
path_main = 'C:/dat_2023/emit/'
path_main = '/home/milton/g4_works/data_2023/data_sim_2024_11/'
path_main = '/home/mbonnett/mik/dat_sim_2024/g4_'+source+'_2024/'
#path_name = 'data_'+source+'_pixl_thickZ_2um_' + str(z_count)+'level_'+str(winback)+'_100um_ev'+str(ev_actv)+'_emit/'
#path_main = 'C:\data2023\data_sim_2024_11/'
#path_name = f'data_{source}_pixl_thickZ_2um_{z_count}level_{winback}_100um_ev{ev_actv}_emit/'

# Conditions for energy deposition
condit_edep = ''  # '', '_dif0', '_zero' or other conditions can be added here
#condit_edep = '_dif0'

# List of particle types
lista = [
    #'electron',
    #'anti_nu_e',
    #'gamma',
    'particle',
    ]

emax_adc = 0

gauss_dist = 'Dg2'

for sim_n in range(10):
    sim_n ='_s'+str(sim_n)
    #sim_n ='_0'
    path_name = f'data_{source}_pixl_thickZ_2um_{z_count}level_{winback}_100um_ev{ev_actv}'+sim_n+'/'

#sim_n = ''
#path_name = f'data_{source}_pixl_thickZ_2um_{z_count}level_{winback}_100um_ev{ev_actv}'+sim_n+'/'
# Loop over all particle types
#for partic_name in lista:
    partic_name ='particle'
    # Number of sigma for Gaussian spread
    for nsigm in [0, ]:
        #for k in np.arange(0.002, 0.064, 0.012):
        for k in [0]:
            k=np.round(k,3)
            k_str = str(int(1000*np.round(k,3)))
            #i+=1; print(i, np.round(k,3))
            #for alpha in np.arange(1.73, 1.775, 0.005):
            for alpha in [0]:
                alpha=np.round(alpha,3)
                a_str = str(int(1000*np.round(alpha,3)))
                #print(np.round(alpha,3))
                # Field-free region thickness in micrometers
                #for zff in np.arange(0.2, 1.1, 0.2):
                for zff in [0]:
                    zff = np.round(zff,3)
                    if nsigm > 0 :
                        gaus_dif = f"{gauss_dist}_{k}k_{alpha}a_{nsigm}sig_{zff}zff"
                       
                    if nsigm == 0 :
                        gaus_dif = f"{gauss_dist}_{k}k_{alpha}a_{zff}zff"
                        
                    # Listas para almacenar sigma_i y sigma_ff
                    sigma_i_list = []
                    sigma_ff_list = []
                    sigma_total_list = []

                    for z in level_z:
                        nameframe = f'E_sim_z{str(2*z).zfill(3)}_f'  #filename format
                        # Define data path based on source and conditions
                        path = f'data_{source}_pixl_thickZ_2um_distSD_{2*z}mm_{z_count}level_csv_tar_bz/'
                        #path = 'data_'+source+'_pixl_thickZ_2um_'+'distSD_' + str(z)+'mm_'+str(z_count)+'level'+'_csv_tar_bz'+'/'

                        if(partic_name == 'particle' or partic_name == 'electron' or partic_name == 'anti_nu_e'or partic_name == 'gamma'):
                            #type_partic = '_edep'+condit_edep+'_all_'+partic_name+'_arrived_detector'
                            type_partic = f'_edep{condit_edep}_all_{partic_name}_arrived_detector'
                            if(partic_name == 'particle'):
                                colum = [0,1, 2, 12,13,14]
                                #colum = [0, 1, 2, 7, 8, 9] # data 2023
                            else:colum = [4,7,9]
                        # Directories for saving simulation data
                        dir_filesave = f"{path_main}data_{source}_pixZ_2um_{z_count}l_{winback}um_ev{ev_actv}{sim_n}{condit_edep}"
                        #os.makedirs(dir_filesave, exist_ok=True)
                        if(emax_adc == 0):
                            dir_filesave_gauss = f"{dir_filesave}_{gaus_dif}"
                        if(emax_adc > 0):
                            dir_filesave_gauss = f"{dir_filesave}_{gaus_dif}"+"_E"+str(15)
                        os.makedirs(dir_filesave_gauss, exist_ok=True)
                        os.makedirs(dir_filesave_gauss+'/plot_conser', exist_ok=True)

                        eng_consr =  np.zeros(nframes)

                        # Loop over frames
                        for n in range(nframes):
                        #for n in range(0,1000):
                            #tmpfile = path_main+path_name+path+'pixel' + \
                            #    str(n)+'_SD'+str(z)+'mm_'+source+type_partic
                            tmpfile = f"{path_main}{path_name}{path}pixel{n}_SD{2*z}mm_{source}{type_partic}"
                            dat = np.loadtxt(tmpfile + '.csv', delimiter='\t', skiprows=1, usecols=colum)

                            # Extract data from CSV simulated data from Geant4
                            #pxlX, pxlY, edep, posX, posY, posZ = dat.T  # Usar transposición para extraer las columnas
                            pxlX = dat[:, 0]  # Pixel X position in array
                            pxlY = dat[:, 1]  # Pixel Y position in array
                            edep = dat[:, 2]  # Energy deposited in keV
                            posX = dat[:, 3]  # X position in micrometers
                            posY = dat[:, 4]  # Y position in micrometers
                            posZ = dat[:, 5]  # Z position in micrometers

                            # Initialize pixel matrices with 0
                            pixel_matrix_org =  np.zeros((height, width))
                            pixel_matrix_tesg =  np.zeros((height, width))
                            pixel_matrix_gauss =  np.zeros((height, width))

                            # Process each data point
                            for i in range(dat[:,0].size):
                            #for i in range(200,400,1):
                                pixel_matrix_org[int(pxlY[i]), int(pxlX[i])] += edep[i]
                                z0 = posZ[i] - posZ.min()

                                if edep[i] > 0 and z0 >=0 :
                                    #if zff > 0 and z0 <= 2*zff and .2:
                                    if zff > 0 and z0 <= 2*zff :
                                        # Calculate initial charge cloud radius
                                        if gaus_dif != 'Dg2_0k_0a_0zff' : sigma_i = k * (edep[i])**alpha
                                        if gaus_dif == 'Dg2_0k_0a_0zff' :
                                            if(edep[i] <= emax_adc):
                                                sigma_i = 0
                                            if(edep[i] > emax_adc):
                                                sigma_i =  k * (edep[i])**alpha
                                                #print(sigma_i)
                                        # Aplicar la restricción: sigma_i no debe ser mayor que zff
                                        #if sigma_i > z_si:  sigma_i = z_si  # Limitar sigma_i al valor máximo permitido

                                        # Calculate Gaussian width due to diffusion
                                        #if zff > 0: sigma_ff = 0.5 * np.sqrt(z0 * (2.0*zff - z0))
                                        #if zff == 0: sigma_ff = 0.
                                        sigma_ff = 0.5 * np.sqrt(z0 * (2.0*zff - z0))
                                        #sigma_ff =0
                                        # Total Gaussian width
                                        sigma_total = np.sqrt(sigma_i**2 + sigma_ff**2)

                                        # Limitar sigma_total a un máximo de 2 micrómetros
                                        if sigma_total > z_si:
                                            sigma_total = z_si

                                        # Guardar los valores de sigma
                                        sigma_i_list.append(sigma_i)
                                        sigma_ff_list.append(sigma_ff)
                                        sigma_total_list.append(sigma_total)

                                        # Determine range of pixels for diffusion
                                        #if nsigm > 0 : range_pixels = math.ceil(nsigm * sigma_total) # Round up to cover enough area
                                        #if nsigm == 0 : range_pixels = 1
                                        range_pixels = math.ceil(nsigm * sigma_total) if nsigm > 0 else 1

                                        # Charge cloud parameters
                                        x_center = pxlX[i] * pixel_size  # Center x coordinate of charge cloud in micrometers
                                        y_center = pxlY[i] * pixel_size  # Center y coordinate of charge cloud in micrometers
                                        edep_total = edep[i]  # Total energy in the cloud

                                        """Distribuir la energia en la matriz de píxeles."""
                                        # Define the limits of the calculation region
                                        x_start = max(0, int(pxlX[i] - range_pixels))
                                        x_end = min(width, int(pxlX[i] + range_pixels))
                                        y_start = max(0, int(pxlY[i] - range_pixels))
                                        y_end = min(height, int(pxlY[i] + range_pixels))

                                        # Distribute energy in the pixel grid using Gaussian diffusion
                                        total_fraction = 0
                                        resd_e_dif = edep_total
                                        if sigma_total >0 :
                                            for x in range(x_start, x_end):
                                                for y in range(y_start, y_end):
                                                    # Calculate dx, dy coordinates distances from the center of the neighboring pixel in micrometers
                                                    dx = (x + 0.5) * pixel_size - x_center
                                                    dy = (y + 0.5) * pixel_size - y_center
                                                    dz = z0
                                                    if gauss_dist == 'Dg2' : edep_fraction = edep_total*gaussian2D(dx, dy, sigma_total)
                                                    #if gauss_dist == 'Dg3' :edep_fraction = edep_total*gaussian3D(dx, dy, dz, sigma_total)
                                                    resd_e_dif = (resd_e_dif-edep_fraction)
                                                    total_fraction += np.copy(edep_fraction)
                                                    # Accumulate the energy distributed in neighboring pixels
                                                    pixel_matrix_gauss[y, x] += edep_fraction
                                                    pixel_matrix_tesg[y, x] +=  edep_fraction

                                            # Accumulate residual energy in the central pixel
                                            pixel_matrix_gauss[int(pxlY[i]), int(pxlX[i])] += resd_e_dif
                                            pixel_matrix_tesg[int(pxlY[i]), int(pxlX[i])]  += edep_total - total_fraction
                                        else:
                                            # Accumulate residual energy in the central pixel
                                            pixel_matrix_gauss[int(pxlY[i]), int(pxlX[i])] += edep[i]
                                            pixel_matrix_tesg[int(pxlY[i]), int(pxlX[i])]  += edep[i]

                                    else:
                                        # Accumulate residual energy in the central pixel
                                        pixel_matrix_gauss[int(pxlY[i]), int(pxlX[i])] += edep[i]
                                        pixel_matrix_tesg[int(pxlY[i]), int(pxlX[i])]  += edep[i]

                            # Verify energy conservation
                            eng_consr[n] = pixel_matrix_org.sum() - pixel_matrix_gauss.sum()
                            print(f"{n} Energy conservation: {pixel_matrix_org.sum() - pixel_matrix_gauss.sum()}")
                            print(n, pixel_matrix_org.sum(), pixel_matrix_gauss.sum(), pixel_matrix_tesg.sum())

                            # Save the original simulated and Gaussian diffusion matrices in npz file
                            #np.savez_compressed(f"{dir_filesave}/{nameframe}{n:03d}", np.float64(pixel_matrix_org))
                            print (f"{dir_filesave}/{nameframe}{n:03d}")
                            np.savez_compressed(f"{dir_filesave_gauss}/{nameframe}{n:03d}", np.float64(pixel_matrix_gauss))
                            print (f"{dir_filesave_gauss}/{nameframe}{n:03d}")
                            #plt.imshow(pixel_matrix_org, norm=LogNorm(), cmap=plt.cm.rainbow)
                            #plt.show()
                            #plt.imshow(pixel_matrix_gauss, norm=LogNorm(), cmap=plt.cm.rainbow)
                            #plt.show()

                        # Plot energy conservation histogram
                        plt.hist(eng_consr, bins=1000, histtype='step', label='Energy Conservation')
                        plt.xticks(size=14)
                        plt.yticks(size=14)
                        plt.xlabel('Energy Difference', fontsize=14)
                        plt.title('Histogram Energy Conser ' + f"{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                        plt.legend(loc='upper center', fontsize=12)
                        plt.savefig(f"{dir_filesave_gauss}/plot_conser/plot_energy_{gaus_dif}_{nsigm}sig_{zff}zff"+'.png', dpi=150)
                        #plt.show()

                    # Guardar las listas sigma_i y sigma_ff en archivos .npy y .txt
                    np.save(f"{dir_filesave_gauss}/plot_conser/sigma_i_list_{gaus_dif}_{nsigm}sig_{zff}zff.npy", sigma_i_list)
                    np.save(f"{dir_filesave_gauss}/plot_conser/sigma_ff_list_{gaus_dif}_{nsigm}sig_{zff}zff.npy", sigma_ff_list)
                    np.save(f"{dir_filesave_gauss}/plot_conser/sigma_total_list_{gaus_dif}_{nsigm}sig_{zff}zff.npy", sigma_total_list)
                    np.savetxt(f"{dir_filesave_gauss}/plot_conser/sigma_i_list_{gaus_dif}_{nsigm}sig_{zff}zff.txt", sigma_i_list, delimiter=',')
                    np.savetxt(f"{dir_filesave_gauss}/plot_conser/sigma_ff_list_{gaus_dif}_{nsigm}sig_{zff}zff.txt", sigma_ff_list, delimiter=',')
                    np.savetxt(f"{dir_filesave_gauss}/plot_conser/sigma_total_list_{gaus_dif}_{nsigm}sig_{zff}zff.txt", sigma_total_list, delimiter=',')

                    # Graficar histograma sigma_i
                    lista_filtrada = [num for num in sigma_ff_list if num > 0]

                    plt.figure(figsize=(10, 6))
                    plt.hist(sigma_i_list, bins=1000, histtype='step'
                                , density=False, color='blue',  alpha=0.5, linewidth=2, label='$\sigma_i$')

                    # Set logarithmic scale for both x and y axes
                    #plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('$\sigma$ value (micrometers)', fontsize=14)
                    plt.ylabel('Frequency', fontsize=14)
                    plt.title('Histogram $\sigma_i$ ' + f"{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.savefig(f"{dir_filesave_gauss}/plot_conser/histos_sigma_i_{gaus_dif}_{nsigm}sig_{zff}zff.png", dpi=150)
                    #plt.show()

                    # Graficar histograma sigma_ff
                    lista_filtrada = [num for num in sigma_ff_list if num > 0]

                    plt.figure(figsize=(10, 6))
                    plt.hist(sigma_ff_list, bins=100, histtype='step'
                                , density=False, color='green', alpha=1, linewidth=2, label='$\sigma_{ff}$')

                    # Set logarithmic scale for both x and y axes
                    #plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('$\sigma$ value (micrometers)', fontsize=14)
                    plt.ylabel('Frequency', fontsize=14)
                    plt.title('Histograma $\sigma_{ff}$'+ f"{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.savefig(f"{dir_filesave_gauss}/plot_conser/histos_sigma_ff_{gaus_dif}_{nsigm}sig_{zff}zff.png", dpi=150)
                    #plt.show()

                    # Graficar histograma sigma_total
                    plt.figure(figsize=(10, 6))
                    plt.hist(sigma_total_list, bins=100, histtype='step'
                                , density=False, color='red', alpha=0.5, linewidth=2, label='$\sigma_{total}$')

                    # Set logarithmic scale for both x and y axes
                    #plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('$\sigma$ value (micrometers)', fontsize=14)
                    plt.ylabel('Frequency', fontsize=14)
                    plt.title('Histogram $\sigma_{total}$'+ f"{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.savefig(f"{dir_filesave_gauss}/plot_conser/histos_sigma_total_{gaus_dif}_{nsigm}sig_{zff}zff.png", dpi=150)
                    #plt.show()

                    if zff > 0 :
                        # Graficar histogramas de sigma_i y sigma_ff
                        lista_filtrada = [num for num in sigma_ff_list if num > 0]

                        intervalo_inferior = 0
                        intervalo_superior = 3*max(sigma_ff_list)
                        lista_interv_i = [num for num in sigma_i_list if intervalo_inferior <= num <= intervalo_superior]
                        lista_interv_t = [num for num in sigma_total_list if intervalo_inferior <= num <= intervalo_superior]


                        # Obtener el rango mínimo y máximo de todos los datos
                        data = np.concatenate([sigma_ff_list, sigma_i_list, sigma_total_list])
                        data = np.concatenate([sigma_ff_list, lista_interv_i, lista_interv_t])
                        min_val = np.min(data)
                        max_val = np.max(data)

                        # Especificar el número deseado de bins (ajusta según tus necesidades)
                        num_bins = 100

                        # Crear los bins con un rango uniforme
                        bins = np.linspace(min_val, max_val, num_bins + 1)


                        plt.figure(figsize=(10, 6))
                        plt.hist(lista_interv_i, bins=bins, histtype='step'
                                    , density=False, color='blue', alpha=0.5, linewidth=2, label='$\sigma_i$')
                        plt.hist(sigma_ff_list, bins=bins, histtype='step'
                                    , density=False, color='green', alpha=1, linewidth=2, label='$\sigma_{ff}$')

                        # Set logarithmic scale for both x and y axes
                        #plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel('$\sigma$ value (micrometers)', fontsize=14)
                        plt.ylabel('Frequency', fontsize=14)
                        plt.title('Histograms $\sigma_i$ y $\sigma_{ff}$' + f"{gaus_dif}_{nsigm}sig_{zff}zff"+ "{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                        plt.legend(fontsize=14)
                        plt.savefig(f"{dir_filesave_gauss}/plot_conser/histos_sigmas_i_ff_{gaus_dif}_{nsigm}sig_{zff}zff.png", dpi=150)
                        #plt.show()

                        # Graficar histogramas de sigma_i, sigma_ff y sigma_total
                        plt.figure(figsize=(10, 6))
                        plt.hist(lista_interv_i, bins=bins, histtype='step'
                                    , density=False,  color='blue', alpha=1, linewidth=2, label='$\sigma_i$')
                        plt.hist(sigma_ff_list, bins=bins, histtype='step'
                                    , density=False,  color='green', alpha=1, linewidth=2, label='$\sigma_{ff}$')
                        plt.hist(lista_interv_t, bins=bins, histtype='step'
                                    , density=False,  color='red', alpha=1, linewidth=2, label='$\sigma_{total}$')

                        # Set logarithmic scale for both x and y axes
                        #plt.xscale('log')
                        plt.yscale('log')
                        plt.xlabel('$\sigma$ value (micrometers)', fontsize=14)
                        plt.ylabel('Frequency', fontsize=14)
                        plt.title('Histograms $\sigma_i$, $\sigma_{ff}$ & $\sigma_{total}$' + f"{gaus_dif}_{nsigm}sig_{zff}zff", fontsize=16)
                        plt.legend(fontsize=12)
                        plt.savefig(f"{dir_filesave_gauss}/plot_conser/histos_sigma_i_ff_total_{gaus_dif}_{nsigm}sig_{zff}zff.png", dpi=150)
                        #plt.show()
