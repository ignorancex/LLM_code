import numpy as np

from TOVsolver import unit
from TOVsolver.main import OutputMR

def maximum_central_density(energy_density, pressure, 
                             rho_lo=10**14.3 * unit.g_cm_3, rho_up=10**15.6 * unit.g_cm_3, 
                             tol_for_mass=1e-3, tol_for_rho=1e-7, threshold_R=40*unit.km):
    """
    Find the center density of a stable point using dichotomy approach
    Args:
        energy_density (numpy 1Darray): Density of EoS
        pressure (numpy 1Darray): Pressure of EoS
        rho_lo (float): The lower bound of central density
        rho_up (float): The upper bound of central density
        tol_for_mass (float): Relative tolerance of the mass
        tol_for_rho (float): Relative tolerance of the density
        threshold_R (float): Threshold of Radius. Once the minmum Radius of the star is lager than the value of threshold, it means the results are very unphysical
    Returns:
        density (float): The maxium central density, in g_cm_3
        mass_large (float): The maximum mass, in mass of sun
        radius_small (float): The corresponding radius of the maximum mass, in km
    """
    # Store R_min to roughly chech whether the result is physical
    R_min = 0
    ###### The first time *****
    # Ms and cumul_rho2s are used to store points which are closest to the peak of mass
    Ms = []
    R_min_list = []
    Peak = False # Peak Used to store the boolean value indicating whether a peak has been found
    Maximum_l_endpoint = False # Used to store the boolean value indicating whether the peak is at the left boundary of rhos
    Maximum_r_endpoint = False # Used to store the boolean value indicating whether the peak is at the right boundary of rhos
    Catch = 0 # Used to store the index of the peak in each search
    rhos = np.geomspace(rho_lo, rho_up, 5) # The densities to be calculated
    for i, rho in enumerate(rhos):
        # The positions of the first five points are very sparse, so using query_nearest-values directly to shorten the interval is faster
        result = OutputMR('', energy_density, pressure, [rho])[0]
        Ms.append(result[0])

        # print(Ms[-1])

        R_min_list.append(result[1])

    # Look for peaks from large to small Mass, regardless of valley values
    for i in range(5)[::-1]:
        if i==4 and Ms[i] >= Ms[i-1]:
            Maximum_r_endpoint =True
            Catch = i
            R_min = R_min_list[Catch]
            break
        elif i!=0 and i!=4 and Ms[i] >= Ms[i-1] and Ms[i] >= Ms[i+1]:
            Peak = True
            Catch = i
            R_min = R_min_list[Catch]
            break
        elif i==0 and Ms[i] >= Ms[i+1]:
            Maximum_l_endpoint =True
            Catch = i
            R_min = R_min_list[Catch]
            break
    # print('R_min',R_min/unit.km)
    # print('first five times')

    ####### Reinitialize *****
    Ms_larg =  -1*unit.Msun
    if Peak == True:
        Ms_larg = Ms[Catch] # Maximum value
        # Length is 3
        Ms = Ms[Catch-1 : Catch+2] # Update the peak and the two adjacent points
        # Length is 3
        store_rho_bound = [rhos[Catch-1], rhos[Catch+1]] # Store_rho-bound is the point on either side of the peak value
    elif Maximum_r_endpoint == True:
        Ms_larg = Ms[Catch] # Maximum value
        # Length is 2
        Ms = Ms[Catch-1:] # Update peak and a nearby point
        store_rho_bound = rhos[Catch-1:] # Store_rho-bound is the peak value and a point next to it
    elif Maximum_l_endpoint == True:
        Ms_larg = Ms[Catch] # Maximum value
        # Length is 2
        Ms = Ms[:Catch+1] # Update peak and a nearby point
        store_rho_bound = rhos[:Catch+1] # Store_rho-bound is the peak value and a point next to it
    # print(Peak, Maximum_r_endpoint, Maximum_l_endpoint)

    # If R_min > threshold_R, then the results are definitely not physical.
    if R_min>threshold_R:
        # print('R',R_min/unit.km)
        return rhos[Catch]/unit.g_cm_3, Ms_larg/unit.Msun, R_min/unit.km # density in g_cm3, mass in sun, radius in km
    
    ###### Start looping code *****
    # Update and refine the central density range, and ultimately return the central density of the maximum points
    while ((Ms_larg - Ms[0]) > tol_for_mass*Ms_larg) and ((store_rho_bound[1] - store_rho_bound[0]) > tol_for_rho*store_rho_bound[0]): # Stop calculating when the relative difference is less than tol_for_mass and tol_for_rho
        rhos = np.geomspace(store_rho_bound[0], store_rho_bound[1], 5) # This is an expanded density array that includes already calculated points
        # print(np.log10(rhos/unit.g_cm_3))
        # print(Peak, Maximum_r_endpoint, Maximum_l_endpoint)
        # Note that when Peak==True, points 0, 2, and 4 in rhos have already been calculated, so there is no need to traverse them. 
        # When Peak==False, the index has already been calculated for points 0 and 4, so there is no need to traverse them
        if Peak == True:
            for i in [1,3]:
                # be careful! The indexes 0, 2, and 4 in Rhos have already been calculated
                if i == 1:
                    rho = rhos[1]
                    # Note that the useful accumulation amounts here are 0 and 2 points, and the point to be solved is in the middle, so we take the value of Cumul_rho2s
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(1, result[0]) # Insert points into the Ms of the original 3 points

                    # Firstly, we store the data, and secondly, we judge the data
                    if Ms[1] >= Ms[2]:
                        Catch = 1
                        break
                else:
                    rho = rhos[3]
                    # Note that the useful accumulation amounts here are 0, 1, and 2 points, and the point to be solved is on the right side, so we take Cumul_rho2s [: 3]
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    # Since the refinement calculation is carried out on the basis of the previous stage, if d2_min is continuously performed, The operation of d2_max=query_nearest-values (* paras_for_strink_0) is actually cumbersome
                    Ms.insert(3, result[0]) # Insert a point into Ms with 4 points

                    if Ms[3] >= Ms[2]:
                        Catch = 3
                    else:
                        Catch = 2
        elif Maximum_r_endpoint == True:
            for i in [1,2,3]:
                # print(i)
                if i == 1:
                    rho = rhos[1]
                    # Note that the useful accumulation amounts here are 0 and 4 points, and the point to be calculated is on the left side of the middle, so we take Cumul_rho2s
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(1, result[0])

                elif i == 2:
                    rho = rhos[2]
                    # Note that the useful accumulation amounts here are 0 and 1 points, and the point to be solved is on the right side, so we take Cumul_rho2s [: 2]
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(2, result[0])

                    if Ms[1] >= Ms[2]:
                        Peak == True
                        Catch = 1
                        break
                elif i == 3:
                    rho = rhos[3]
                    # Note that the useful accumulation amounts here are 0, 1, and 2 points, and the point to be solved is on the right side, so we take Cumul_rho2s [: 3]
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(3, result[0])

                    if Ms[2] >= Ms[3]:
                        Peak == True
                        Catch = 2
                        break
                    elif Ms[3] >= Ms[4]:
                        Peak == True
                        Catch = 3
                        break
                    else:
                        Catch = 4
                        break
        elif Maximum_l_endpoint == True:
            # This is the least probable thing
            # print('Maximum_l_endpoint == True')
            # print('rhos:',(np.array(rhos)/unit.g_cm_3))
            for i in [1,2,3]:
                if i == 1:
                    rho = rhos[1]
                    # Note that the useful accumulation amounts here are 0 and 4 points, and the point to be calculated is on the left side of the middle, so we take Cumul_rho2s
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(1, result[0])

                elif i == 2:
                    rho = rhos[2]
                    # Note that the useful accumulation amounts here are 0 and 1 points, and the point to be solved is on the right side, so we take Cumul_rho2s [: 2]
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(2, result[0])

                    if (Ms[1] >= Ms[2]) and (Ms[1] > Ms[0]):
                        Peak == True
                        Catch = 1
                        break
                elif i == 3:
                    rho = rhos[3]
                    # Note that the useful accumulation amounts here are 0, 1, and 2 points, and the point to be solved is on the right side, so we take Cumul_rho2s [: 3]
                    result = OutputMR('', energy_density, pressure, [rho])[0]
                    Ms.insert(3, result[0])

                    if (Ms[2] >= Ms[3]) and (Ms[2] > Ms[1]):
                        Peak == True
                        Catch = 2
                        break
                    elif (Ms[3] >= Ms[4]) and (Ms[3] > Ms[2]):
                        Peak == True
                        Catch = 3
                        break
                    else:
                        Catch = 0
                        break
        # print('rhos:',(np.array(rhos)/unit.g_cm_3))
        # print('Ms',np.array(Ms)/unit.Msun)
        if Peak == True:
            Ms_larg = Ms[Catch]
            Ms = Ms[Catch-1 : Catch+2]
            store_rho_bound = [rhos[Catch-1], rhos[Catch+1]]
        elif Maximum_r_endpoint == True:
            Ms_larg = Ms[Catch]
            Ms = Ms[Catch-1 : ]
            store_rho_bound = rhos[Catch-1 : ]
        elif Maximum_r_endpoint == True:
            Ms_larg = Ms[Catch]
            Ms = Ms[ : Catch+1]
            store_rho_bound = rhos[ : Catch+1]
        
        # print('Ms_larg2',Ms_larg/unit.Msun)
        # print('Peak:',Peak)
        # print('Ms',np.array(Ms)/unit.Msun)

    return rhos[Catch]/unit.g_cm_3, Ms_larg/unit.Msun, R_min/unit.km