import stellar_luminosity
from scipy.integrate import quad
import math
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
import Kroupa_IMF
import sys

def createList(r1, r2):
    # Testing if range r1 and r2 are equal
    if (r1 == r2):
        return [r1]
    else:
        a_list = []
        while (r1 < r2 + 1):
            a_list.append(r1)
            r1 += 1
        return a_list


obs_time_list = [i * 1e7 for i in createList(1, 15)] + [2e8, 4e8, 7e8] + [i * 1e8 for i in createList(10, 25)] + [
    i * 1e9 for i in createList(3, 14)]

sys.path.append('simulation_results_from_galaxy_evol/example')
file_path = 'simulation_results_from_galaxy_evol/example/chemical_and_SN_evolution.txt'
file_evo = open(file_path, 'r')
data = file_evo.readlines()
file_evo.close()



# Initialize empty lists to hold the data
Z__list = []
simulation_time_list = []
stellar_mass_formed_at_each_epoch = []

# Open and read the file
with open(file_path, 'r') as file_evo:
    lines = file_evo.readlines()

# Helper function to find line numbers based on markers
def find_line_number(marker, lines):
    for i, line in enumerate(lines):
        if marker in line:
            return i + 1  # +1 to get the line after the marker
    return None

# Find the line numbers for the required data
SFEN__line = find_line_number('# Number of star formation event epoch (10^7 yr):', lines)
Z__line = find_line_number('# Z_list:', lines)
simulation_time_line = find_line_number('# time step list:', lines)
stellar_mass_line = find_line_number('# formed stellar mass list:', lines)

# Extract the data
SFEN = int(lines[SFEN__line])
epoch_number_list = createList(1, SFEN)

if Z__line is not None:
    Z__list = [float(x) for x in lines[Z__line].split()]

if simulation_time_line is not None:
    simulation_time_list = [float(x) for x in lines[simulation_time_line].split()]

if stellar_mass_line is not None:
    stellar_mass_formed_at_each_epoch = [float(x) for x in lines[stellar_mass_line].split()]



print("This code take the galaxy simualtion result saved in simulation_results_from_galaxy_evol/example and calculates the galaxy luminosity and mass evolution.")
print("This example galaxy form stars continuously for 1 Gyr at each 10 Myr timestep with a SFR = 100 Msun/yr.")
print("That is, in total 100 star formation epoch.")
print("The simulation record the galaxy metallicity Z__list at simulation_time_list with a length of {}.".format(len(Z__list)))
print("simulation_time_list:", simulation_time_list)
print("Z__list:", Z__list)
print("The result are given at T years after the formation of the first star in that galaxy and T is a list with length {}. T =".format(len(obs_time_list), obs_time_list))

# The photGalIMF code provides at its current stage only the Ks-band, the IRAC [3.6]-band, and the V-band ("V", "Ks", or "IRAC36")
band_selection = stellar_luminosity.band_selection
isochrones_selection = stellar_luminosity.isochrones_selection

# metallicity grid for stellar lifetime and remnant mass tables:
Z_table_list = [0.0004, 0.0008, 0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036, 0.004, 0.008, 0.012]
### If Z_table_list is not correct, use function_get_avaliable_Z() to generate Z_table_list
# def function_get_avaliable_Z():
#     # Extract available metallicity in the given grid table
#     # Stellar lifetime tables and metal production tables have different available metal grids.
#     import os
#
#     yield_path = 'yield_tables'
#     if os.path.isdir(yield_path) == False:
#         yield_path = '/galIMF/yield_tables'
#         if os.path.isdir(yield_path) == False:
#             cwd = os.getcwd()
#             yield_path = cwd + '/galIMF/yield_tables'
#
#     file_names_setllar_lifetime_from_str_yield_table = os.listdir(
#         yield_path + '/rearranged___/setllar_lifetime_from_portinari98')
#     Z_table_list = []
#     for name in file_names_setllar_lifetime_from_str_yield_table:
#         length_file_name = len(name)
#         i = 0
#         i_start = 0
#         i_end = 0
#         while i < length_file_name:
#             if name[i] == '=':
#                 i_start = i
#             if name[i] == '.':
#                 i_end = i
#             (i) = (i + 1)
#         i = i_start + 1
#         Z = ''
#         while i < i_end:
#             Z += name[i]
#             (i) = (i + 1)
#         Z_table_list += [float(Z)]
#     sorted_Z_table_list = sorted(Z_table_list)
#     return sorted_Z_table_list
# Z_table_list = function_get_avaliable_Z()
# print(Z_table_list)


def function_select_metal(Z, Z_table_list):
    # The list for the stellar lifetime is
    # [0.0004, 0.0008, 0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036, 0.004, 0.008, 0.012]
    # the list for stellar metallicity is
    # [0.0004, 0.004, 0.008, 0.0127] or [0, 0.004, 0.02] for Kobayashi2006 massive star table
    if Z <= Z_table_list[0]:
        Z_select__ = Z_table_list[0]
        return ('out', Z_select__, Z_select__, Z_select__)
        # The 'out' flag means the current gas metallicity is outside the range of the provided stellar yield table.
    elif Z >= Z_table_list[-1]:
        Z_select__ = Z_table_list[-1]
        return ('out', Z_select__, Z_select__, Z_select__)
    else:
        i = 1
        while i < len(Z_table_list):
            if Z < Z_table_list[i]:
                Z_select__low = Z_table_list[i - 1]
                Z_select__high = Z_table_list[i]
                return ('in', Z_select__low, Z, Z_select__high)
            (i) = (i + 1)


def function_mass_boundary(this_time, data_AGB):
    logAge_mass_boundary = np.round(data_AGB[:, 0], 5)
    logAge_value = np.log10(this_time)
    logAge_list_value = np.round(sorted(set(logAge_mass_boundary)), 5)
    logAge_list_index = np.argmin(np.abs(np.array(logAge_list_value) - np.round(logAge_value, 5)))
    logAge_value_extrapolated = logAge_list_value[logAge_list_index]
    index = np.where((logAge_mass_boundary == np.round(logAge_value_extrapolated, 5)))
    index = index[0]
    AGB_mass_boundary_low = 10**data_AGB[index, 2]
    AGB_mass_boundary_up = 10**data_AGB[index, 3]
    return AGB_mass_boundary_low, AGB_mass_boundary_up


def igimf_mass_function(mass, igimf_of_this_epoch):
    return igimf_of_this_epoch.custom_imf(mass, 0) * mass


# def igimf_luminous_function(mass, igimf_of_this_epoch, Z_of_this_epoch__, age_of_this_epoch):
#     xi__ = igimf_of_this_epoch.custom_imf(mass, age_of_this_epoch)
#     lum__ = stellar_luminosity.stellar_luminosity_function(mass, Z_of_this_epoch__, max(age_of_this_epoch, 1))  #######################
#     # lum__ = stellar_luminosity_interpolation.stellar_luminosity_function(mass, MH__, max(age_of_this_epoch, 1))  #######################
#     # lum__ = mass_luminosity_fit_Yan.stellar_luminosity_function(mass, MH__, max(age_of_this_epoch, 1))  #######################
#     # lum__ = stellar_luminosity.stellar_luminosity_function(mass)  #######################
#     return xi__ * lum__


def get_remnant_mass(mass_boundary, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table):
    BH_mass = 0
    NS_mass = 0
    WD_mass = 0
    if mass_boundary < 150:
        BH_mass = function_get_target_mass_in_range(max(mass_boundary, 40), 150, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table)
        if mass_boundary < 40:
            NS_mass = function_get_target_mass_in_range(max(mass_boundary, 8), 40, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table)
            if mass_boundary < 8:
                WD_mass = function_get_target_mass_in_range(max(mass_boundary, 0.1), 8, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table)
    remnant_mass = BH_mass + NS_mass + WD_mass
    return remnant_mass


def function_get_target_mass_in_range(lower_mass_limit, upper_mass_limit, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table):
    integrate_in_range = quad(integrator_for_function_get_target_mass_in_range, lower_mass_limit, upper_mass_limit, args=(igimf_of_this_epoch, mass_grid_table, Mfinal_table), limit=50)[0]
    target_mass_in_range = mass_calibration_factor * integrate_in_range
    return target_mass_in_range


def integrator_for_function_get_target_mass_in_range(initial_mass, igimf_of_this_epoch, mass_grid_table, Mfinal_table):
    mass = igimf_mass_function(initial_mass, igimf_of_this_epoch)
    mass_fraction = function_get_target_mass(initial_mass, mass_grid_table, Mfinal_table) / initial_mass
    integrator = mass * mass_fraction
    return integrator


def function_get_target_mass(initial_mass, mass_grid_table, Mfinal_table):
    if initial_mass < mass_grid_table[0] or initial_mass > mass_grid_table[-1]:
        print('Warning: function_get_remnant_mass initial_mass out of range')
        print("initial_mass=", initial_mass, "< mass grid lower boundary =", mass_grid_table[0])
    length_list_mass = len(mass_grid_table)
    x = round(length_list_mass / 2)
    i__ = 0
    low = 0
    high = length_list_mass
    if initial_mass == mass_grid_table[0]:
        x = 0
    elif initial_mass == mass_grid_table[-1]:
        x = -1
    else:
        while i__ < math.ceil(math.log(length_list_mass, 2)):
            if initial_mass == mass_grid_table[x]:
                break
            elif initial_mass > mass_grid_table[x]:
                low = x
                x = x + round((high - x) / 2)
            else:
                high = x
                x = x - round((x - low) / 2)
            (i__) = (i__ + 1)
    if mass_grid_table[x - 1] < initial_mass < mass_grid_table[x]:
        x = x - 1
    target_mass = (Mfinal_table[x] + (Mfinal_table[x + 1] - Mfinal_table[x]) * (initial_mass - mass_grid_table[x]) / (mass_grid_table[x + 1] - mass_grid_table[x]))
    return target_mass


def function_read_Mfinal(Z_select_in_table):
    if Z_select_in_table[0] == 'out':
        file_final_mass = open("stellar_remnant_mass/Portinari98/portinari98_Z={}.txt".format(Z_select_in_table[1]), 'r')
        data = file_final_mass.readlines()
        mass_ = data[3]
        Mfinal_ = data[5]
        file_final_mass.close()
        mass = [float(x) for x in mass_.split()]
        Mfinal_table = [float(x) for x in Mfinal_.split()]
    else:
        file_final_mass = open("stellar_remnant_mass/Portinari98/portinari98_Z={}.txt".format(Z_select_in_table[1]), 'r')
        data = file_final_mass.readlines()
        mass_ = data[3]
        Mfinal_low = data[5]
        file_final_mass.close()
        mass = [float(x) for x in mass_.split()]
        Mfinal_table_low = [float(x) for x in Mfinal_low.split()]
        file_final_mass = open("stellar_remnant_mass/Portinari98/portinari98_Z={}.txt".format(Z_select_in_table[3]), 'r')
        data = file_final_mass.readlines()
        Mfinal_high = data[5]
        file_final_mass.close()
        Mfinal_table_high = [float(x) for x in Mfinal_high.split()]
        x1 = Z_select_in_table[1]
        x2 = Z_select_in_table[2]
        x3 = Z_select_in_table[3]
        Mfinal_table = [y1 + (y3 - y1) * (x2 - x1) / (x3 - x1) for y1, y3 in zip(Mfinal_table_low, Mfinal_table_high)]
    return mass, Mfinal_table


data_AGB_list = []
for Z__ in stellar_luminosity.Z_list_value:
    data_AGB_list.append(np.loadtxt('stellar_isochrones/{}/{}_band/{}_{}_band_Mass_lifetime_relation_Z_{}.txt'.format(isochrones_selection, band_selection, isochrones_selection, band_selection, Z__)))


def mass_to_light(epoch_number):
    gwIMF = "igimf"
    # or gwIMF = "Kroupa"
    obs_time_list = [i * 1e7 for i in createList(1, 15)] + [2e8, 4e8, 7e8] + [i * 1e8 for i in createList(10, 25)] + [i * 1e9 for i in createList(3, 14)]
    lumi_list = [1e-9] * len(obs_time_list)
    dynamic_mass_list = [1e-9] * len(obs_time_list)
    stellar_mass_list = [1e-9] * len(obs_time_list)

    print("=== epoch_number", epoch_number)
    if gwIMF == "Kroupa":
        igimf_of_this_epoch = Kroupa_IMF
    else:
        file_name = 'igimf_epoch_{}'.format(epoch_number)
        igimf_of_this_epoch = __import__(file_name)

    #### get metallicity
    epoch_index = simulation_time_list.index(epoch_number*1e7)
    Z_of_this_epoch = Z__list[epoch_index]
    # Z_of_this_epoch = 0.04
    Z_select_in_table = function_select_metal(Z_of_this_epoch, Z_table_list)
    (mass_grid_table, Mfinal_table) = function_read_Mfinal(Z_select_in_table)

    Z_list_index = np.argmin(np.abs(np.array(stellar_luminosity.Z_list_value) - Z_of_this_epoch))
    stellar_Z_select = stellar_luminosity.Z_list_value[Z_list_index]

    data_AGB = data_AGB_list[Z_list_index]

    obs_time_number = 0
    while obs_time_number < len(obs_time_list):
        obs_time = obs_time_list[obs_time_number]
        if obs_time - (epoch_number * 1e7) > 0:
            age_of_this_epoch = max(obs_time - (epoch_number * 1e7), 3e6)
            # integrate luminosity
            (AGB_mass_boundary_low, AGB_mass_boundary_up) = function_mass_boundary(age_of_this_epoch, data_AGB)
            # # Calculate the stellar luminosity of a SSP with an interpolated metallicity.
            luminosity_SSP_MS = stellar_luminosity.SSP_luminosity(igimf_of_this_epoch, Z_of_this_epoch, max(age_of_this_epoch, 1), 0.1, AGB_mass_boundary_low*0.9, 200)
            luminosity_SSP_AGB = stellar_luminosity.SSP_luminosity(igimf_of_this_epoch, Z_of_this_epoch, max(age_of_this_epoch, 1), AGB_mass_boundary_low*0.9, AGB_mass_boundary_up, 50)
            stellar_luminosity_of_a_epoch_at_a_time_step = luminosity_SSP_MS + luminosity_SSP_AGB
            integrate_star_mass_of_a_epoch_at_a_time_step = quad(igimf_mass_function, 0.1, AGB_mass_boundary_up, args=(igimf_of_this_epoch), limit=40)[0]
            mass_calibration_factor = 1
            remnant_mass_of_this_epoch_at_a_time_step = get_remnant_mass(AGB_mass_boundary_up, mass_calibration_factor, igimf_of_this_epoch, mass_grid_table, Mfinal_table)
            if gwIMF == "Kroupa":
                lumi_list[obs_time_number] += stellar_luminosity_of_a_epoch_at_a_time_step * stellar_mass_formed_at_each_epoch[epoch_number]
                stellar_mass_list[obs_time_number] += integrate_star_mass_of_a_epoch_at_a_time_step * stellar_mass_formed_at_each_epoch[epoch_number]
                dynamic_mass_list[obs_time_number] += (integrate_star_mass_of_a_epoch_at_a_time_step + remnant_mass_of_this_epoch_at_a_time_step) * stellar_mass_formed_at_each_epoch[epoch_number]
            else:
                lumi_list[obs_time_number] += stellar_luminosity_of_a_epoch_at_a_time_step
                stellar_mass_list[obs_time_number] += integrate_star_mass_of_a_epoch_at_a_time_step
                dynamic_mass_list[obs_time_number] += (integrate_star_mass_of_a_epoch_at_a_time_step + remnant_mass_of_this_epoch_at_a_time_step)
            # print(".....", obs_time_number)
        (obs_time_number) = (obs_time_number + 1)
    return (epoch_number, lumi_list, dynamic_mass_list, stellar_mass_list, Z_of_this_epoch, stellar_Z_select)


if __name__ == '__main__':
    start_time = time.time()
    processors_number = mp.cpu_count()
    # processors_number = 10
    print("Number of processors: ", processors_number)
    pool = mp.Pool(processors_number)
    results = pool.map(mass_to_light, epoch_number_list)
    pool.close()

    Z_of_this_epoch = []
    stellar_Z_select = []

    lumi_list_sum = [1e-9] * len(obs_time_list)
    dynamic_mass_list_sum = [1e-9] * len(obs_time_list)
    stellar_mass_list_sum = [1e-9] * len(obs_time_list)

    for j in range(len(results)):
        (epoch_number, lumi_list, dynamic_mass_list, stellar_mass_list, Z_of_this_epoch_j, stellar_Z_select_j) = results[j]
        Z_of_this_epoch.insert(0, Z_of_this_epoch_j)
        stellar_Z_select.insert(0, stellar_Z_select_j)

        for i in range(len(obs_time_list)):
            lumi_list_sum[i] += lumi_list[i]
            dynamic_mass_list_sum[i] += dynamic_mass_list[i]
            stellar_mass_list_sum[i] += stellar_mass_list[i]

    print("Z_of_this_epoch =", Z_of_this_epoch)
    print("stellar_Z_select =", stellar_Z_select)

    stellar_mass_to_light_list = []
    dynamic_mass_to_light_list = []
    i = 0
    while i < len(stellar_mass_list_sum):
        stellar_mass_to_light_list.append(stellar_mass_list_sum[i] / lumi_list_sum[i])
        dynamic_mass_to_light_list.append(dynamic_mass_list_sum[i] / lumi_list_sum[i])
        (i) = (i + 1)

    computation_time_seconds = round((time.time() - start_time), 2)
    minutes, seconds = divmod(computation_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    days = int(days)
    hours = int(hours)
    minutes = int(minutes)
    seconds = round(seconds, 4)
    print("- Simulation complete. Computation time: {} d {} h {} m {} s -".format(days, hours, minutes, seconds))

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(1, figsize=(6, 5))
    fig.add_subplot(1, 1, 1)
    plt.plot([], [], c='0.5', ls='dashed', label="stellar mass")
    plt.plot([], [], c='k', label="dynamic mass")
    plt.plot([], [])
    plt.plot(obs_time_list, stellar_mass_to_light_list, c='0.5', ls='dashed', lw=4)
    plt.plot(obs_time_list, dynamic_mass_to_light_list, lw=4)
    print("stellar_mass_to_light_list =", stellar_mass_to_light_list)
    print("dynamic_mass_to_light_list =", dynamic_mass_to_light_list)
    # plt.xlim(7e6, 2e10)
    plt.ylim(7e-2, 1.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'age [yr]')
    plt.ylabel(r'$M/L~[M_\odot/L_\odot]$')
    plt.legend(prop={'size': 8})
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(2, figsize=(6, 5))
    fig.add_subplot(1, 1, 1)
    plt.plot([], [], c='0.5', ls='dashed', label="stellar mass")
    plt.plot([], [], c='k', label="dynamic mass")
    plt.plot(obs_time_list, stellar_mass_list_sum, c='0.5', ls='dashed')
    plt.plot(obs_time_list, dynamic_mass_list_sum)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'age [yr]')
    plt.ylabel(r'$M~[M_\odot]$')
    plt.legend(prop={'size': 8})
    plt.tight_layout()

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig = plt.figure(3, figsize=(6, 5))
    fig.add_subplot(1, 1, 1)

    plt.plot(obs_time_list, lumi_list_sum)
    print("lumi_list_sum =", lumi_list_sum)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'age [yr]')
    plt.ylabel(r'$L~[L_\odot]$')
    plt.tight_layout()
    plt.show()
