import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import os
import sys
import inspect
from scipy.interpolate import interp1d

currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir=os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import model_main
import initialization

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu') # ensures we use the CPU
jax.config.update("jax_debug_nans", False)

# GO AT THE BOTTOM TO MANAGE THE STUDY
# DONT FORGET TO PUT CRRT ON

def blockPrint():
    # Disable printing
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    # Restore printing
    sys.stdout = sys.__stdout__

def fitted_curve(x, y, degree=3):
    p = np.polyfit(x, y, degree)
    #print(np.polyval(p,x))
    return np.polyval(p, x)

def convert_index_to_access(index, CRRT=False):

    compartmentDataRaw = np.genfromtxt('Access/access_list.txt', dtype=[('compartment','U20'), ('index',int)])
    compartmentData = {k:d for k,d in zip(compartmentDataRaw['index'], compartmentDataRaw['compartment'])}
    
    if CRRT==True:
        locationRaw = np.genfromtxt('Access/CRRT_access_list.txt', delimiter=",",
                                    dtype=[('compartment','U20'), ('location', 'U20')])
        location = {k:d for k,d in zip(locationRaw['compartment'], locationRaw['location'])}

        return location[compartmentData[index]]

    return compartmentData[index]

def convert_num_to_access(combinations):

    combiNames = []
    
    for i in range(len(combinations)):
        CRRTdraintmp = convert_index_to_access(int(float(combinations[i].split('-')[0])), CRRT=True)
        CRRTreturntmp = convert_index_to_access(int(float(combinations[i].split('-')[1])), CRRT=True)

        combiNames.append(CRRTdraintmp + '\n' + CRRTreturntmp)

    return combiNames

def mean_over_time(timeSteppedQuantity):
    return np.mean(timeSteppedQuantity)

def min_over_time(timeSteppedQuantity):
    return np.min(timeSteppedQuantity)

def max_over_time(timeSteppedQuantity):
    return np.max(timeSteppedQuantity)

def find_deviation_less(mean, min):
    return abs((mean-min)/mean)*100

def find_deviation_plus(mean, max):
    return abs((max-mean)/mean)*100

def mean_over_combinations(combinationsList, quantityName, timeSteppedQuantity):
    # For each combination we have a time step values
    # Quantity example pressure: Qtity0, Qtity1, Qtity2, Qtity3 0, 1, 2, 3 for combination
    #print('In mean for ', quantityName)
    totalQuantity=np.zeros(len(timeSteppedQuantity[quantityName+'0']))
    for i in range(len(combinationsList)):
        totalQuantity=totalQuantity+timeSteppedQuantity[quantityName+str(i)]
    #print('Output mean for ', quantityName, '=', totalQuantity/(i+1))
    return totalQuantity/(i+1)

def min_over_combinations(combinationsList, quantityName, timeSteppedQuantity): 
    # We define the min as the first one
    minCombination=timeSteppedQuantity[quantityName+'0']
    for i in range(len(combinationsList)):
        if (timeSteppedQuantity[quantityName+str(i)]<minCombination).all():
            minCombination=timeSteppedQuantity[quantityName+str(i)]
    return minCombination

def max_over_combinations(combinationsList, quantityName, timeSteppedQuantity): 
    # We define the max as the first one
    # TODO: CHANGE TO NEW VERSION HERE!
    maxCombination=timeSteppedQuantity[quantityName+'0']
    for i in range(len(combinationsList)):
        if (timeSteppedQuantity[quantityName+str(i)]>maxCombination).all():
            maxCombination=timeSteppedQuantity[quantityName+str(i)]
    return maxCombination

    #maxCombinationValue = timeSteppedQuantity[quantityName+'0']
    #maxCombinationName = [combinationsList[0]]
    #for i in range(len(combinationsList)):
    #    if (timeSteppedQuantity[quantityName+str(i)]>maxCombinationValue).all():
    #        maxCombinationValue = timeSteppedQuantity[quantityName+str(i)]
    #        maxCombinationName = [combinationsList[i]]
    #        
    #maxCombinationName = convert_num_to_access(maxCombinationName)

    #return [maxCombinationName, maxCombinationValue]

def plot_Psart_Psvn_Ppart_mean_and_deviation(x, xlabel,
                                             meanPsart, minPsart, maxPsart,
                                             meanPsvn, minPsvn, maxPsvn,
                                             meanPpart, minPpart, maxPpart):
    """
    Plots the mean and deviation of Psart, Psvn, and Ppart.

    Args:
        x (array-like): The x-axis values.
        xlabel (str): The label for the x-axis.
        meanPsart (array-like): The mean values of Psart.
        minPsart (array-like): The minimum values of Psart.
        maxPsart (array-like): The maximum values of Psart.
        meanPsvn (array-like): The mean values of Psvn.
        minPsvn (array-like): The minimum values of Psvn.
        maxPsvn (array-like): The maximum values of Psvn.
        meanPpart (array-like): The mean values of Ppart.
        minPpart (array-like): The minimum values of Ppart.
        maxPpart (array-like): The maximum values of Ppart.

    Example:
        plot_Psart_Psvn_Ppart_mean_and_deviation(TimeNormalized, 'Normalized cardiac cycle',
            MeasurementP['PsartStats'][0], MeasurementP['PsartStats'][1], MeasurementP['PsartStats'][2],
            MeasurementP['PsvnStats'][0], MeasurementP['PsvnStats'][1], MeasurementP['PsvnStats'][2],
            MeasurementP['PpartStats'][0], MeasurementP['PpartStats'][1], MeasurementP['PpartStats'][2])

        plot_Psart_Psvn_Ppart_mean_and_deviation(jnp.array(MeasurementQrpm['Qecmopump']['Global']['mean'])*0.06, 'ECMO pump flow in L/min',
            MeasurementPrpm['Psart']['Global']['mean'], MeasurementPrpm['Psart']['Global']['min'], MeasurementPrpm['Psart']['Global']['max'],
            MeasurementPrpm['Psvn']['Global']['mean'], MeasurementPrpm['Psvn']['Global']['min'], MeasurementPrpm['Psvn']['Global']['max'],
            MeasurementPrpm['Ppart']['Global']['mean'], MeasurementPrpm['Ppart']['Global']['min'], MeasurementPrpm['Ppart']['Global']['max'])

    """

    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']=10
    plt.rcParams["figure.figsize"]=[16/2.54, 13.6/2.54]
    
    plt.figure()
    # Psart - Psvn, Ppart means and deviations
    plt.plot(x, meanPsart, linewidth=2.0, color='red', label='Arterial')
    plt.fill_between(x, minPsart, maxPsart, alpha=0.2, color='red')
    plt.plot(x, minPsart, "r--")
    plt.plot(x, maxPsart, "r--")

    plt.plot(x, meanPsvn, linewidth=2.0, color='blue', label='Venous')
    plt.fill_between(x, minPsvn, maxPsvn, alpha=0.2, color='blue')
    plt.plot(x, minPsvn, "b--")
    plt.plot(x, maxPsvn, "b--")

    plt.plot(x, meanPpart, linewidth=2.0, color='green', label='Pulm. artery')
    plt.fill_between(x, minPpart, maxPpart, alpha=0.2, color='green')
    plt.plot(x, minPpart, "g--")
    plt.plot(x, maxPpart, "g--")

    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=3)
    plt.tight_layout()
    #plt.show()

    #plt.savefig('Analyses/CRRT_Study_patient_pressures.svg')
    #plt.savefig('Analyses/CRRT_Study_patient_pressures.png')

def plot_ventricular_pV_Loop_mean_and_deviation(P, V, combinations):
    """
    Plot the mean and deviation of ventricular pV Loop.

    Args:
        P (dict): Dictionary containing ventricular pressure data.
        V (dict): Dictionary containing ventricular volume data.
        combinations (list): List of combinations.

    Example:
        plot_ventricular_pV_Loop_mean_and_deviation(MeasurementP, MeasurementV, combinations)
    """
    # Check which pV Loop to plot
    if 'Plv' in P:
        VRaw = V['Vlv']
        PRaw = P['Plv']
        VStats = V['VlvStats']
        PStats = P['PlvStats']
        figname = 'LV'
    elif 'Prv' in P:
        VRaw = V['Vrv']
        PRaw = P['Prv']
        VStats = V['VrvStats']
        PStats = P['PrvStats']
        figname = 'RV'
    else:
        print('No ventricular pressure found in the dictionary.')
        return
    
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']=10
    plt.rcParams["figure.figsize"]=[16/2.54, 13.6/2.54]

    plt.figure(num=figname)
 
    plt.plot(VStats[0], PStats[0], color='red', label='Mean')
    plt.plot(VStats[1], PStats[1], "b", label=find_combination_name(combinations, PRaw, VRaw, 
                                                                    PStats[1], VStats[1]))
    plt.plot(VStats[2], PStats[2], "b:", label=find_combination_name(combinations, PRaw, VRaw, 
                                                                     PStats[2], VStats[2]))

    plt.xlabel('Volume in mL')
    plt.ylabel('Pressure in mmHg')
    plt.rcParams['font.size']=10
    plt.legend(loc='lower center', bbox_to_anchor= (0.5, 0.99), ncol=3)
    #plt.show()

def plot_dPfil_Pcrrttuin_Pcrrttuout_mean_and_deviation(x, xlabel,
                                           meandPfil, mindPfil, maxdPfil,
                                           meanPcrrttuin, minPcrrttuin, maxPcrrttuin,
                                           meanPcrrttuout, minPcrrttuout, maxPcrrttuout,
                                           meanPecmodrain, minPecmodrain, maxPecmodrain,
                                           meanPecmoreturn, minPecmoreturn, maxPecmoreturn):
    """
    Plot the mean and deviation of dPfil, Pcrrttuin, and Pcrrttuout for all rpms.

    Args:
        x (array-like): The x-axis values.
        xlabel (str): The label for the x-axis.
        meandPfil (array-like): The mean values of dPfil.
        mindPfil (array-like): The minimum values of dPfil.
        maxdPfil (array-like): The maximum values of dPfil.
        meanPcrrttuin (array-like): The mean values of Pcrrttuin.
        minPcrrttuin (array-like): The minimum values of Pcrrttuin.
        maxPcrrttuin (array-like): The maximum values of Pcrrttuin.
        meanPcrrttuout (array-like): The mean values of Pcrrttuout.
        minPcrrttuout (array-like): The minimum values of Pcrrttuout.
        maxPcrrttuout (array-like): The maximum values of Pcrrttuout.
        meanPecmodrain (array-like): The mean values of Pecmodrain.
        minPecmodrain (array-like): The minimum values of Pecmodrain.
        maxPecmodrain (array-like): The maximum values of Pecmodrain.
        meanPecmoreturn (array-like): The mean values of Pecmoreturn.
        minPecmoreturn (array-like): The minimum values of Pecmoreturn.
        maxPecmoreturn (array-like): The maximum values of Pecmoreturn.

    Example:
        plot_dPfil_Pcrrttuin_Pcrrttuout_mean_and_deviation(jnp.array(MeasurementQrpm['Qecmopump']['Global']['mean'])*0.06, 'ECMO pump flow in L/min',
            MeasurementPrpm['dPfil']['Global']['mean'], MeasurementPrpm['dPfil']['Global']['min'], MeasurementPrpm['dPfil']['Global']['max'],
            MeasurementPrpm['Pcrrttuin']['Global']['mean'], MeasurementPrpm['Pcrrttuin']['Global']['min'], MeasurementPrpm['Pcrrttuin']['Global']['max'],
            MeasurementPrpm['Pcrrttuout']['Global']['mean'], MeasurementPrpm['Pcrrttuout']['Global']['min'], MeasurementPrpm['Pcrrttuout']['Global']['max'],
            MeasurementPrpm['Pecmodrain']['Global']['mean'], MeasurementPrpm['Pecmodrain']['Global']['min'], MeasurementPrpm['Pecmodrain']['Global']['max'],
            MeasurementPrpm['Pecmoreturn']['Global']['mean'], MeasurementPrpm['Pecmoreturn']['Global']['min'], MeasurementPrpm['Pecmoreturn']['Global']['max'])
    
    """

    # Pressure alarms
    drainageMin=-250
    drainageMax=250
    returnMin=-50
    returnMax=350

    plt.figure('dPfilter')
    # Psart - Psvn, Ppart means and deviations
    plt.plot(x, meandPfil, linewidth=2.0, label='Mean ΔPfil (filter pressure drop)')
    plt.fill_between(x, mindPfil, maxdPfil, alpha=0.2, color='blue')
    plt.plot(x, mindPfil, "b--")
    plt.plot(x, maxdPfil, "b--")
    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    plt.legend(loc='lower left')
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_filter_pressure_drop.svg')
    #plt.savefig('Analyses/CRRT_Study_filter_pressure_drop.png')

    #########################################################
    ######### CRRT Drain and Return Tubing Pressure #########
    #########################################################

    plt.figure('CRRT Drain tubing')
    plt.plot(x, meanPcrrttuin, linewidth=2.0, label='Drain tubing')
    plt.fill_between(x, minPcrrttuin, maxPcrrttuin, alpha=0.2, color='blue')
    plt.plot(x, minPcrrttuin, "b--")
    plt.plot(x, maxPcrrttuin, "b--")
    plt.plot(x, [drainageMin for e in x], "r")
    plt.plot(x, [drainageMax for e in x], color='red')
    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=3)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_CRRT_drain_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_CRRT_drain_tubing_pressure.png')

    plt.figure('CRRT Return tubing')
    plt.plot(x, meanPcrrttuout, linewidth=2.0, label='Return tubing')
    plt.fill_between(x, minPcrrttuout, maxPcrrttuout, alpha=0.2, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    plt.plot(x, minPcrrttuout, "b--")
    plt.plot(x, maxPcrrttuout, "b--")
    plt.plot(x, [returnMin for e in x], "r")
    plt.plot(x, [returnMax for e in x], color='red')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=3)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_CRRT_return_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_CRRT_return_tubing_pressure.png')

    #########################################################
    ######### ECMO Drain and Return Tubing Pressure #########
    #########################################################

    plt.figure('ECMO Drain tubing')
    plt.plot(x, meanPecmodrain, linewidth=2.0, label='Drain tubing')
    plt.fill_between(x, minPecmodrain, maxPecmodrain, alpha=0.2, color='blue')
    plt.plot(x, minPecmodrain, "b--")
    plt.plot(x, maxPecmodrain, "b--")
    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=3)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_ECMO_drain_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_ECMO_drain_tubing_pressure.png')

    plt.figure('ECMO Return tubing')
    plt.plot(x, meanPecmoreturn, linewidth=2.0, label='Return tubing')
    plt.fill_between(x, minPecmoreturn, maxPecmoreturn, alpha=0.2, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel('Pressure in mmHg')
    plt.plot(x, minPecmoreturn, "b--")
    plt.plot(x, maxPecmoreturn, "b--")
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=3)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_ECMO_return_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_ECMO_return_tubing_pressure.png')

def plot_bar_plots(combinations, Psart, Psvn, Ppart, dPfil, Pcrrttuin, Pcrrttuout, 
                   Pecmodrain, Pecmoreturn):
    """
    Plot the mean of the quantities for each combination in bar plots.

    Args:
        combinations (list): List of combinations.
        Psart (dict): Dictionary containing Psart data.
        Psvn (dict): Dictionary containing Psvn data.
        Ppart (dict): Dictionary containing Ppart data.
        dPfil (dict): Dictionary containing dPfil data.
        Pcrrttuin (dict): Dictionary containing Pcrrttuin data.
        Pcrrttuout (dict): Dictionary containing Pcrrttuout data.
        Pecmodrain (dict): Dictionary containing Pecmodrain data.
        Pecmoreturn (dict): Dictionary containing Pecmoreturn data.

    Example:
        rementP['Psart'], MeasurementP['Psvn'], MeasurementP['Ppart'],
                       MeasurementP['dPfil'], MeasurementP['Pcrrttuin'], MeasurementP['Pcrrttuout'],
                       MeasurementP['Pecmodrain'], MeasurementP['Pecmoreturn'])
    """
  
    Psartcombi=[np.round(np.array(mean_over_time(Psart['Psart'+str(i)])), 2) for i in range(len(combinations))]
    Psvncombi=[np.round(np.array(mean_over_time(Psvn['Psvn'+str(i)])), 2) for i in range(len(combinations))]
    Ppartcombi=[np.round(np.array(mean_over_time(Ppart['Ppart'+str(i)])), 2) for i in range(len(combinations))]
    dPfilcombi=[np.round(np.array(mean_over_time(dPfil['dPfil'+str(i)])), 2) for i in range(len(combinations))]
    Pcrrttuincombi=[np.round(np.array(mean_over_time(Pcrrttuin['Pcrrttuin'+str(i)])), 2) for i in range(len(combinations))]
    Pcrrttuoutcombi=[np.round(np.array(mean_over_time(Pcrrttuout['Pcrrttuout'+str(i)])), 2) for i in range(len(combinations))]
    Pecmodraincombi=[np.round(np.array(mean_over_time(Pecmodrain['Pecmodrain'+str(i)])), 2) for i in range(len(combinations))]
    Pecmoreturncombi=[np.round(np.array(mean_over_time(Pecmoreturn['Pecmoreturn'+str(i)])), 2) for i in range(len(combinations))]
    
    combinations=convert_num_to_access(combinations)

    plt.figure()
    bars=plt.barh(combinations, Psartcombi, color='red', alpha=0.8)
    plt.bar_label(bars, padding=10, color='black', fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean art. pressure in mmHg')
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_art_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_art_pressure.png')

    plt.figure()
    bars=plt.barh(combinations, Psvncombi, color='blue', alpha=0.8)
    plt.bar_label(bars, padding=10, color='black', fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean svn. pressure in mmHg')
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_ven_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_ven_pressure.png')

    plt.figure()
    bars=plt.barh(combinations, Ppartcombi, color='green', alpha=0.8)
    plt.bar_label(bars, padding=10, color='black', fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean pulm. art. pressure in mmHg')
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_pulm_art_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_pulm_art_pressure.png')

    plt.figure()
    bars=plt.barh(combinations, dPfilcombi)
    plt.bar_label(bars, padding=10, color='black', fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean ΔP filter in mmHg')
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_filter_pressure_drop.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_filter_pressure_drop.png')

    # Pin and Pout
    # Pressure alarms
    drainageMin=-250
    drainageMax=250
    returnMin=-50
    returnMax=350

    #########################################################
    ######### CRRT Drain and Return Tubing Pressure #########
    #########################################################

    plt.figure('CRRT Drain tubing')
    bars= plt.barh(combinations, Pcrrttuincombi)
    plt.bar_label(bars, color='black', padding=10, fontsize=10, label_type='edge', fmt='%.2f')
    #plt.axvline(drainageMin, color='red')
    plt.axvline(drainageMax, color='red')
    plt.xlabel('Mean drain tube pressure in mmHg')
    plt.xlim(-150, 320)
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=2)
    plt.tight_layout()
    #plt.show()

    #plt.savefig('Analyses/CRRT_Study_bar_drain_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_drain_tubing_pressure.png')

    plt.figure('CRRT Return tubing')
    bars= plt.barh(combinations, Pcrrttuoutcombi)
    plt.bar_label(bars, color='black', padding=10, fontsize=10, label_type='edge', fmt='%.2f')
    plt.axvline(returnMin, color='red', label='Pressure alarm min')
    plt.axvline(returnMax, color='red', label='Pressure alarm max')
    plt.xlabel('Mean return tube pressure in mmHg')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=2)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_return_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_return_tubing_pressure.png')

    #########################################################
    ######### ECMO Drain and Return Tubing Pressure #########
    #########################################################
    
    plt.figure('ECMO Drain tubing')
    bars = plt.barh(combinations, Pecmodraincombi)
    plt.bar_label(bars, color='black', padding=10, fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean drain tube pressure in mmHg')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=2)
    plt.tight_layout()
    #plt.show()

    #plt.savefig('Analyses/CRRT_Study_bar_drain_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_drain_tubing_pressure.png')

    plt.figure('ECMO Return tubing')
    bars= plt.barh(combinations, Pecmoreturncombi)
    plt.bar_label(bars, color='black', padding=10, fontsize=10, label_type='edge', fmt='%.2f')
    plt.xlabel('Mean return tube pressure in mmHg')
    #plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=2)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Analyses/CRRT_Study_bar_return_tubing_pressure.svg')
    #plt.savefig('Analyses/CRRT_Study_bar_return_tubing_pressure.png')

def sort_data(data, name=False, sorting=False):
    """
    Sorts the data based on the minimum values for the specified quantity name.

    Parameters:
    - data (pandas.DataFrame): The input data.
    - name (str): The name of the quantity to be sorted.

    Returns:
    - sortedMinData_indices (numpy.ndarray): The indices of the sorted data.
    - dataList_sorted (list): The sorted data.

    """

    if name is False:
        dataList = list(data.values())

    else:
        dataList = list(data[name].values())

    MinData_array = [np.min(values) for values in dataList]

    if isinstance(name, str) and name[0] == 'Q':
        if sorting is False: sorting = np.argsort(np.absolute(MinData_array))
        
        dataList_sorted = [[element * 0.06 for element in dataList[i]] for i in sorting]

    else:
        if sorting is False: sorting = np.argsort(MinData_array)
        
        dataList_sorted = [dataList[i] for i in sorting]

    return sorting, dataList_sorted

def plot_all_curves_median_filling(x_array, y_array, indices, ax, color, alpha, label):


    # Combine all x arrays and sort
    x_common = np.unique(np.concatenate(x_array))

    # Interpolate y values at the common x points
    interp_funcs = [
        interp1d(x_array[i], y_array[i], fill_value='extrapolate')
        for i in range(len(x_array))
    ]

    y_interpolated = [func(x_common) for func in interp_funcs]
    y_min = np.min(y_interpolated, axis=0)
    y_max = np.max(y_interpolated, axis=0)

    # Plot median curve
    ax.plot(x_common, y_interpolated[len(indices) // 2], \
                   linewidth=1.0, color=color, label=label)

    # Fill polygon area in between each curve
    ax.fill_between(x_common, y_min, y_max, alpha=alpha, color=color)

    # Plott all curves
    for i in range(len(y_array)):
        ax.plot(x_common, y_interpolated[i], color=color, linewidth=0.75, alpha=alpha*2)

def combined_plot_static_rpm(xTime, combinations, pressures, volumes):
    
    # Plotting
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10

    cm = 1/2.54

    fig_height = 23 * cm
    fig_width = 19 * cm

    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # [0, 1]: Ventricular pV Loops
    VolumesVen = volumes['Vlv'] if 'Plv' in pressures else volumes['Vrv']
    PressuresVen = pressures['Plv'] if 'Plv' in pressures else pressures['Prv']

    # Sort volumes w.r.t. ESV and get min, max and median ESV arrays for plotting
    sortedESV_indices, VolumesVenList_sorted = sort_data(VolumesVen)
    PressuresVenList_sorted = sort_data(PressuresVen, sorting=sortedESV_indices)[1]

    # Plot all pV Loops
    for i in range(len(VolumesVenList_sorted)):
        axs[0, 1].plot(VolumesVenList_sorted[i], PressuresVenList_sorted[i], color='grey', linewidth=0.75, zorder=7, alpha=0.2)

    # Plot pV Loop with max ESV
    axs[0, 1].fill(VolumesVenList_sorted[-1], PressuresVenList_sorted[-1], color='#fc8d62', alpha=0.2, zorder=1)
    axs[0, 1].plot(VolumesVenList_sorted[-1], PressuresVenList_sorted[-1], color='#fc8d62', zorder=2, \
        label=find_combination_name(combinations, index=sortedESV_indices[-1]))
    
    # TBR: plotting median
    # Plot pV Loop with median ESV
    #axs[0, 1].fill(VolumesVenList_sorted[len(sortedESV_indices) // 2], PressuresVenList_sorted[len(sortedESV_indices) // 2], "white", alpha=1.0, zorder=3)
    #axs[0, 1].fill(VolumesVenList_sorted[len(sortedESV_indices) // 2], PressuresVenList_sorted[len(sortedESV_indices) // 2], "black", alpha=0.2, zorder=3)
    #axs[0, 1].plot(VolumesVenList_sorted[len(sortedESV_indices) // 2], PressuresVenList_sorted[len(sortedESV_indices) // 2], color='black', \
    #               label=find_combination_name(combinations, index=sortedESV_indices[len(sortedESV_indices) // 2]), zorder=4)
    
    # Plot pV Loop with min ESV
    axs[0, 1].fill(VolumesVenList_sorted[0], PressuresVenList_sorted[0], "white", alpha=1.0, zorder=5)
    axs[0, 1].fill(VolumesVenList_sorted[0], PressuresVenList_sorted[0], color='#8da0cb', alpha=0.2, zorder=5)
    axs[0, 1].plot(VolumesVenList_sorted[0], PressuresVenList_sorted[0], color='#8da0cb', zorder=6, 
        label=find_combination_name(combinations, index=sortedESV_indices[0]))

    axs[0, 1].set_xlabel(f"{'LV' if 'Plv' in pressures else 'RV'} volume in mL")
    axs[0, 1].set_ylabel(f"{'LV' if 'Plv' in pressures else 'RV'} pressure in mmHg")
    axs[0, 1].set_xlim(left = int(0.85*np.min(VolumesVenList_sorted)))
    axs[0, 1].set_ylim(bottom = 0)
    legendorder = [1, 0]
    handles, labels = axs[0, 1].get_legend_handles_labels()

    axs[0, 1].legend([handles[idx] for idx in legendorder],[labels[idx] for idx in legendorder], \
                     loc='upper right', ncol=len(legendorder), bbox_to_anchor= (0.25, -0.16), columnspacing=1.)

    axs[0, 1].text(-0.16, 1.0, 'b)', transform=axs[0, 1].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    # [0, 0]: Pressure comparison
    # Sort pressures w.r.t. DP and get min, max and median DP arrays for plotting
    # Arterial Pressure
    sortedDPArterial_indices, PressuresArterialList_sorted = sort_data(pressures, 'Psart')

    print('\nArterial - Combination with maximum DP: \n' + find_combination_name(combinations, index=sortedDPArterial_indices[-1]))
    print('Arterial - Combination with median DP: \n' + find_combination_name(combinations, index=sortedDPArterial_indices[len(sortedDPArterial_indices) // 2]))
    print('Arterial - Combination with minimum DP: \n' + find_combination_name(combinations, index=sortedDPArterial_indices[0]))

    # TBR: plotting median and filling between min and max
    #axs[0, 0].plot(xTime, PressuresArterialList_sorted[len(sortedDPArterial_indices) // 2], linewidth=1.0, color='red', label='Arterial')
    #axs[0, 0].fill_between(xTime, PressuresArterialList_sorted[0], PressuresArterialList_sorted[-1], alpha=0.1, color='red')

    for i in range(len(PressuresArterialList_sorted)):
        axs[0, 0].plot(xTime, PressuresArterialList_sorted[i], color='black', linewidth=0.75, alpha=0.1)

    PressuresArterialList = list(pressures['Psart'].values())
    axs[0, 0].plot(xTime, PressuresArterialList[sortedESV_indices[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 0].plot(xTime, PressuresArterialList[sortedESV_indices[-1]], color='#fc8d62',  linewidth=1.25)

    # Venous Pressure
    sortedDPVenous_indices, PressuresVenousList_sorted = sort_data(pressures, 'Psvn')

    print('\nVenous - Combination with maximum DP: \n' + find_combination_name(combinations, index=sortedDPVenous_indices[-1]))
    print('Venous - Combination with median DP: \n' + find_combination_name(combinations, index=sortedDPVenous_indices[len(sortedDPVenous_indices) // 2]))
    print('Venous - Combination with minimum DP: \n' + find_combination_name(combinations, index=sortedDPVenous_indices[0]))

    # TBR: plotting median and filling between min and max
    #axs[0, 0].plot(xTime, PressuresVenousList_sorted[len(sortedDPVenous_indices) // 2], linewidth=1.0, color='blue', label='Venous')
    #axs[0, 0].fill_between(xTime, PressuresVenousList_sorted[0], PressuresVenousList_sorted[-1], alpha=0.1, color='blue')

    for i in range(len(PressuresVenousList_sorted)):
        axs[0, 0].plot(xTime, PressuresVenousList_sorted[i], color='black', linewidth=0.75, alpha=0.1)

    PressuresVenousList = list(pressures['Psvn'].values())
    axs[0, 0].plot(xTime, PressuresVenousList[sortedESV_indices[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 0].plot(xTime, PressuresVenousList[sortedESV_indices[-1]], color='#fc8d62',  linewidth=1.25)

    # Pulmonary Artery Pressure
    sortedDPPulmArterial_indices, PressuresPulmArterialList_sorted = sort_data(pressures, 'Ppart')

    print('\nPulm. Arterial - Combination with maximum DP: \n' + find_combination_name(combinations, index=sortedDPPulmArterial_indices[-1]))
    print('Pulm. Arterial - Combination with median DP: \n' + find_combination_name(combinations, index=sortedDPPulmArterial_indices[len(sortedDPPulmArterial_indices) // 2]))
    print('Pulm. Arterial - Combination with minimum DP: \n' + find_combination_name(combinations, index=sortedDPPulmArterial_indices[0]))

    # TBR: plotting median and filling between min and max
    #axs[0, 0].plot(xTime, PressuresPulmArterialList_sorted[len(sortedDPPulmArterial_indices) // 2], linewidth=1.0, color='green', label='Pulm. artery')
    #axs[0, 0].fill_between(xTime, PressuresPulmArterialList_sorted[0], PressuresPulmArterialList_sorted[-1], alpha=0.2, color='green')
    
    for i in range(len(PressuresPulmArterialList_sorted)):
        axs[0, 0].plot(xTime, PressuresPulmArterialList_sorted[i], color='black', linewidth=0.75, alpha=0.1)

    PressuresPulmArterialList = list(pressures['Ppart'].values())
    axs[0, 0].plot(xTime, PressuresPulmArterialList[sortedESV_indices[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 0].plot(xTime, PressuresPulmArterialList[sortedESV_indices[-1]], color='#fc8d62',  linewidth=1.25)

    axs[0, 0].set_xlabel('Normalized cardiac cycle')
    axs[0, 0].set_ylabel('Pressure in mmHg')
    #axs[0, 0].legend(loc='upper right', ncol=3, bbox_to_anchor= (1.02, 1.15), columnspacing=0.5)

    # Create annotations for "arterial", "venous" and "pulmonary artery
    positionX = xTime[np.argmin(PressuresArterialList_sorted[0])]
    positionY = np.min(PressuresArterialList_sorted[0])
    axs[0, 0].annotate('Arterial', xy=(positionX, positionY * 0.975), \
            xytext=(positionX * 1.1, positionY * 0.8), \
            arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    positionX = xTime[len(PressuresPulmArterialList_sorted[-1])//2 + np.argmax(PressuresPulmArterialList_sorted[-1][len(PressuresPulmArterialList_sorted[-1])//2:])]
    positionY = np.max(PressuresPulmArterialList_sorted[-1][len(PressuresPulmArterialList_sorted[-1])//2:])
    axs[0, 0].annotate('Pulm. Arterial', xy=(positionX, positionY * 1.05), \
    xytext=(positionX * 1.0, positionY * 1.6), \
    arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    positionX = xTime[np.argmax(PressuresVenousList_sorted[-1][:len(PressuresVenousList_sorted[-1])//2])]
    positionY = np.max(PressuresVenousList_sorted[-1][:len(PressuresVenousList_sorted[-1])//2])
    axs[0, 0].annotate('Venous', xy=(positionX * 1.05, positionY * 1.1), \
    xytext=(positionX * 0.4, positionY * 1.35), \
    arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    axs[0, 0].text(-0.2, 1.0, 'a)', transform=axs[0, 0].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')
    
    # [1, 0]: CRRT drain tubing pressure
    Pcrrttuincombi = [np.round(np.array(mean_over_time(pressures['Pcrrttuin']['Pcrrttuin'+str(i)])), 2) for i in range(len(combinations))]
    combinations = convert_num_to_access(combinations)
    n = len(combinations)
    combinations_labels = [f'{chr(96 + n - i)}) {combination}' for i, combination in enumerate(combinations)]
    
    #label_prefix_length = 4
    #combinations_labels = [
    #    f'{chr(96 + n - i)}) {combo.split("\n")[0]}\n{" " * label_prefix_length}{combo.split("\n")[1]}' 
    #    for i, combo in enumerate(combinations)
    #    ]'''
    
    bars = axs[1, 0].barh(combinations_labels, Pcrrttuincombi, color='black', alpha=0.2)

    axs[1, 0].barh(combinations_labels[sortedESV_indices[0]], Pcrrttuincombi[sortedESV_indices[0]], color='white', alpha=1.0)
    axs[1, 0].barh(combinations_labels[sortedESV_indices[0]], Pcrrttuincombi[sortedESV_indices[0]], color='#8da0cb', alpha=0.6)

    axs[1, 0].barh(combinations_labels[sortedESV_indices[-1]], Pcrrttuincombi[sortedESV_indices[-1]], color='white', alpha=1.0)
    axs[1, 0].barh(combinations_labels[sortedESV_indices[-1]], Pcrrttuincombi[sortedESV_indices[-1]], color='#fc8d62', alpha=0.6)

    axs[1, 0].bar_label(bars, padding=3, fmt='%.2f', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1'))
    axs[1, 0].axvline(-250, color='k', linestyle='--', label='Min Drainage')
    axs[1, 0].axvline(250, color='k', linestyle='--', label='Max Drainage')
    axs[1, 0].set_xlabel('Pressure in mmHg')
    axs[1, 0].set_xlim(-300, 300)
    #axs[1, 0].set_yticklabels(combinations_labels, ha='left', position=(-0.275,0))

    axs[1, 0].text(-0.2, 1.05, 'c)', transform=axs[1, 0].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    # [1, 1]: CRRT return tubing pressure
    Pcrrttuoutcombi = [np.round(np.array(mean_over_time(pressures['Pcrrttuout']['Pcrrttuout'+str(i)])), 2) for i in range(len(combinations))]
    bars = axs[1, 1].barh(combinations_labels, Pcrrttuoutcombi, color='black', alpha=0.2)

    axs[1, 1].barh(combinations_labels[sortedESV_indices[0]], Pcrrttuoutcombi[sortedESV_indices[0]], color='white', alpha=1.0)
    axs[1, 1].barh(combinations_labels[sortedESV_indices[0]], Pcrrttuoutcombi[sortedESV_indices[0]], color='#8da0cb', alpha=0.6)

    axs[1, 1].barh(combinations_labels[sortedESV_indices[-1]], Pcrrttuoutcombi[sortedESV_indices[-1]], color='white', alpha=1.0)
    axs[1, 1].barh(combinations_labels[sortedESV_indices[-1]], Pcrrttuoutcombi[sortedESV_indices[-1]], color='#fc8d62', alpha=0.6)

    axs[1, 1].bar_label(bars, padding=3, fmt='%.2f', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.1'))
    axs[1, 1].axvline(-50, color='k', linestyle='--', label='Min Return')
    axs[1, 1].axvline(350, color='k', linestyle='--', label='Max Return')
    axs[1, 1].set_xlabel('Pressure in mmHg')
    axs[1, 1].set_xlim(-75, 400)
    axs[1, 1].set_yticklabels([])

    axs[1, 1].text(-0.16, 1.05, 'd)', transform=axs[1, 1].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    file_path = os.path.join('Analyses', 'Results', '01_Connections')

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = 'CRRT_Study_Connections'

    fig.savefig(os.path.join(file_path, file_name) + ".pdf")
    fig.savefig(os.path.join(file_path, file_name) + ".svg")
    fig.savefig(os.path.join(file_path, file_name) + ".png")

    return sortedESV_indices

def combined_plot_varying_rpm(Flows, timeAveragePressure, timeAverageFlow, curveIdentifier, combinations):
                
    meanQecmopump = np.array(Flows['Qecmopump']['Global']['mean'])*0.06

    # Plotting
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10

    cm = 1/2.54

    fig_height = 23 * cm
    fig_width = 19 * cm

    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # [0, 0]: Pressure comparison
    # Arterial Pressure
    # Sort pressures w.r.t. min(pressure) and get min, max and median arrays for plotting
    sortedMinArterial_indices, PressuresArterialList_sorted = sort_data(timeAveragePressure, 'Psart')
    FlowArterialECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinArterial_indices)[1]

    #plot_all_curves_median_filling(FlowArterialECMOPumpList_sorted, PressuresArterialList_sorted, sortedMinArterial_indices, \
    #                  axs[0, 0], 'red', 0.1, 'Arterial')
    
    for i in range(len(PressuresArterialList_sorted)):
        axs[0, 0].plot(FlowArterialECMOPumpList_sorted[i], PressuresArterialList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresArterialList = list(timeAveragePressure['Psart'].values())
    FlowECMOPumpList = np.array(list(timeAverageFlow['Qecmopump'].values())) * 0.06
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresArterialList[curveIdentifier[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresArterialList[curveIdentifier[-1]], color='#fc8d62',  linewidth=1.25)

    # Venous Pressure
    sortedMinVenous_indices, PressuresVenousList_sorted = sort_data(timeAveragePressure, 'Psvn')
    FlowVenousECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinVenous_indices)[1]

    #plot_all_curves_median_filling(FlowVenousECMOPumpList_sorted, PressuresVenousList_sorted, sortedMinVenous_indices, \
    #                  axs[0, 0], 'blue', 0.1, 'Venous')

    for i in range(len(PressuresVenousList_sorted)):
        axs[0, 0].plot(FlowVenousECMOPumpList_sorted[i], PressuresVenousList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresVenousList = list(timeAveragePressure['Psvn'].values())
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresVenousList[curveIdentifier[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresVenousList[curveIdentifier[-1]], color='#fc8d62',  linewidth=1.25)

    # Pulmonary Artery Pressure
    sortedMinPulmArterial_indices, PressuresPulmArterialList_sorted = sort_data(timeAveragePressure, 'Ppart')
    FlowPulmArterialECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinPulmArterial_indices)[1]

    #plot_all_curves_median_filling(FlowPulmArterialECMOPumpList_sorted, PressuresPulmArterialList_sorted, sortedMinPulmArterial_indices, \
    #                  axs[0, 0], 'green', 0.1, 'Pulm. artery')

    for i in range(len(PressuresPulmArterialList_sorted)):
        axs[0, 0].plot(FlowPulmArterialECMOPumpList_sorted[i], PressuresPulmArterialList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresPulmArterialList = list(timeAveragePressure['Ppart'].values())
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresPulmArterialList[curveIdentifier[0]], \
                   color='#8da0cb', linewidth=1.25, label=find_combination_name(combinations, index=curveIdentifier[0]))
    axs[0, 0].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresPulmArterialList[curveIdentifier[-1]], \
                   color='#fc8d62',  linewidth=1.25, label=find_combination_name(combinations, index=curveIdentifier[-1]))

    axs[0, 0].set_xlabel('ECMO pump flow in L/min')
    axs[0, 0].set_ylabel('Pressure in mmHg')
    #axs[0, 0].legend(loc='upper right', ncol=3, bbox_to_anchor= (1.035, 1.12), columnspacing=0.5)

    axs[0, 0].text(-0.2, 1.0, 'a)', transform=axs[0, 0].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    # Create annotations for "arterial", "venous" and "pulmonary artery
    positionX = FlowArterialECMOPumpList_sorted[0][len(PressuresArterialList_sorted[0])//2]
    positionY = PressuresArterialList_sorted[0][len(PressuresArterialList_sorted[0])//2]
    axs[0, 0].annotate('Arterial', xy=(positionX * 0.8, positionY * 0.875), \
    xytext=(positionX * 1.0, positionY * 0.8), \
    arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    positionX = FlowPulmArterialECMOPumpList_sorted[-1][int(0.15*len(PressuresPulmArterialList_sorted[-1]))]
    positionY = PressuresPulmArterialList_sorted[-1][int(0.15*len(PressuresPulmArterialList_sorted[-1]))]
    axs[0, 0].annotate('Pulm. Arterial', xy=(positionX * 2.0, positionY * 1.0), \
    xytext=(positionX * 1.0, positionY * 1.4), \
    arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    positionX = FlowVenousECMOPumpList_sorted[-1][int(0.05*len(PressuresVenousList_sorted[-1]))]
    positionY = PressuresVenousList_sorted[-1][int(0.05*len(PressuresVenousList_sorted[-1]))]
    axs[0, 0].annotate('Venous', xy=(positionX * 100.0, positionY * 0.9), \
    xytext=(positionX * 1.2, positionY * 1.2), \
    arrowprops=dict(arrowstyle='-', facecolor='black'), color='black')

    legendorder = [0, 1]
    handles, labels = axs[0, 0].get_legend_handles_labels()

    axs[0, 0].legend([handles[idx] for idx in legendorder],[labels[idx] for idx in legendorder], \
                     loc='upper right', ncol=len(legendorder), bbox_to_anchor= (1.5, -0.16), columnspacing=1.)
    
    # [0, 1]: ECMO drain tubing pressure
    sortedMinECMODrain_indices, PressuresECMODrainList_sorted = sort_data(timeAveragePressure, 'Pecmodrain')
    FlowECMODrainECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinECMODrain_indices)[1]

    #plot_all_curves_median_filling(FlowECMODrainECMOPumpList_sorted, PressuresECMODrainList_sorted, sortedMinECMODrain_indices, \
    #                  axs[0, 1], 'blue', 0.1, 'Drain tubing')
    
    for i in range(len(PressuresECMODrainList_sorted)):
        axs[0, 1].plot(FlowECMODrainECMOPumpList_sorted[i], PressuresECMODrainList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresECMODrainList = list(timeAveragePressure['Pecmodrain'].values())
    axs[0, 1].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresECMODrainList[curveIdentifier[0]], color='#8da0cb', linewidth=1.25)
    axs[0, 1].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresECMODrainList[curveIdentifier[-1]], color='#fc8d62',  linewidth=1.25)

    axs[0, 1].set_xlabel('ECMO pump flow in L/min')
    axs[0, 1].set_ylabel('Pressure in mmHg')

    axs[0, 1].text(-0.2, 1.0, 'b)', transform=axs[0, 1].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    # [1, 0]: CRRT drain tubing pressure
    # Pressure alarms for CRRT circuit
    drainageMin=-250
    drainageMax=250
    returnMin=-50
    returnMax=350

    sortedMinCRRTDrain_indices, PressuresCRRTDrainList_sorted = sort_data(timeAveragePressure, 'Pcrrttuin')
    FlowCRRTDrainECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinCRRTDrain_indices)[1]

    #plot_all_curves_median_filling(FlowCRRTDrainECMOPumpList_sorted, PressuresCRRTDrainList_sorted, sortedMinCRRTDrain_indices, \
    #                  axs[1, 0], 'blue', 0.1, 'Drain tubing')
   
    for i in range(len(PressuresCRRTDrainList_sorted)):
        axs[1, 0].plot(FlowCRRTDrainECMOPumpList_sorted[i], PressuresCRRTDrainList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresCRRTDrainList = list(timeAveragePressure['Pcrrttuin'].values())
    axs[1, 0].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresCRRTDrainList[curveIdentifier[0]], color='#8da0cb', linewidth=1.25)
    axs[1, 0].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresCRRTDrainList[curveIdentifier[-1]], color='#fc8d62',  linewidth=1.25)

    axs[1, 0].plot(meanQecmopump, [drainageMin for e in meanQecmopump], 'k--')
    axs[1, 0].plot(meanQecmopump, [drainageMax for e in meanQecmopump], 'k--')

    axs[1, 0].set_xlabel('ECMO pump flow in L/min')
    axs[1, 0].set_ylabel('Pressure in mmHg')
    axs[1, 0].text(-0.2, 1.0, 'c)', transform=axs[1, 0].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')

    # [1, 1]: CRRT return tubing pressure
    sortedMinCRRTReturn_indices, PressuresCRRTReturnList_sorted = sort_data(timeAveragePressure, 'Pcrrttuout')
    FlowCRRTReturnECMOPumpList_sorted = sort_data(timeAverageFlow, 'Qecmopump', sorting=sortedMinCRRTReturn_indices)[1]

    #plot_all_curves_median_filling(FlowCRRTReturnECMOPumpList_sorted, PressuresCRRTReturnList_sorted, sortedMinCRRTReturn_indices, \
    #                  axs[1, 1], 'blue', 0.1, 'Return tubing')

    for i in range(len(PressuresCRRTReturnList_sorted)):
        axs[1, 1].plot(FlowCRRTReturnECMOPumpList_sorted[i], PressuresCRRTReturnList_sorted[i], color='black', linewidth=0.75, alpha=0.2)

    # Highlight the curves identified in static run
    PressuresCRRTReturn = list(timeAveragePressure['Pcrrttuout'].values())
    axs[1, 1].plot(FlowECMOPumpList[curveIdentifier[0]], PressuresCRRTReturn[curveIdentifier[0]], color='#8da0cb', linewidth=1.25)
    axs[1, 1].plot(FlowECMOPumpList[curveIdentifier[-1]], PressuresCRRTReturn[curveIdentifier[-1]], color='#fc8d62',  linewidth=1.25)

    axs[1, 1].plot(meanQecmopump, [returnMin for e in meanQecmopump], 'k--')
    axs[1, 1].plot(meanQecmopump, [returnMax for e in meanQecmopump], 'k--')

    axs[1, 1].set_xlabel('ECMO pump flow in L/min')
    axs[1, 1].set_ylabel('Pressure in mmHg')
    axs[1, 1].text(-0.2, 1.0, 'd)', transform=axs[1, 1].transAxes, fontweight='bold', \
                   verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.show()

    file_path = os.path.join('Analyses', 'Results', '02_RPM')

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name = 'CRRT_Study_RPM'

    fig.savefig(os.path.join(file_path, file_name) + ".pdf")
    fig.savefig(os.path.join(file_path, file_name) + ".svg")
    fig.savefig(os.path.join(file_path, file_name) + ".png") 
   
def plot_pV_loops_over_rpm(rpm, P, V):
    """
    Plot mean, min and max pV Loops for all rpms

    Args:
        rpm (list): list of rpms
        P (dict): dictionary containing pressure data
        V (dict): dictionary containing volume data

    Example:
        plot_pV_loops_over_rpm(rpmBatch, MeasurementPrpm, MeasurementVrpm)

    """

    if 'Plv' in P:
        ven = 'lv'
    else:
        ven = 'rv'
    
    plt.figure(num=ven + ' pV loops')

    for i in range(len(rpm)):
        plt.plot(V['V'+ven][rpm[i]][0], P['P'+ven][rpm[i]][0], label='Mean' + str(rpm[i]))
        plt.plot(V['V'+ven][rpm[i]][1], P['P'+ven][rpm[i]][1], label='Min' + str(rpm[i]))
        plt.plot(V['V'+ven][rpm[i]][2], P['P'+ven][rpm[i]][2], label='Max' + str(rpm[i]))
    
    plt.xlabel('Volume in mL')
    plt.ylabel('Pressure in mmHg')
    plt.legend() # TODO: name labels of legend!

def plot_pV_loops_min_max_rpm_mean(rpmECMO, P, V):
    """
    Plot mean, min and max pV Loops for min and max rpm

    Args:
        rpmECMO (list): list of rpms
        P (dict): dictionary containing pressure data
        V (dict): dictionary containing volume data

    Example:
        plot_pV_loops_min_max_rpm_mean(rpmBatch, MeasurementPrpm, MeasurementVrpm)

    """
    
    if 'Plv' in P:
        ven = 'lv'
    else:
        ven = 'rv'
    
    # Data preparation
    minRpm=np.min(np.array(rpmECMO))
    maxRpm=np.max(np.array(rpmECMO))

    VMeanMax = V['V'+ven]['Global']['mean'][rpmBatch.index(maxRpm)]
    PMeanMax = P['P'+ven]['Global']['mean'][rpmBatch.index(maxRpm)]
    
    VMeanMin = V['V'+ven]['Global']['mean'][rpmBatch.index(minRpm)]
    PMeanMin = P['P'+ven]['Global']['mean'][rpmBatch.index(minRpm)]

    # Plotting pv Loops for min and max rpm
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']=10
    plt.rcParams["figure.figsize"]=[16/2.54, 13.6/2.54]

    plt.figure(num = 'Mean ' + ven + ' pV Loop for ' + str(minRpm) + ' and ' + str(maxRpm) + ' rev/min')
    plt.plot(VMeanMax, PMeanMax, color='r', label=str(maxRpm))
    plt.plot(VMeanMin, PMeanMin, color='b', label=str(minRpm))
    
    plt.xlabel('Volume in mL')
    plt.ylabel('Pressure in mmHg')

def plot_pV_loops_mean_min_max_rpm_combined(rpms, RPMFull, P, V):
    """
    Plot mean, min and max pV Loops for given input rpms

    Args:
        rpms (list): list of rpms
        RPMFull (list): list of all rpms
        P (dict): dictionary containing pressure data
        V (dict): dictionary containing volume data

    Example:    
        plot_pV_loops_mean_min_max_rpm_combined([np.min(np.array(rpmBatch))], rpmBatch, MeasurementPrpm, MeasurementVrpm)

        plot_pV_loops_mean_min_max_rpm_combined([np.min(np.array(rpmBatch)), np.max(np.array(rpmBatch))], 
                                                rpmBatch, MeasurementPrpm, MeasurementVrpm)

    """
        
    # Plot mean, min and max pv loops for given input rpms
    if 'Plv' in P:
        ven = 'lv'
    else:
        ven = 'rv'

    plt.figure(num = 'Mean, min and max ' + ven + ' pV loops for ' + str(rpms) + ' rev/min')
    color_map = plt.get_cmap('Set1')
    for i in rpms:

        index = RPMFull.index(i)
        color = color_map(i % 9)

        plt.plot(V['V'+ven]['Global']['min'][index], P['P'+ven]['Global']['min'][index], "-.", alpha=0.8, color=color)
        plt.plot(V['V'+ven]['Global']['max'][index], P['P'+ven]['Global']['max'][index], "-.", alpha=0.8, color=color)
        plt.plot(V['V'+ven]['Global']['mean'][index], P['P'+ven]['Global']['mean'][index], label=str(i), alpha=0.8, color=color)

    # Plotting Lines for Connecting pv Loops
    # Pink right down
    x=np.linspace(97.45, 104.54, 50)
    a=0.1918
    b=-3.05
    y=a*x+b

    # Pink upper left
    x4=np.linspace(34.54, 37.37, 50)
    a4=0.7456
    b4=0.5676
    y4=a4*x4+b4

    # Pink lower left
    x5=np.linspace(34.59, 37.33, 50)
    a5=0.1387
    b5=1.743
    y5=a5*x5+b5

    #plt.plot(x, y, "b-.", alpha=0.8)
    #plt.plot(x4, y4, "b-.", alpha=0.8)
    #plt.plot(x5, y5, "b-.", alpha=0.8)

    # Blue right down
    x21=np.linspace(92.64, 97.45, 50)
    x22=np.linspace(104.66, 117.66, 50)
    a2=0.1795
    b2=-2.445
    y21=a2*x21+b2
    y22=a2*x22+b2

    # Blue upper left
    x31=np.linspace(32.14, 34.54, 50)
    x32=np.linspace(37.37, 41.63, 50)
    a3=0.7597
    b3=0.1
    y31=a3*x31+b3
    y32=a3*x32+b3

    # Blue lower left
    x61=np.linspace(32.18, 34.59, 50)
    x62=np.linspace(37.33, 41.67, 50)
    a6=0.1665
    b6=0.9523
    y61=a6*x61+b6
    y62=a6*x62+b6

    #plt.plot(x21, y21, "r-.", alpha=0.8) # right down curve blue part 1
    #plt.plot(x22, y22, "r-.", alpha=0.8) # right down curve blue part 2
    #plt.plot(x31, y31, "r-.", alpha=0.8)
    #plt.plot(x32, y32, "r-.", alpha=0.8)
    #plt.plot(x61, y61, "r-.", alpha=0.8)
    #plt.plot(x62, y62, "r-.", alpha=0.8)

    plt.xlabel('Volume in mL')
    plt.ylabel('Pressure in mmHg')
    plt.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=2)
    #plt.show()

def print_ECMO_return_flow_and_pressure(Qecmoreturn, Pecmoreturn, Qecmodrain, Pecmodrain):
    """
    Print mean and deviation averaged over time for ECMO flow and return pressure at the current rpm.

    Args:
        Qecmoreturn (List): List containing the mean value and deviations of Qecmoreturn.
        Pecmoreturn (List): List containing the mean value and deviations of Pecmoreturn.
        Qecmodrain (List): List containing the mean value and deviations of Qecmodrain.
        Pecmodrain (List): List containing the mean value and deviations of Pecmodrain.

    Examples:
        print_ECMO_return_flow_and_pressure([mean_over_time(e) for e in MeasurementQ['QecmoreturnStats']], 
                                        [mean_over_time(e) for e in MeasurementP['PecmoreturnStats']],
                                        [mean_over_time(e) for e in MeasurementQ['QecmodrainStats']], 
                                        [mean_over_time(e) for e in MeasurementP['PecmodrainStats']])
    """

    print('\nFlow at ECMO System Return = ', np.round(np.array(Qecmoreturn[0]*0.06), 2), 'L/min; -', np.round(np.array(find_deviation_less(Qecmoreturn[0], Qecmoreturn[1])), 2), ' %', 
          ' +', np.round(np.array(find_deviation_plus(Qecmoreturn[0], Qecmoreturn[2])), 2), ' %')
    
    print('Pressure at ECMO System Return = ', np.round(np.array(Pecmoreturn[0]), 2), 'mmHg; -', np.round(np.array(find_deviation_less(Pecmoreturn[0], Pecmoreturn[1])), 2), ' %', 
          ' +', np.round(np.array(find_deviation_plus(Pecmoreturn[0], Pecmoreturn[2])), 2), ' %')
    
    print('\nFlow at ECMO System Drain = ', np.round(np.array(Qecmodrain[0]*0.06), 2), 'L/min; -', np.round(np.array(find_deviation_less(Qecmodrain[0], Qecmodrain[1])), 2), ' %', 
          ' +', np.round(np.array(find_deviation_plus(Qecmodrain[0], Qecmodrain[2])), 2), ' %')
    
    print('Pressure at ECMO System Drain = ', np.round(np.array(Pecmodrain[0]), 2), 'mmHg; -', np.round(np.array(find_deviation_less(Pecmodrain[0], Pecmodrain[1])), 2), ' %', 
          ' +', np.round(np.array(find_deviation_plus(Pecmodrain[0], Pecmodrain[2])), 2), ' %')

def print_mean_time_combinations_deviation(Psart, Psvn, Ppart, dPfil, Pcrrttuin, Pcrrttuout):
    """
    Print mean and deviation averaged over time for different quantities at the current rpm.

    Args:
        Psart (List): List containing the mean value and deviations of Psart.
        Psvn (List): List containing the mean value and deviations of Psvn.
        Ppart (List): List containing the mean value and deviations of Ppart.
        dPfil (List): List containing the mean value and deviations of dPfil.
        Pcrrttuin (List): List containing the mean value and deviations of Pcrrttuin.
        Pcrrttuout (List): List containing the mean value and deviations of Pcrrttuout.

    Examples:
        print_mean_time_combinations_deviation([mean_over_time(e) for e in MeasurementP['PsartStats']], 
                                    [mean_over_time(e) for e in MeasurementP['PsvnStats']],
                                    [mean_over_time(e) for e in MeasurementP['PpartStats']],
                                    [mean_over_time(e) for e in MeasurementP['dPfilStats']],
                                    [mean_over_time(e) for e in MeasurementP['PcrrttuinStats']],
                                    [mean_over_time(e) for e in MeasurementP['PcrrttuoutStats']])

        print_mean_time_combinations_deviation(MeasurementPrpm['Psart'][rpm], 
                                MeasurementPrpm['Psvn'][rpm],
                                MeasurementPrpm['Ppart'][rpm],
                                MeasurementPrpm['dPfil'][rpm],
                                MeasurementPrpm['Pcrrttuin'][rpm],
                                MeasurementPrpm['Pcrrttuout'][rpm])
    """

    print('\nPsart = ', np.round(np.array(Psart[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(Psart[0], Psart[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(Psart[0], Psart[2])), 2), '%')
    
    print('Psvn = ', np.round(np.array(Psvn[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(Psvn[0], Psvn[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(Psvn[0], Psvn[2])), 2), '%')
    
    print('Ppart = ', np.round(np.array(Ppart[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(Ppart[0], Ppart[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(Ppart[0], Ppart[2])), 2), '%')
    
    print('ΔPfil = ', np.round(np.array(dPfil[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(dPfil[0], dPfil[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(dPfil[0], dPfil[2])), 2), '%')
    
    print('Pcrrttuin = ', np.round(np.array(Pcrrttuin[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(Pcrrttuin[0], Pcrrttuin[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(Pcrrttuin[0], Pcrrttuin[2])), 2), '%')
    
    print('Pcrrttuout = ', np.round(np.array(Pcrrttuout[0]), 2), 'mmHg ; -', np.round(np.array(find_deviation_less(Pcrrttuout[0], Pcrrttuout[1])), 2), '%', 
        ' +', np.round(np.array(find_deviation_plus(Pcrrttuout[0], Pcrrttuout[2])), 2), '%')
    
def print_median_SD_of_mean_time(data, name):
    """
    Prints the median and standard deviation of the mean time for a given data set.

    Parameters:
    - data: A dictionary containing the data.
    - name: The name of the data set.

    Returns:
    -
    """

    # Get data type
    first_letter = name[0]

    if first_letter == 'P' or first_letter == 'd':
        unit = 'mmHg'
        conversion_factor = 1.
    elif first_letter == 'Q':
        unit = 'L/min'
        conversion_factor = 0.06  # Conversion from mL/min to L/min
    elif first_letter == 'V':
        unit = 'mL'
        conversion_factor = 1.
    else:
        unit = 'Unknown'
        conversion_factor = 1.

    # Calcuate median and standard deviation of the time averaged data
    dataList = np.array(list(data[name].values())) * conversion_factor

    data_mean = np.mean(dataList, axis=1)
    data_median = np.median(data_mean)
    data_STD = np.std(data_mean)
   
    print(f"{name} = {data_median:.2f} {unit} ; ±{data_STD:.2f} {unit}")

def find_combination_name(combinations, PTotal=False, VTotal=False, PCOI=False, VCOI=False, index=False):
    """ Gets the location where CRRT is connected to ECMO system for the combination of interest.

    Args:
        - combinations: list() -> list containing all possible combinations
        - PTotal: dict() -> dict containing pressure of all possible combinations
        - VTotal: dict() -> dict containing volume of all possible combinations
        - PCOI: array() -> array containing pressure of combination of interest
        - VCOI: array() -> array containing volume of combination of interest

    Return:
        - combiName: str() -> location of the combination of interest.
    """

    # If index is not provided, find the index of the combination of interest
    if index is False:
        if 'Plv0' in PTotal:
            ven = 'lv'
        else:
            ven = 'rv'

        i=0
        while (PTotal['P'+ven+str(i)]!=PCOI).all() and (VTotal['V'+ven+str(i)]!=VCOI).all():
            i = i+1

        index = i

    # Get names for given index
    CRRTdrain = convert_index_to_access(int(float(combinations[index].split('-')[0])), CRRT=True)
    CRRTreturn = convert_index_to_access(int(float(combinations[index].split('-')[1])), CRRT=True)

    combiName = CRRTdrain + '\n' + CRRTreturn

    return combiName

def create_structure_specific_rpm(pressuredata, flowdata, volumedata, pressureMeasures, flowMeasures, volumeMeasures, combinations):
    
    pressureStructure = {}
    flowStructure = {}
    volumeStructure = {}
    
    # Store pressure, flow and volume data for current combination
    for i in range(len(combinations)):
        for measure in pressureMeasures:
            if measure not in pressureStructure:
                pressureStructure[measure] = {}
            pressureStructure[measure][measure+str(i)] = pressuredata[measure + str(i)]
            
        for measure in flowMeasures:
            if measure not in flowStructure:
                flowStructure[measure] = {}
            flowStructure[measure][measure+str(i)] = flowdata[measure + str(i)]

        for measure in volumeMeasures:
            if measure not in volumeStructure:
                volumeStructure[measure] = {}
            volumeStructure[measure][measure+str(i)] = volumedata[measure + str(i)]

        if 'dPfil' not in pressureStructure:
            pressureStructure['dPfil'] = {}
        pressureStructure['dPfil']['dPfil'+str(i)] = pressuredata['Pcrrtfil'+str(i)] - pressuredata['Pcrrttuout'+str(i)]

    # Store mean, min and max values for current combination and store in 'Stats'
    for measure in pressureStructure.copy():
        pressureStructure[measure + 'Stats'] = [mean_over_combinations(combinations, str(measure), pressureStructure[measure]),
                                            min_over_combinations(combinations, str(measure), pressureStructure[measure]), 
                                            max_over_combinations(combinations, str(measure), pressureStructure[measure])]

    for measure in flowMeasures:
        flowStructure[measure + 'Stats'] = [mean_over_combinations(combinations, str(measure), flowStructure[measure]),
                                            min_over_combinations(combinations, str(measure), flowStructure[measure]), 
                                            max_over_combinations(combinations, str(measure), flowStructure[measure])]

    for measure in volumeMeasures:
        volumeStructure[measure + 'Stats'] = [mean_over_combinations(combinations, str(measure), volumeStructure[measure]),
                                            min_over_combinations(combinations, str(measure), volumeStructure[measure]), 
                                            max_over_combinations(combinations, str(measure), volumeStructure[measure])]
            
    return pressureStructure, flowStructure, volumeStructure

def create_structure_timemean_varying_rpm(pressureMeasures, flowMeasures, volumeMeasures, combinations, \
                                    pressuredata=False, flowdata=False, volumedata=False, \
                                    pressureMeasuresRPM=False, flowMeasuresRPM=False, volumeMeasuresRPM=False):
    """
    Create structure containing results for each combination in form of time averaged data for each rpm.
    data    ->  'Prv'   ->  'Prv0' -> [1000, 2000, 3000, ...]
                            'Prv1' -> [1000, 2000, 3000, ...]

    Args:
        - pressureMeasures: list() -> list containing all pressure measures
        - flowMeasures: list() -> list containing all flow measures
        - volumeMeasures: list() -> list containing all volume measures
        - combinations: list() -> list containing all possible combinations
        - pressuredata: dict() -> dict containing pressure data for all combinations
        - flowdata: dict() -> dict containing flow data for all combinations
        - volumedata: dict() -> dict containing volume data for all combinations
        - pressureMeasuresRPM: dict() -> dict containing pressure data for all combinations for each rpm
        - flowMeasuresRPM: dict() -> dict containing flow data for all combinations for each rpm
        - volumeMeasuresRPM: dict() -> dict containing volume data for all combinations for each rpm
    
        Returns:
        - pressureMeasuresRPM: dict() -> dict containing pressure data for all combinations for each rpm
        - flowMeasuresRPM: dict() -> dict containing flow data for all combinations for each rpm
        - volumeMeasuresRPM: dict() -> dict containing volume data for all combinations for each rpm

    """
    # Initializing data structure
    if pressuredata is False:
   
        pressureMeasuresRPM = {}
        flowMeasuresRPM = {}
        volumeMeasuresRPM = {}

        for i in range(len(combinations)):
            for measure in pressureMeasures:
                if measure not in pressureMeasuresRPM:
                    pressureMeasuresRPM[measure] = {}
                pressureMeasuresRPM[measure][measure+str(i)] = []
                
            for measure in flowMeasures:
                if measure not in flowMeasuresRPM:
                    flowMeasuresRPM[measure] = {}
                flowMeasuresRPM[measure][measure+str(i)] = []

            for measure in volumeMeasures:
                if measure not in volumeMeasuresRPM:
                    volumeMeasuresRPM[measure] = {}
                volumeMeasuresRPM[measure][measure+str(i)] = []

            if 'dPfil' not in pressureMeasuresRPM:
                pressureMeasuresRPM['dPfil'] = {}
            pressureMeasuresRPM['dPfil']['dPfil'+str(i)] = []
    
    # Append pressure, flow and volume data for current rpm to each combination
    else:
        for i in range(len(combinations)):
            for measure in pressureMeasures:
                pressureMeasuresRPM[measure][measure+str(i)].append(np.mean(pressuredata[measure][measure + str(i)]))
                
            for measure in flowMeasures:
                flowMeasuresRPM[measure][measure+str(i)].append(np.mean(flowdata[measure][measure + str(i)]))

            for measure in volumeMeasures:
                volumeMeasuresRPM[measure][measure+str(i)].append(np.mean(volumedata[measure][measure + str(i)]))

            pressureMeasuresRPM['dPfil']['dPfil'+str(i)].append(np.mean(pressuredata['Pcrrtfil']['Pcrrtfil'+str(i)] - pressuredata['Pcrrttuout']['Pcrrttuout'+str(i)]))

    return pressureMeasuresRPM, flowMeasuresRPM, volumeMeasuresRPM
          
def create_structure_statistical_mean_min_max(pressureMeasures, flowMeasures, volumeMeasures, rpmBatch, rpm=None, \
                                        pressuredata=False, flowdata=False, volumedata=False, \
                                        pressureMeasuresRPM=False, flowMeasuresRPM=False, volumeMeasuresRPM=False, \
                                        globalWriting=False, fitting=False):
    """
    Create structure containing results for each combination in form of mean, min and max values for each rpm.

    Args:
        - pressureMeasures: list() -> list containing all pressure measures
        - flowMeasures: list() -> list containing all flow measures
        - volumeMeasures: list() -> list containing all volume measures
        - rpmBatch: list() -> list containing all rpms
        - rpm: int() -> rpm of interest
        - pressuredata: dict() -> dict containing pressure data for all combinations
        - flowdata: dict() -> dict containing flow data for all combinations
        - volumedata: dict() -> dict containing volume data for all combinations
        - pressureMeasuresRPM: dict() -> dict containing pressure data for all combinations for each rpm
        - flowMeasuresRPM: dict() -> dict containing flow data for all combinations for each rpm
        - volumeMeasuresRPM: dict() -> dict containing volume data for all combinations for each rpm
        - globalWriting: bool() -> True if global writing is required
        - fitting: bool() -> True if fitting is required

    Returns:
        - pressureMeasuresRPM: dict() -> dict containing pressure data for all combinations for each rpm
        - flowMeasuresRPM: dict() -> dict containing flow data for all combinations for each rpm
        - volumeMeasuresRPM: dict() -> dict containing volume data for all combinations for each rpm

    """
    
    # Initializing data structure
    if globalWriting is False:
        if pressuredata is False:
    
            pressureMeasuresRPM = {}
            flowMeasuresRPM = {}
            volumeMeasuresRPM = {}

            for measure in pressureMeasures:
                if 'dPfil' not in pressureMeasuresRPM:
                    pressureMeasuresRPM['dPfil'] = {}
                    for rpm in rpmBatch:
                        pressureMeasuresRPM['dPfil'][rpm] = {}

                pressureMeasuresRPM[measure] = {}
                for rpm in rpmBatch:
                    pressureMeasuresRPM[measure][rpm] = {}

            for measure in flowMeasures:
                flowMeasuresRPM[measure] = {}
                for rpm in rpmBatch:
                    flowMeasuresRPM[measure][rpm] = {}

            for measure in volumeMeasures:
                volumeMeasuresRPM[measure] = {}
                for rpm in rpmBatch:
                    volumeMeasuresRPM[measure][rpm] = {}

        # Store mean, min and max averaged over time for the current rpm in MeasurementXXrpm
        else:
            for measure in pressureMeasuresRPM.copy():
                if measure == 'Plv' or measure == 'Prv':
                    pressureMeasuresRPM[measure][rpm] = [pressuredata[measure + 'Stats'][0],
                                                        pressuredata[measure + 'Stats'][1],
                                                        pressuredata[measure + 'Stats'][2]]
                    continue
                    
                pressureMeasuresRPM[measure][rpm] = [mean_over_time(pressuredata[measure + 'Stats'][0]),
                                                    mean_over_time(pressuredata[measure + 'Stats'][1]),
                                                    mean_over_time(pressuredata[measure + 'Stats'][2])]
            for measure in flowMeasures:
                flowMeasuresRPM[measure][rpm] = [mean_over_time(flowdata[measure + 'Stats'][0]),
                                                    mean_over_time(flowdata[measure + 'Stats'][1]),
                                                    mean_over_time(flowdata[measure + 'Stats'][2])]
            for measure in volumeMeasures:  
                volumeMeasuresRPM[measure][rpm] = [volumedata[measure + 'Stats'][0],
                                                    volumedata[measure + 'Stats'][1],
                                                    volumedata[measure + 'Stats'][2]]

    if globalWriting == True:
        for measure in pressureMeasuresRPM.copy():
            pressureMeasuresRPM[measure]['Global'] = {}
            pressureMeasuresRPM[measure]['Global']['mean'] = []
            pressureMeasuresRPM[measure]['Global']['min'] = []
            pressureMeasuresRPM[measure]['Global']['max'] = []

            for rpm in rpmBatch:
                pressureMeasuresRPM[measure]['Global']['mean'].append(pressureMeasuresRPM[measure][rpm][0])
                pressureMeasuresRPM[measure]['Global']['min'].append(pressureMeasuresRPM[measure][rpm][1])
                pressureMeasuresRPM[measure]['Global']['max'].append(pressureMeasuresRPM[measure][rpm][2])

        for measure in flowMeasures:
            flowMeasuresRPM[measure]['Global'] = {}
            flowMeasuresRPM[measure]['Global']['mean'] = []
            flowMeasuresRPM[measure]['Global']['min'] = []
            flowMeasuresRPM[measure]['Global']['max'] = []
            
            if measure == 'Qecmopump':
                    flowMeasuresRPM[measure]['Global']['rpm'] = []

            for rpm in rpmBatch:
                if measure == 'Qecmopump':
                    flowMeasuresRPM[measure]['Global']['rpm'].append(
                        np.round(np.array(flowMeasuresRPM[measure][rpm][0]/16.667), 2))
                
                flowMeasuresRPM[measure]['Global']['mean'].append(flowMeasuresRPM[measure][rpm][0])
                flowMeasuresRPM[measure]['Global']['min'].append(flowMeasuresRPM[measure][rpm][1])
                flowMeasuresRPM[measure]['Global']['max'].append(flowMeasuresRPM[measure][rpm][2])

        for measure in volumeMeasures:
            volumeMeasuresRPM[measure]['Global'] = {}
            volumeMeasuresRPM[measure]['Global']['mean'] = []
            volumeMeasuresRPM[measure]['Global']['min'] = []
            volumeMeasuresRPM[measure]['Global']['max'] = []

            for rpm in rpmBatch:
                volumeMeasuresRPM[measure]['Global']['mean'].append(volumeMeasuresRPM[measure][rpm][0])
                volumeMeasuresRPM[measure]['Global']['min'].append(volumeMeasuresRPM[measure][rpm][1])
                volumeMeasuresRPM[measure]['Global']['max'].append(volumeMeasuresRPM[measure][rpm][2])

        if fitting is True:
            for measure in pressureMeasuresRPM.copy():

                if measure == 'Plv' or measure == 'Prv': continue

                pressureMeasuresRPM[measure]['Global']['mean'] = fitted_curve(
                    flowMeasuresRPM['Qecmopump']['Global']['rpm'], pressureMeasuresRPM[measure]['Global']['mean'])
                pressureMeasuresRPM[measure]['Global']['min'] = fitted_curve(
                    flowMeasuresRPM['Qecmopump']['Global']['rpm'], pressureMeasuresRPM[measure]['Global']['min'])
                pressureMeasuresRPM[measure]['Global']['max'] = fitted_curve(
                    flowMeasuresRPM['Qecmopump']['Global']['rpm'], pressureMeasuresRPM[measure]['Global']['max'])

    return pressureMeasuresRPM, flowMeasuresRPM, volumeMeasuresRPM

pressures={}
flows={}
volumes={}

def launch_study(combinations, pressureMeasures, flowMeasures, volumeMeasures, patientId, 
                 patientFitted, plotFunctionOfECMOPF=False, rpmBatch=[], curveIdentifier=[]):
    
    if plotFunctionOfECMOPF==True:
        ###############################################################################
        ########### Study effect of CRRT for different ECMO pump flows ################
        ###############################################################################

        ########### Launch the model for each combination ###########            
        print('\n---- Studying the effect of CRRT connections on pressures and flows accross rpms of the ECMO pump. ---')
        print('Modeling and collecting the data...')

        timeAveragePRPM, timeAverageQRPM, timeAverageVRPM = \
            create_structure_timemean_varying_rpm(pressureMeasures, flowMeasures, volumeMeasures, combinations)

        # Initializing Data Structure that contains results for all RPMs.
        MeasurementPrpm, MeasurementQrpm, MeasurementVrpm = \
            create_structure_statistical_mean_min_max(pressureMeasures, flowMeasures, volumeMeasures, rpmBatch)

        for rpm in rpmBatch:

            print('\n=======================')
            print('RPM: ' + str(rpm))
            print('=======================\n')

            # Launch the model for each combinations
            i = 0
            for c in combinations:
                print('Connection: ' + c + ' ⌛', end=' ')
                blockPrint()
                # Convert combination into access (access_list)
                CRRTdrainAccess=str(convert_index_to_access(int(float(c.split('-')[0]))))
                CRRTreturnAccess=str(convert_index_to_access(int(float(c.split('-')[1]))))
                
                initTree, simPostTree, simCompTree = initialize_study(patientId, patientFitted)

                initTree.CRRT['access'].update({'drain': {CRRTdrainAccess: 1}})
                initTree.CRRT['access'].update({'return': {CRRTreturnAccess: 1}})

                initTree.ECMO['pump'].update({'rpm': rpm})
                
                simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)

                # Store the results
                for p in pressureMeasures:
                    pressures.update({p+str(i): simPostTree.Pressures[p]})
        
                for f in flowMeasures:
                    flows.update({f+str(i): simPostTree.Flows[f]})

                for v in volumeMeasures:
                    volumes.update({v+str(i): simPostTree.Volumes[v]})
            
                i+=1
                enablePrint()
                print('✅')

            ########### Post-processing of results ###########
            # Create dictionaries to store the results.These dict() always contain results for 
            # for the latest rpm in analysis.
            MeasurementP, MeasurementQ, MeasurementV = \
                create_structure_specific_rpm(pressures, flows, volumes, pressureMeasures, flowMeasures, volumeMeasures, combinations)
            
            timeAveragePRPM, timeAverageQRPM, timeAverageVRPM = \
                create_structure_timemean_varying_rpm(pressureMeasures, flowMeasures, volumeMeasures, \
                                                             combinations, \
                                                             MeasurementP, MeasurementQ, MeasurementV,
                                                             timeAveragePRPM, timeAverageQRPM, timeAverageVRPM)
                    
            # Store mean, min and max averaged over time for the current rpm in MeasurementXXrpm
            MeasurementPrpm, MeasurementQrpm, MeasurementVrpm = \
                create_structure_statistical_mean_min_max(pressureMeasures, flowMeasures, volumeMeasures, rpmBatch, rpm, \
                                                pressuredata=MeasurementP, flowdata=MeasurementQ, volumedata=MeasurementV, \
                                                pressureMeasuresRPM=MeasurementPrpm, flowMeasuresRPM=MeasurementQrpm, \
                                                volumeMeasuresRPM=MeasurementVrpm)

            # Print median and STD of time averaged quantities
            print('\nMedian and standard deviation of time averaged quantities:')
            print_median_SD_of_mean_time(MeasurementP, 'Psart')
            print_median_SD_of_mean_time(MeasurementP, 'Psvn')
            print_median_SD_of_mean_time(MeasurementP, 'Ppart')
            print_median_SD_of_mean_time(MeasurementP, 'dPfil')
            print_median_SD_of_mean_time(MeasurementP, 'Pcrrttuin')
            print_median_SD_of_mean_time(MeasurementP, 'Pcrrttuout')
            print_median_SD_of_mean_time(MeasurementQ, 'Qecmoreturn')
            print_median_SD_of_mean_time(MeasurementP, 'Pecmoreturn')
            print_median_SD_of_mean_time(MeasurementQ, 'Qecmodrain')
            print_median_SD_of_mean_time(MeasurementP, 'Pecmodrain')
        
        # Rearrange data structure for plotting mean P, Q and V for different rpm's
        print('Data treatment across rpm (mean, deviation) ...')

        MeasurementPrpm, MeasurementQrpm, MeasurementVrpm = \
                create_structure_statistical_mean_min_max(pressureMeasures, flowMeasures, volumeMeasures, rpmBatch, \
                                                pressuredata=MeasurementP, flowdata=MeasurementQ, volumedata=MeasurementV, \
                                                pressureMeasuresRPM=MeasurementPrpm, flowMeasuresRPM=MeasurementQrpm, \
                                                volumeMeasuresRPM=MeasurementVrpm, globalWriting=True, fitting=True)
            
        print('Data treatment done.')
        print('Plotting the results...')

        combined_plot_varying_rpm(MeasurementQrpm, timeAveragePRPM, timeAverageQRPM, curveIdentifier, combinations)

        print('Plotting done.')
        print('End of the study')
        
    else:
        #############################################################
        ########### Study effect of CRRT for one rpm ################
        #############################################################

        ########### Launch the model for each combination ###########
        print('\n---- Studying the effect of CRRT connections on pressures and flows ---')
        print('Modeling and collecting the data...')

        i = 0
        for c in combinations:

            print('Connection: ' + c + ' ⌛', end=' ')
            blockPrint()

            # Convert combination into access (access_list)
            CRRTdrainAccess=str(convert_index_to_access(int(float(c.split('-')[0]))))
            CRRTreturnAccess=str(convert_index_to_access(int(float(c.split('-')[1]))))
            
            initTree, simPostTree, simCompTree = initialize_study(patientId, patientFitted)

            initTree.CRRT['access'].update({'drain': {CRRTdrainAccess: 1}})
            initTree.CRRT['access'].update({'return': {CRRTreturnAccess: 1}})
        
            simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)

            # Store the results
            for p in pressureMeasures:
                pressures.update({p+str(i): simPostTree.Pressures[p]})
            for f in flowMeasures:
                flows.update({f+str(i): simPostTree.Flows[f]})
            for v in volumeMeasures:
                volumes.update({v+str(i): simPostTree.Volumes[v]})
            i+=1

            enablePrint()
            print('✅')

        print('Modeling done.')
        print('Data treatment (mean, deviation) ...')
     
        ########### Post-processing of results ###########
        # Create dictionaries to store the results.
        MeasurementP, MeasurementQ, MeasurementV = \
            create_structure_specific_rpm(pressures, flows, volumes, \
                                          pressureMeasures, flowMeasures, volumeMeasures, combinations)
        
        print('Data treatment done.')
        print('Plotting the results...')

        ########### Plotting of results ###########
        TimeNormalized=(initTree.timeTree['T']-initTree.timeTree['T'][0])*initTree.clinicalData['bpm']/60
        indices_ESV_sorting = combined_plot_static_rpm(TimeNormalized, combinations, MeasurementP, MeasurementV)
    
        # Print median and STD of time averaged quantities
        print('\nMedian and standard deviation of time averaged quantities:')
        print_median_SD_of_mean_time(MeasurementP, 'Psart')
        print_median_SD_of_mean_time(MeasurementP, 'Psvn')
        print_median_SD_of_mean_time(MeasurementP, 'Ppart')
        print_median_SD_of_mean_time(MeasurementP, 'dPfil')
        print_median_SD_of_mean_time(MeasurementP, 'Pcrrttuin')
        print_median_SD_of_mean_time(MeasurementP, 'Pcrrttuout')
        print_median_SD_of_mean_time(MeasurementQ, 'Qecmoreturn')
        print_median_SD_of_mean_time(MeasurementP, 'Pecmoreturn')
        print_median_SD_of_mean_time(MeasurementQ, 'Qecmodrain')
        print_median_SD_of_mean_time(MeasurementP, 'Pecmodrain')
        
        print('Plotting done.')
        print('End of the study')

        return indices_ESV_sorting

def initialize_study(patientId, patientFitted):

    runtimeTree = initialization.parseRunTimeVariables()

    runtimeTree['patientId'] = patientId
    runtimeTree['patientFitted'] = patientFitted

    initTree = model_main.parameter_pytree(runtimeTree, runtimeTree['patientId'], runtimeTree['patientFitted'])
    simPostTree = model_main.simPost_pytree()
    simCompTree = model_main.simComp_pytree()

    # Set CRRT status to 'on'
    initTree.CRRT.update({'status': 1.0})

    return initTree, simPostTree, simCompTree

#####################################
########### STUDY MANAGER ###########
#####################################

if __name__ == '__main__':

    # If patientFitted = False, it runs the study with default parameters, no matter what is specified
    # for patientId
    patientFitted = True
    patientId = 1

    # ECMO rpm to study; ECMO pump corresponding single rpm needs to specified in patient.txt
    rpmBatch = [x for x in range(1000, 5001, 250)]
    #rpmBatch = [x for x in range(1000, 5001, 1000)]
    
    # These are the combinations to be tested; make sure to activate CRRT; otherwise you see just the effect of ECMO flow on the CVS
    # Make sure that the combinations are corresponding the access from the access_list.txt
    #combinations = ['13-12', '12-12', '15-14', '14-15', '14-16', '15-16', '15-12', '14-12', '13-14']
    combinations=['13-12', '12-12', '15-14']

    # Pressures, Flows and Volumes to collect (if you want to add a quantity, please add the corresponding treatment and plotting in the code)
    pressureMeasures = ['Prv', 'Psart', 'Psvn', 'Ppart', 'Pcrrtfil', 'Pcrrttuin', 'Pcrrttuout', \
                        'Pecmodrain', 'Pecmoreturn'] #pdrop across filter = pcrrtfil-Pcrrtuout
    flowMeasures = ['Qecmopump', 'Qecmodrain', 'Qecmoreturn']
    volumeMeasures = ['Vrv']

    # Launch the study with fixed rpm and get ESV ranking of the combinations
    indices_ESV_sorting = launch_study(combinations, pressureMeasures, flowMeasures, volumeMeasures, patientId, patientFitted)

    # Launch the study with varying rpm and use ESV ranking from static run for plotting
    launch_study(combinations, pressureMeasures, flowMeasures, volumeMeasures, patientId, patientFitted, \
                 plotFunctionOfECMOPF=True, rpmBatch=rpmBatch, curveIdentifier=indices_ESV_sorting)

#####################################
########### STUDY MANAGER ###########
#####################################