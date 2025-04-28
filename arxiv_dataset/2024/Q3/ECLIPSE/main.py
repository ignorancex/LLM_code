"""

This is the script to launch if you want to explore the model.
Just choose a run: 'p', 's' or 'f' for respectively production, sensitivity, fitting
This script initializes pytrees and data for the model, before launching
the corresponding run.

"""

import os
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import sys
import math
import model_main
import sensitivity_analysis
import model_plot
import model_post_process
import optimization
import matplotlib.pyplot as plt
import time
import initialization
import file_manager

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_log_compiles', False)
jax.config.update('jax_platform_name', 'cpu') # ensures we use the CPU
jax.config.update("jax_debug_nans", False)

os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_JOB_ID") else int(os.cpu_count())}'

#os.environ["EQX_ON_ERROR"] = 'breakpoint'
#os.environ["EQX_ON_ERROR_BREAKPOINT_FRAMES"] = 100

def analyze_ECMO_behavior(pump = 'DP3'):

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral' 
    plt.rcParams['font.size'] = 16 

    cm = 1/2.54

    # Test VA ECMO
    plt.figure(figsize=(32*cm, 11*cm))

    # Healthy without ECMO
    print('< Running healthy patient >')
    initTree, simPostTree, simCompTree = initialization.pytrees_init(patientId, patientFitted, runtimeTree)
    initTree.clinicalData['ECMOactive'] = 0
    initTree.ECMO.update({'status': 0})
    initTree.paramsModel['Emaxlv'] = 2.8
    simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)
    plt.plot(simPostTree.Volumes['Vlv'], simPostTree.Pressures['Plv'], label='Healthy')

    # Cardiac failure without ECMO
    print('< Running cardiac failure patient >')
    initTree.paramsModel['Emaxlv'] = 0.6
    simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)
    plt.plot(simPostTree.Volumes['Vlv'], simPostTree.Pressures['Plv'], label='Cardiac failure')

    # Cardiac failure with VA ECMO
    initTree.clinicalData['ECMOactive'] = 1
    initTree.ECMO.update({'status': 1})
    initTree.clinicalData['ECMOtype'] = 2
    initTree.clinicalData['ECMOdrainAccess'] = 3
    initTree.clinicalData['ECMOreturnAccess'] = 0
    initTree.ECMO.update({'access': {'drain': {'ra': 1}, 'return': {'ao': 1}}})

    # Read all pump parameters
    with open(file_manager.find_path('pump'), 'r') as file:
        file_content = file.readlines()

    params = {}
    for line in file_content[1:]:
        parts = line.strip().split()
        params[parts[0]] = [float(x) for x in parts[1:(len(parts)-2)]]

    if pump == 'DP3':

        initTree.clinicalData['ECMOpump'] = 0
        initTree.ECMO['pump'].update({'centrifugal': np.array(params['DP3'])})
        rpmRange = [4000, 5000, 6000]

    elif pump == 'Rotaflow':

        initTree.clinicalData['ECMOpump'] = 1
        initTree.ECMO['pump'].update({'centrifugal': np.array(params['Rotaflow'])})
        rpmRange = [1000, 2000, 3000]

    else:

        sys.exit('Not allowed pump chosen. Please use Rotaflow or DP3.')
    
    print('< Running VA ECMO with ' + pump + ' pump >')
    print('< Check patient_data.txt for cannulae used!')
        
    for rpm in rpmRange:
        print('< Running ECMO at ' + str(rpm) + ' rpm >')
        
        initTree.clinicalData['ECMOrpm'] = rpm
        initTree.ECMO['pump'].update({'rpm': rpm})

        simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)
        plt.plot(simPostTree.Volumes['Vlv'], simPostTree.Pressures['Plv'], label=str(rpm)+' Q: ' + str(round(jnp.mean(simPostTree.Flows['Qecmopump']), 2)) + ' mL/s')
    
    plt.title('pV-Loops for different patient conditions. ECMO using ' + pump + ' pump')
    plt.legend(fontsize=10, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('Volume in mL')
    plt.ylabel('Pressure in mmHg')
    plt.tight_layout()
    plt.savefig('Plots/Analysis_ECMO_Behavior.png')

if __name__ == '__main__':
    
    runtimeTree = initialization.parseRunTimeVariables()

    device_count = int(os.environ["XLA_FLAGS"].split("=")[1])
    print("----------------------")
    print(f"{device_count} core(s) available.")
    print("----------------------")

    ###### Patient Management #######
    patientId=runtimeTree['patientId']
    patientFitted=runtimeTree['patientFitted']

    # If it is on True, automatically adapts GSA, optimization and production runs 
    # based on specific parameter model for the patient 
    # (You must do a fitting for the current patient with the default config of 
    # parameters model before setting it to True)
    #########################

    # Module for sensitivity analysis
    if runtimeTree['runType'] == 's':

        print('Launching the sensitivity analysis...')
       
        GSATree, initTree, simPostTree, simCompTree, paramGroupGSA = initialization.pytrees_init(patientId, patientFitted, 
                                                                                  runtimeTree, runtimeTree['perturbation'])

        if runtimeTree['checkConvergenceGSA'] == True:
            sampleSize = [2**i for i in range(1, int(math.log(runtimeTree['sampleSize'], 2)) + 1)]
            print('\nSamples used for convergence check of GSA: ', sampleSize)
        else:
            sampleSize = [runtimeTree['sampleSize']]

        start = time.perf_counter()
        for ii in range(len(sampleSize)):
            problem = sensitivity_analysis.main(GSATree, sampleSize[ii], initTree, simPostTree, simCompTree, paramGroupGSA)
        end = time.perf_counter()

        print('Sensitivity analysis done.')
        print(f"Took {round(end - start, 2)} seconds.")

    # Module for parameter identification  
    elif runtimeTree['runType'] == 'f':

        print('Launching the fitting...')
       
        clinicalData, x0, bounds, x0_batch, initTree, simPostTree, simCompTree = \
            initialization.pytrees_init(patientId, patientFitted, runtimeTree)

        start=time.perf_counter()
        optimization.main(clinicalData, x0, bounds, x0_batch, initTree, simPostTree, simCompTree, \
                          runtimeTree['exploreParameterSpace'])
        end=time.perf_counter()

        print('Fitting done.')
        print(f"Took {round(end - start, 2)} seconds.")

    # Production run 
    elif runtimeTree['runType'] == 'p':

        print('Modeling ...')
        start = time.perf_counter()

        if runtimeTree['analyzeECMObehaviour']: analyze_ECMO_behavior(pump='DP3') # 'Rotaflow' or 'DP3'

        initTree, simPostTree, simCompTree = initialization.pytrees_init(patientId, patientFitted, runtimeTree)
        simCompTree, simPostTree = model_main.modeling(initTree, simPostTree, simCompTree)
        end = time.perf_counter()
        print('Production done.')
        print('Timings:')
        print(f"Took {round(end - start, 2)} seconds.")

        # Plotting the results
        model_plot.plot_results(initTree, simPostTree)    
    
    if runtimeTree['saveSolutionToCSV'] == 1:
        if runtimeTree['runType'] in ['p','f']:
            model_post_process.saveSimulationResultsToCSVFile(initTree,simPostTree)