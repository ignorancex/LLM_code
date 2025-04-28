"""

This file manages the sensitivity analysis also called GSA.

"""
import seaborn as sns
import matplotlib.pyplot as plt
from SALib import ProblemSpec
from SALib.util import extract_group_names, _check_groups
import jax
import equinox
import numpy as np
import pandas as pd
import os
import csv
from math import ceil
import model_main
from jax import jit, vmap, pmap
from jax.experimental.maps import xmap
from jax.sharding import Mesh
from joblib import Parallel, delayed
from scipy.stats.qmc import LatinHypercube
from SALib.sample import saltelli
from SALib.analyze import sobol

import file_manager
import initialization

jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_JOB_ID") else int(os.cpu_count())}'

class GSA_pytree(equinox.Module):

    """ Initializes the GSATree of the model (a pytree) which contains every information (clinicalData, resultsTmp, paramSensivitity)

    Args:
        - clinicalData: dict() -> clinical data of the current patient
        - resultsTmp: dict() -> output results for GSA (datas comparable to clinicalData)
        - params: dict() -> names and bounds of parameters for sensivitiy analysis
        - perturbation: jax.Array -> perturbation of the parameters for sensitivity analysis
        - convergenceAnalysis: bool() -> if True, the convergence analysis is conducted
        - includeECLSGSA: bool() -> if True, ECMO and CRRT parameters are included in GSA

    Returns:
        An instance of the class (an GSATree object).
    """

    clinicalData: dict()
    resultsTmp: dict()
    params: dict()
    perturbation: jax.Array
    convergenceAnalysis: bool()
    includeECLSGSA: bool()

    def __init__(self, clinicalData, resultsTmp, paramGSAOrdered, perturbation, \
                 convergenceAnalysis, includeECLSGSA):
        
        self.clinicalData = clinicalData
        self.resultsTmp = resultsTmp
        self.params = paramGSAOrdered
        self.perturbation = perturbation
        self.convergenceAnalysis = convergenceAnalysis
        self.includeECLSGSA = includeECLSGSA

    def update(self, update):
  
        # Update temp dict for results of GSA for each sample, s.th. it can be stored in y[ii] later
        keys = tuple(self.resultsTmp.keys())
        vals = tuple([update[key] for key in keys])

        replace = dict(zip(keys, vals))

        return equinox.tree_at(lambda tree: tree.resultsTmp, self, replace)
    
def main(GSATree, sample_size, initTree, simPostTree, simCompTree, paramGroupGSA):
    
    """ Orders the parameters, bounds and outputs as a problem (ProblemSpec),
    Divides and distributes the problem through local devices.
    Launch the GSA.
    Plots the results.

    Args:
        - GSATree: pytree() instance of GSA_pytree
        - sample_size: [int] -> [2**1], [2**2], ... [2**10]
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main
        - paramGroupGSA: dict() -> Containing parameter names and its groups

    Returns:
        ProblemSpec() instance -> problem.
    """

    problem = ProblemSpec(
        {
            'num_vars': len(GSATree.params),
            'names': list(GSATree.params.keys()),
            'bounds': list(GSATree.params.values()),
            'outputs': list(GSATree.resultsTmp.keys()),
            'groups': list(paramGroupGSA.values()) if paramGroupGSA is not None else None
        }
    )
    
    problem.sample_sobol(sample_size)
   
    if GSATree.includeECLSGSA == True:
    
        problem, param_indices_ECLS, param_indices_CVS = sample_ECLS(problem, GSATree)

        results = Parallel(n_jobs=int(os.environ["XLA_FLAGS"].split("=")[1]), backend="threading")(delayed(process_sample)(problem.samples[i], param_indices_ECLS, param_indices_CVS, GSATree, \
                                            initTree, simPostTree, simCompTree) for i in range(len(problem.samples)))
        
        # As np.array( LIST ) performs vertical stacking its similar to np.transpose(np.stack(results, axis=1))
        Y = np.array(results)
        
    else:

        all_devices = jax.local_devices()
        sample_count = len(problem.samples)
        device_count = len(all_devices)

        if sample_count <= device_count:
            num_devices = sample_count
        # highest number of devices num_devices <= len(all_devices) that divides len(problem.samples) 
        # evenly
        else:
            for divisor in range(device_count, 0, -1):
                if sample_count % divisor == 0:
                    num_devices = divisor
                    break

        devices = np.array(all_devices[:num_devices])

        print("\n-----------------------------------------")
        print("-----------------------------------------\n")
        print(f"Devices used for sensitivity analysis:")
        print(f"{devices}")
        print("\n-----------------------------------------")
        print("-----------------------------------------\n")

        problem.set_samples(problem.samples)
        eval_xmap = xmap(evaluate, in_axes=(['Sample', ...], [...], [...], [...], [...]), out_axes=(['Sample', ...]),
                                    axis_resources={'Sample': 'x'})
        with Mesh(devices, ('x',)):
            Y = eval_xmap(problem.samples, GSATree, initTree, simPostTree, simCompTree)

        Y = np.transpose(np.array(Y))
    
    problem.set_results(Y)
    problem.analyze_sobol()

    post_processing(problem, GSATree)

    return problem

@jit
def evaluate(SensitivitySample, GSATree, initTree, simPostTree, simCompTree):

    """ Evaluate the model for the specified SensitivitySample.

    Args:
        - SensitivitySample
        - GSATree: pytree() -> instance of GSA_pytree
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        list() -> output results for the specified SensitivitySample.
    """
   
    initTree = initTree.update(dict(zip((list(GSATree.params.keys())), SensitivitySample)))
    simulatedData = model_main.modeling(initTree, simPostTree, simCompTree)[0]
    GSATree = GSATree.update(simulatedData.results)

    return list([*GSATree.resultsTmp.values()])

def evaluateEC(SensitivitySample, index, indexBasicParams, GSATree, initTree, simPostTree, simCompTree):

    """ Evaluate the model for the specified SensitivitySample.

    Args:
        - SensitivitySample
        - index: [int] -> index of ECMOrpm, CRRTdrain, CRRTreturn
        - indexBasicParams: [int] -> index of all other parameters
        - GSATree: pytree() -> instance of GSA_pytree
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        list() -> output results for the specified SensitivitySample.
    """
    # index = [ecmorpm index, crrtdrain index, crrtreturn index]
    initTree = initTree.update(dict(zip((np.take(list(GSATree.params.keys()), indexBasicParams)), np.take(SensitivitySample, indexBasicParams))))
    initTree.ECMO['pump'].update({'rpm': SensitivitySample[index['rpm']]})
    initTree.CRRT['access'] = {'drain': {initialization.convert_index_to_access(SensitivitySample[index['drain']]): 1}, 'return': {initialization.convert_index_to_access(SensitivitySample[index['return']]): 1}}
    simulatedData = model_main.modeling(initTree, simPostTree, simCompTree)[0]
    GSATree = GSATree.update(simulatedData.results)

    return list([*GSATree.resultsTmp.values()])

def process_sample(sample, index, indexBasicParams, GSATree, initTree, simPostTree, simCompTree):
    return evaluateEC(sample, index, indexBasicParams, GSATree, initTree, simPostTree, simCompTree)

def sample_ECLS(problem, GSATree):
    """
    Creates sample for drain and return access of CRRT when including ECLS into sensitivity analysis.

    Args:
        problem: dict() -> The problem object of GSA containing the samples.
        - GSATree: pytree() -> instance of GSA_pytree.

    Returns:
        - problem: dict() -> The updated problem object with the samples.
        - param_indices_ECLS: List() -> A dictionary mapping the ECLS parameter names to their indices in the GSATree.
        - param_indices_CVS: List() -> A list of indices representing the non-ECLS parameters in the GSATree.
    """

    param_names = ['rpm', 'drain', 'return']

    combinations = [
        (13, 12), (12, 12), (15, 14), (14, 15), 
        (14, 16), (15, 16), (15, 12), (14, 12), (13, 14)
    ]

    param_indices_ECLS = {name: list(GSATree.params.keys()).index(name) for name in param_names}    
    param_indices_CVS = [i for i in range(len(list(GSATree.params.keys()))) if i not in list(param_indices_ECLS.values())]

    sampler = LatinHypercube(d=1)
    sample_indices_combinations = sampler.integers(l_bounds=0, \
                                u_bounds=len(combinations) -1, \
                                n=len(problem.samples[:, 0]), endpoint=True)
    
    sample_combinations = np.array([combinations[i[0]] for i in sample_indices_combinations])

    samples = problem.samples.copy()

    samples[:, param_indices_ECLS['drain']] = sample_combinations[:, 0]
    samples[:, param_indices_ECLS['return']] = sample_combinations[:, 1]

    problem.set_samples(samples)

    print('Total GSA samples', len(samples))

    return problem, param_indices_ECLS, param_indices_CVS

def post_processing(problem, GSATree):

    """ Post process the problem, printing and plotting.

    Args:
        - problem: ProblemSpec() instance -> the problem
        - GSATree: pytree() instance of GSA_pytree

    Returns:
        X.
    """

    categories = {
        'Volume': {
            'keys': ['ESVLV', 'EDVLV', 'ESVLA', 'EDVLA', 'ESVRV', 'EDVRV', 'ESVRA', 'EDVRA'],
            'unit': 'mL'
        },
        'Pressure': {
            'keys': ['SP', 'DP', 'MAP', 'SPAP', 'DPAP', 'MPAP', 'PCWP'],
            'unit': 'mmHg'
        },
        'Flow': {
            'keys': ['CO', 'PF'],
            'unit': 'L/min'
        }
    }

    # Create sorting of outputs w.r.t. categories
    sorted_outputs = []
    outputs_index_map = []  # To track the original indices for sorting ST and SDiff
    for category, details in categories.items():
        for key in details['keys']:
            if key in problem['outputs']:
                sorted_outputs.append((key, details['unit']))
                outputs_index_map.append(problem['outputs'].index(key))

    # Create directory
    base_dir = file_manager.sensitivityFolder + 'Results'
    perturbation_dir = os.path.join(base_dir, str(GSATree.perturbation) + '%')

    if not os.path.exists(perturbation_dir):
        os.makedirs(perturbation_dir)

    # Plotting
    plot_density_functions(problem, GSATree, perturbation_dir, sorted_outputs)

    print(problem)

    heatmap(problem, GSATree, perturbation_dir, sorted_outputs, outputs_index_map)

def heatmap(problem, GSATree, perturbation_dir, sorted_outputs, outputs_index_map):

    """ Plots the problem through a heatmap.

    Args:
        - problem: ProblemSpec() instance -> the problem
        - GSATree: pytree() instance of GSA_pytree
        - perturbation_dir: str() -> path to the directory where the plots and csv are stored.
        - categories: dict() -> Dictionary containing categories of outputs.

    Returns:
        -
    """
    # Get names in sorted_outputs
    sorted_outputs = [tuple[0] for tuple in sorted_outputs]

    groups = _check_groups(problem)

    if not groups:
        namesGSA = problem['names']
    else:
        namesGSA, _ = extract_group_names(groups)

    # Get most important parameters making up 90% of total sensitivity for each output
    ST, SDiff = get_important_params(problem, GSATree, perturbation_dir, namesGSA)

    # Reorder ST and SDiff according to sorted_outputs
    ST = ST[outputs_index_map, :]
    SDiff = SDiff[outputs_index_map, :]

    # Write ST and SDiff to .csv
    if GSATree.convergenceAnalysis: write_sensitivities(problem, ST, SDiff, GSATree, perturbation_dir, \
                                                        namesGSA, sorted_outputs)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral' 
    plt.rcParams['font.size'] = 10

    cm = 1/2.54
    fig_height = 9 * cm

    if ST.shape[1] > 8:
        fig_width = 19 * cm
        aspect = 40
    else:
        fig_width = 9 * cm
        aspect = 20

    # Plotting heatmap for ST
    fig1, axs1 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    im1 = axs1.imshow(ST)
    axs1.yaxis.set_ticks(range(0, len(sorted_outputs)))
    axs1.yaxis.set_ticklabels(sorted_outputs)
    axs1.xaxis.set_ticks(range(0, ST.shape[1]))
    axs1.xaxis.set_ticklabels(namesGSA, rotation=45, ha='right', va='top', rotation_mode='anchor')

    if not GSATree.convergenceAnalysis: im1.set_clim(0, 1)

    fig1.tight_layout()
    fig1.colorbar(im1, ax=axs1, location='top', orientation='horizontal', 
                    label='Total Order Sensitivity Index', aspect=aspect)

    file_name1 = os.path.join(perturbation_dir, 'Result_ST_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId'])) 

    fig1.savefig(file_name1 + '.pdf')
    fig1.savefig(file_name1 + '.tiff')
    fig1.savefig(file_name1 + '.jpg')
  
    # Plotting heatmap for ST-S1
    fig2, axs2 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    im2 = axs2.imshow(SDiff)
    axs2.yaxis.set_ticks(range(0, len(sorted_outputs)))
    axs2.yaxis.set_ticklabels(sorted_outputs)
    axs2.xaxis.set_ticks(range(0, SDiff.shape[1]))
    axs2.xaxis.set_ticklabels(namesGSA, rotation=45, ha='right', va='top', rotation_mode='anchor')

    if not GSATree.convergenceAnalysis: im2.set_clim(0, 1)
    
    fig2.tight_layout()
    fig2.colorbar(im2, ax=axs2, location='top', orientation='horizontal', 
                    label='Difference Between Total and First Order Sensitivity Index', aspect=40)

    file_name2 = os.path.join(perturbation_dir, 'Result_SDiff_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId']))      

    fig2.savefig(file_name2 + '.pdf')
    fig2.savefig(file_name2 + '.tiff')
    fig2.savefig(file_name2 + '.jpg')

def get_important_params(problem, GSATree, perturbation_dir, namesGSA):
    """
    Calculate the important parameters based on sensitivity analysis.

    Args:
        - problem: dict() -> The problem dictionary containing the analysis results.
        - GSATree: pytree() -> Instance of GSA_pytree
        - perturbation_dir: str -> The directory fo results with given perturbation.
        - namesGSA: list() -> A list containing parameter or group names.

    Returns:
        - ST: np.array() -> A numpy array containing the total order sensitivity indices (ST) for each output.
        - SDiff: np.array() -> A numpy array containing the difference between total order sensitivity indices 
                                (ST) and first order sensitivity indices (S1) for each output.
    """

    ST = [None]*len(problem['outputs'])
    S1 = [None]*len(problem['outputs'])
    most_important_parameters_per_output = []

    print("Sum of Total Order Sensitivity Indices for each output:")
    
    # Get parameters that make up 90% of ST for each output
    ii = 0
    for key in problem.analysis.keys():
        ST[ii] = problem.analysis[key]['ST']
        S1[ii] = problem.analysis[key]['S1']

        total_sensitivity = np.sum(ST[ii])
        print(f"{key}: {np.round(total_sensitivity, 4)}")
        
        # Pair parameters with their total order sensitivity (ST)
        param_ST_pairs = list(zip(namesGSA, ST[ii]))
        
        # Sort parameters by their total order sensitivity in descending order
        sorted_parameters = sorted(param_ST_pairs, key=lambda x: x[1], reverse=True)
        
        # Accumulate parameters until reaching 90% of total sensitivity
        cum_sum = 0
        important_parameters = []
        for param, sensitivity in sorted_parameters:
            cum_sum += sensitivity
            important_parameters.append(param)
            if cum_sum >= 0.9 * total_sensitivity:
                break
        
        most_important_parameters_per_output.append(important_parameters)

        ii += 1
    
    # Union of all important parameters across all outputs
    MIP_overall = set.union(*map(set, most_important_parameters_per_output))
    print("\nMost important parameters overall:\n", MIP_overall)

    ST = np.array(ST)
    S1 = np.array(S1)
    SDiff = ST - S1

    plot_important_params(problem, GSATree, MIP_overall, perturbation_dir, namesGSA)

    return ST, SDiff

def plot_important_params(problem, GSATree, MIP_overall, perturbation_dir, namesGSA):
    """
    Plot important parameters based on sensitivity analysis.

    Args:
        - problem: dict() -> The problem dictionary containing the analysis results.
        - GSATree: pytree() -> Instance of GSA_pytree
        - MIP_overall: set() -> Set of most important parameters across all outputs.
        - perturbation_dir: str -> The directory fo results with given perturbation.
        - namesGSA: list() -> A list containing parameter or group names.

    Returns:
        - 
    """
    
    # Calculate the sum of ST for all outputs for each parameter and sort in descending order  
    ST_sums = {name: 0 for name in namesGSA}
    for key in problem.analysis.keys():
        for i, name in enumerate(namesGSA):
            ST_sums[name] += problem.analysis[key]['ST'][i]
    
    sorted_params = sorted(ST_sums.items(), key=lambda x: x[1], reverse=True)
    
    # Assign colors based on whether the parameter is in the important parameters set
    colors = ['orange' if param[0] in MIP_overall else 'gray' for param in sorted_params]
    
    # Create the barplot
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10
    cm = 1/2.54

    fig_height = 9 * cm

    if len(namesGSA) > 8:
        fig_width = 19 * cm
    else:
        fig_width = 9 * cm

    plt.figure(figsize=(fig_width, fig_height))
    plt.bar(range(len(sorted_params)), [param[1] for param in sorted_params], color=colors)
    
    important_params_count = len(MIP_overall)
    plt.axvline(x=important_params_count - 0.5, color='black', linestyle='--', linewidth=1)
    
    plt.xticks(range(len(sorted_params)), [param[0] for param in sorted_params], rotation=45, \
                                                        ha='right', va='top', rotation_mode='anchor')
    plt.ylabel('Total order sensitivity index $S_T$')
    
    plt.tight_layout()

    file_name = os.path.join(perturbation_dir, 'Result_Important_Parameters_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId'])) 

    plt.savefig(file_name + '.pdf')
    plt.savefig(file_name + '.tiff')
    plt.savefig(file_name + '.jpg')

def plot_density_functions(problem, GSATree, perturbation_dir, sorted_outputs):

    """
    Generates a grid of density plots for each output parameter, based on the data stored in problem['results'].
    The names of the outputs are taken from problem['outputs'].

    Args:
    - problem: ProblemSpec() -> Dictionary containing problem specification of GSA using SALib.
    - GSATree: pytree() -> instance of GSA_pytree
    - perturbation_dir: str() -> path to the directory where the plots are stored.
    - categories: dict() -> Dictionary containing categories of outputs.

    Returns:
        Plots containing PDFs for each output.
    """

    Y = problem.results
    output_names = problem['outputs']

    # Calculate the grid size
    num_outputs = len(output_names)
    num_columns = 2 
    num_rows = ceil(num_outputs / num_columns)

    # Set plot formatting
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10

    cm = 1/2.54

    fig_width = 19 * cm
    fig_height = 23 * cm
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height), constrained_layout=True)

    # Make extra axes invisible if not needed
    if num_outputs % num_columns:
        for ax in axs.flat[num_outputs:]:
            ax.set_visible(False)

    # Flatten the axes array for easier indexing
    axs = axs.flatten()

    # Plot density for each output
    for i, (output_name, unit) in enumerate(sorted_outputs):
        # Locate the index of the output_name in the original results array
        original_index = output_names.index(output_name)
        sns.kdeplot(x=Y[:, original_index], ax=axs[i], fill=True)
        axs[i].set_xlabel(f"{output_name} in {unit}", fontsize=10)
        axs[i].set_ylabel('Density', fontsize=10)

    plt.tight_layout()

    file_name = os.path.join(perturbation_dir, 'Result_Density_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId']))    

    fig.savefig(file_name + '.pdf')
    fig.savefig(file_name + '.tiff')
    fig.savefig(file_name + '.jpg')
    
def write_sensitivities(problem, ST, SDiff, GSATree, perturbation_dir, namesGSA, sorted_outputs):
    
    """
    Writes sensitivity indices ST and SDiff to csv files.

    Args:
    - problem: ProblemSpec() -> Dictionary containing problem specification of GSA using SALib.
    - ST: array() -> An array of total order sensitivity indices.
    - SDiff: array() -> An array of total order sensitivity indices minus first order sensitivity indices.
    - GSATree: pytree() -> instance of GSA_pytree
    - perturbation_dir: str() -> path to the directory where the csv files are stored.
    - namesGSA: list() -> A list containing parameter or group names.
    - sorted_outputs: list() -> A list of outputs sorted by categories.

    Returns:
    - 
    """

    convergence_dir = os.path.join(perturbation_dir, '01_Convergence_Analysis')

    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)
    
    df_ST = pd.DataFrame(ST, columns=namesGSA, index=sorted_outputs)
    df_SDiff = pd.DataFrame(SDiff, columns=namesGSA, index=sorted_outputs)

    file_ST = os.path.join(convergence_dir, 'ST_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId']) + '.csv')
    file_SDiff = os.path.join(convergence_dir, 'SDiff_' + str(len(problem.samples)) + \
                '_perturb_' + str(GSATree.perturbation) + '%_patient_'+ str(GSATree.clinicalData['patientId']) + '.csv')

    df_ST.to_csv(file_ST)
    df_SDiff.to_csv(file_SDiff)

def read_param_file_GSA(filename, delimiter=' '):
    
    """
    Read parameter file for Global Sensitivity Analysis (GSA).

    Parameters:
    - filename: str()) -> The path to the parameter file.
    - delimiter: str(), optional -> The delimiter used in the parameter file. Default is a space (' ').

    Returns:
    - names: list() -> A list of parameter names.
    - groups: list() or None -> A list of group names corresponding to each parameter. If all parameters belong to the same group, groups will be set to None.

    Raises:
    - ValueError: If the groups are not defined correctly in the file or if only one group is defined.

    """

    names = []
    groups = []
    
    with open(filename, "r") as csvfile:

        sample = csvfile.read(1024)

        if all(len(line.split()) > 1 for line in sample.split('\n') if line.strip() and not line.strip().startswith('#')):
            has_group = True
            fieldnames = ["name", "group"]
        elif any(len(line.split()) > 1 for line in sample.split('\n') if line.strip() and not line.strip().startswith('#')):
            raise ValueError("Sensitivity Analysis: Groups not defined correctly in file. Check input file again!")
        else:
            has_group = False
            fieldnames = ["name"]

        csvfile.seek(0)
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter)

        for row in reader:
            if row["name"].strip().startswith("#"):
                pass
            else:
                names.append(row["name"])

                # If the fourth column does not contain a group name, use
                # the parameter name
                if has_group is True:
                    if row["group"] is not None:
                        groups.append(row["group"])
                    elif row["group"] == "NA":
                        groups.append(row["name"])
                else:
                    groups.append(row["name"])

    if groups == names:
        groups = None
    elif len(set(groups)) == 1:
        raise ValueError(
            """Only one group defined, results will not be
            meaningful"""
        )

    return names, groups