"""

This file manages the optimization of the model.

"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.io as pio
import jax.numpy as jnp
import jax
import tree_math as tm
import tree_math.numpy as tnp
import os
from jax import jit, vmap
import estimagic as em
from jaxopt import ScipyBoundedMinimize, ScipyBoundedLeastSquares, ProjectedGradient, LBFGSB
from jaxopt.projection import projection_box
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import equinox as eqx
import file_manager

import model_main

jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={int(os.environ["SLURM_NTASKS"]) if os.environ.get("SLURM_JOB_ID") else int(os.cpu_count())}'

pio.kaleido.scope.mathjax = None

def main(clinicalData, x0, bounds, x0Batch, initTree, simPostTree, simCompTree, exploreParameterSpace):

    """ Distributes the calculation through local_devices by sharding initial values.
    Launches the optimization through an optimizer and post process the results.

    Args:
        - clinicalData: clinicalData of the current patient
        - x0: dict() -> initial value for batching job
        - bounds: tuple -> bounds of the parameters
        - x0Batch: dict() -> set of modified x0 values for the fitting (note: old version of multistart - not used anymore!)
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main
        - exploreParameterSpace: bool -> if True, the parameter space will be explored

    Returns:
        res_optimization.
    """
    
    results = estimagic_optimization(x0, bounds, clinicalData, initTree, simPostTree, simCompTree, exploreParameterSpace)

def estimagic_optimization(x, bounds, clinicalData, initTree, simPostTree, simCompTree, exploreParameterSpace):
    
    #algorithms = ["scipy_lbfgsb", "scipy_slsqp", "fides", "scipy_ls_trf"]
    algorithms = ["scipy_ls_trf"]

    algo_options = {
        "stopping.max_criterion_evaluations": 100,
        }
    
    ms_options = {
        "convergence.max_discoveries": 10,
        "n_samples": 1000 * len(x), 
        "share_optimizations": 0.01,
        "n_cores": int(os.environ["XLA_FLAGS"].split("=")[1]),
        "convergence.relative_params_tolerance": 1e-6
        }
    
    scaling_options={"method": "bounds", "clipping_value": 0.0}
    
    if exploreParameterSpace: slice_plot(x, bounds, clinicalData, initTree, simPostTree, simCompTree)

    res_optimization = em_optimize(x, bounds, clinicalData, initTree, simPostTree, simCompTree, \
                                   algorithms, algo_options, ms_options, scaling_options)

    criterion_plot(res_optimization, clinicalData)

    best_optimizer = min(res_optimization, key=lambda opt: res_optimization[opt].criterion)
    print(res_optimization[best_optimizer])
    post_processing(res_optimization[best_optimizer].params, res_optimization[best_optimizer].criterion, \
                    initTree, clinicalData, simPostTree, simCompTree)
    
    return res_optimization

def em_optimize(x, bounds, clinicalData, initTree, simPostTree, simCompTree, algorithms, algo_options, \
                ms_options, scaling_options):

    res_opt = {}
    kwargs = {'clinicalData': clinicalData, 'initTree': initTree, 'simPostTree': simPostTree,
                                  'simCompTree': simCompTree}

    if 'scipy_ls_trf' in algorithms: 
        criterion_lss = jax.jit(obj_func_lss)
        jacobian = jax.jit(jax.jacrev(obj_func_lss))

    if any(algo in ['scipy_lbfgsb', 'scipy_slsqp', 'fides'] for algo in algorithms):
        criterion = jax.jit(obj_func)
        derivative = jax.jit(jax.grad(obj_func))

    for algo in algorithms:
        if algo == "scipy_ls_trf":
            res_opt[algo] = em.minimize(
                criterion=criterion_lss,
                params=x,
                derivative=jacobian,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                criterion_kwargs=kwargs,
                derivative_kwargs=kwargs,
                algo_options=algo_options,
                algorithm=algo,
                multistart=True,
                multistart_options=ms_options,
                scaling=False,
                scaling_options=scaling_options)
        else:
            res_opt[algo] = em.minimize(
                criterion=criterion,
                params=x,
                derivative=derivative,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                criterion_kwargs=kwargs,
                derivative_kwargs=kwargs,
                algo_options=algo_options,
                algorithm=algo,
                multistart=True,
                multistart_options=ms_options,
                scaling=False,
                scaling_options=scaling_options)
    
    return res_opt

@jit
def obj_func(x, clinicalData, initTree, simPostTree, simCompTree):

    """ Objective function to minimize when using "normal" algorithms. 

    Args:
        - x: dict() -> model parameters for the current iteraction
        - clinicalData: clinicalData of the current patient
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        float -> error for the current iteration
    """
    
    initTree = initTree.update(x)
    simulatedData = model_main.modeling(initTree, simPostTree, simCompTree)[0]

    error = norm(simulatedData.results, clinicalData)

    return error

@jit
def obj_func_lss(x, clinicalData, initTree, simPostTree, simCompTree):

    """ Objective function to minimize when using nonlinear least-squares algorithms. 

    Args:
        - x: dict() -> model parameters for the current iteraction
        - clinicalData: clinicalData of the current patient
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        float -> error for the current iteration
    """
    
    initTree = initTree.update(x)
    simulatedData = model_main.modeling(initTree, simPostTree, simCompTree)[0]

    error = norm_lss(simulatedData.results, clinicalData)

    return error

def norm(x, y):

    """ Calculates weighted sum of squared errors of simulation results. 

    Args:
        - x: dict() -> simulation results
        - y: dict() -> clinical data

    Returns:
        float -> Weighted sum of squared errors of current iteration step of optimization.
    """

    IntersectionKeys = x.keys() & y.keys()
    xIntersection = {k: x[k] for k in IntersectionKeys}
    yIntersection = {k: y[k] for k in IntersectionKeys}

    a = tm.Vector(xIntersection)
    b = tm.Vector(yIntersection)

    fvec = (a - b) / b
    
    return tnp.dot(fvec, fvec)

def norm_lss(x, y):

    """ Calculates vector of residuals of squared errors of simulation results. 

    Args:
        - x: dict() -> simulation results
        - y: dict() -> clinical data

    Returns:
        float -> Vector of squared errors (residuals) of current iteration step of optimization.
    """

    IntersectionKeys = x.keys() & y.keys()
    xIntersection = {k: x[k] for k in IntersectionKeys}
    yIntersection = {k: y[k] for k in IntersectionKeys}

    a = tm.Vector(xIntersection)
    b = tm.Vector(yIntersection)

    # root_contributions: An array containing the root (weighted) contributions.
    fvec = (a - b) / b

    return {"root_contributions": fvec.tree}

def post_processing(solution_opt, error, initTree, clinicalData, simPostTree, simCompTree):

    """ Post process the optimization results.

    Args:
        - solution_opt: dict() -> fitted parameters 
        - error: [array] -> errors of the fitting
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        -
    """    
    # Update parameters with results from optimization and run model
    initTree = initTree.update(solution_opt)
    simulatedData = model_main.modeling(initTree, simPostTree, simCompTree)[0]

    # Find intersection between simulated and clinical data
    predicted = {k: simulatedData.results[k] for k in simulatedData.results if k in clinicalData}
    actual = {k: clinicalData[k] for k in clinicalData if k in simulatedData.results}
    patientId = initTree.clinicalData['patientId']

    bar_plot(predicted, actual, patientId)
    write_results(solution_opt, error, predicted, actual, patientId)

def bar_plot(simulatedData, clinicalData, patientId):

    """ Plots the optimization results.

    Args:
        - simulatedData: dict() -> subset of simulated data 
        - clinicalData: dict() -> subset of clinical data
        - patientId: int -> patient id

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

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 10
    cm = 1/2.54

    fig_height_base = 7 * cm
    fig_width = 14 * cm

    # Determine the number of subplots needed
    num_plots = sum(any(key in clinicalData for key in cat['keys']) for cat in categories.values())
    max_keys = max(len(cat['keys']) for cat in categories.values())

    fig, axs = plt.subplots(num_plots, 1, figsize=(fig_width, fig_height_base * num_plots), squeeze=False)

    bar_width = 0.3 / 4
    offset = (1 - max_keys * (bar_width + 0.1)) / 2

    current_plot = 0
    for category, info in categories.items():
        category_keys = [key for key in info['keys'] if key in clinicalData]

        if category_keys:
            num_bars = len(category_keys)
            x_pos = np.linspace(offset, offset + max_keys * (bar_width + 0.1) - bar_width, num_bars)
            
            simulated_vals = [simulatedData[key] for key in category_keys]
            clinical_vals = [clinicalData[key] for key in category_keys]

            axs[current_plot, 0].bar(x_pos - bar_width, simulated_vals, width=bar_width, label='Simulated', align='center')
            axs[current_plot, 0].bar(x_pos + bar_width, clinical_vals, width=bar_width, label='Clinical', align='center')

            axs[current_plot, 0].set_xticks(x_pos)
            axs[current_plot, 0].set_xticklabels(category_keys)
            axs[current_plot, 0].set_ylabel(f"{category} in {info['unit']}")
            current_plot += 1

    # Adjust legend
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig.legend(['Simulated', 'Clinical'], fontsize=11, bbox_to_anchor=(.77, .97), loc="upper left")

    base_dir = file_manager.optimizationFolder + 'Results/Fitting_patient_'+ str(patientId)
    
    plt.savefig(base_dir + '.pdf')
    plt.savefig(base_dir + '.tiff')
    plt.savefig(base_dir + '.svg')
    
def slice_plot(x, bounds, clinicalData, initTree, simPostTree, simCompTree):

    """ Plots the exploration of the parameter space.

    Args:
        - x: dict() -> initial values for the optimization
        - bounds: tuple -> bounds of the parameters
        - clinicalData: dict() -> clinical data
        - initTree: pytree() -> initTree instance from the model main
        - simPostTree: pytree() -> simPostTree instance from the model main
        - simCompTree: pytree() -> simCompTree instance from the model main

    Returns:
        -
    """

    font_settings = {
        'family': "STIXGeneral",
        'size': 10
    }

    dpi = 96
    fig_width_px = 19 / 2.54 * dpi
    fig_height_px = 23 / 2.54 * dpi

    figSlice = em.slice_plot(
        func=obj_func,
        params=x,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        func_kwargs={'clinicalData': clinicalData, 'initTree': initTree, 'simPostTree': simPostTree, 'simCompTree': simCompTree},
        n_gridpoints=200,
        plots_per_row=3,
        share_y=False,
        make_subplot_kwargs={'horizontal_spacing': 0.1, 'vertical_spacing': 0.08}
    )

    figSlice.update_layout(
        width = fig_width_px,
        height = fig_height_px,
        font = font_settings,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    figSlice.update_yaxes(title_standoff=5)
    figSlice.update_xaxes(title_standoff=10)

    figSlice.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
                            tickcolor='black', tickwidth=1, ticklen=5)
    figSlice.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
                            tickcolor='black', tickwidth=1, ticklen=5)

    figSlice.show()

    file_path = file_manager.optimizationFolder+'Results/Exploration_Parameter_Space'
    figSlice.write_image(file_path + ".pdf", width=fig_width_px, height=fig_height_px)
    figSlice.write_image(file_path + ".svg", width=fig_width_px, height=fig_height_px)
    figSlice.write_image(file_path + ".png", width=fig_width_px, height=fig_height_px)

def criterion_plot(result, clinicalData):

    """ Plots the objective function values for all starting values.

    Args:
        - result: dict() -> results of the optimization
        - clinicalData: dict() -> clinical data

    Returns:
        -
    """

    font_settings = {
        'family': "STIXGeneral",
        'size': 10
    }

    dpi = 96
    fig_width_px = 19 / 2.54 * dpi
    fig_height_px = 12 / 2.54 * dpi

    figCriterion = em.criterion_plot(
        result,
        monotone=True
    )

    figCriterion.update_layout(
        width = fig_width_px,
        height = fig_height_px,
        font = font_settings,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    figCriterion.update_yaxes(title_standoff=5)
    figCriterion.update_xaxes(title_standoff=10)

    figCriterion.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
                            tickcolor='black', tickwidth=1, ticklen=5)
    figCriterion.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,
                            tickcolor='black', tickwidth=1, ticklen=5)

    figCriterion.show()

    file_path = file_manager.optimizationFolder+'Results/Criterion_plot_patient_' + str(clinicalData.get('patientId', 'unknown'))
    figCriterion.write_image(file_path + ".pdf", width=fig_width_px, height=fig_height_px)
    figCriterion.write_image(file_path + ".svg", width=fig_width_px, height=fig_height_px)
    figCriterion.write_image(file_path + ".png", width=fig_width_px, height=fig_height_px)

def write_results(solution_opt, error, simulatedData, clinicalData, patientId):
    
    """ Writes the optimization results to a file.

    Args:
        - solution_opt: dict() -> fitted parameters 
        - error: [array] -> errors of the fitting
        - simulatedData: dict() -> subset of simulated data 
        - clinicalData: dict() -> subset of clinical data
        - patientId: int -> patient id

    Returns:
        -
    """
    file_path = file_manager.optimizationFolder + f'Results/result_optimization_patient_{patientId}.txt'

    with open(file_path, 'w') as file:
        file.write('###########################################\n')
        file.write('#### Results of Parameter Optimization ####\n')
        file.write('###########################################\n')
        file.write('#\n')
        file.write(f'Error {error}\n')
        file.write('#\n')
        file.write('########## Outputs ###########\n')
        file.write('# Parameter Simulated Measured\n')
        for outkey in simulatedData.keys():
            file.write(f'{outkey} {simulatedData[outkey]} {clinicalData[outkey]}\n')
        file.write('#\n')
        file.write('######### Parameters #########\n')
        file.write('# Parameter Value')
        for param, value in solution_opt.items():
            file.write(f'\n{param} {value}')