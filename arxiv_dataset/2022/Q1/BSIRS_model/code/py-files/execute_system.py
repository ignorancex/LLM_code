"""
Created on Tue Feb  16 13:47:48 2021

@author: MichalGnacik
"""
import numpy as np
from tqdm import tqdm
from gradient_desc import momentum
from score_fun import mean_squared, regular_l2, regular_nearest
import bsirs_model as model


def initiate_system_single(
    model_params,
    c_params,
    change=False,
    system_change=None,
    loops=10,
    beta=1.0
):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    W = model_params['W']
    pops = model_params['pops']
    sir_params = model_params["SIR parameters"]
    tau = model_params["tau"]
    initial_infected = model_params["initial_infected"]
    length = model_params["length"]
    system = model.BSIRS(W, pops, tau, initial_infected)
    if change:
        system.change_infected(*system_change)
    epsilon, p_rec, p_xi = sir_params
    u_vals = np.zeros(shape=(loops+1, length))
    s_vals = np.zeros(shape=(loops+1, length))
    i_vals = np.zeros(shape=(loops+1, length))
    s_vals[0] = system.susceptibles
    i_vals[0] = system.infected
    u_vals[0] = system.u_value
    for k in range(1, loops+1):
        c_vals = c_params  # + parameters[1][i]* I_hist[k-1] / N
        beta = beta
        system.update(beta, c_vals, epsilon, p_rec, p_xi)
        u_vals[k] = system.u_value
        s_vals[k] = system.susceptibles
        i_vals[k] = system.infected
    return s_vals, i_vals, u_vals


def initiate_system_all(
    model_params,
    c_params,
    loops=10,
    beta=1.0
):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    W = model_params['W']
    pops = model_params['pops']
    sir_params = model_params["SIR parameters"]
    tau = model_params["tau"]
    initial_infected = model_params["initial_infected"]
    length = model_params["length"]
    system = model.BSIRS(W, pops, tau, initial_infected)
    epsilon, p_rec, p_xi = sir_params
    c_size = c_params.shape[0]
    u_vals = np.zeros(shape=(c_size*loops+1, length))
    i_vals = np.zeros(shape=(c_size*loops+1, length))
    i_vals[0] = system.infected
    u_vals[0] = system.u_value
    l_val = 0
    for i in range(c_size):
        for k in range(1, loops + 1):
            c_val = c_params[i]  # + parameters[1][i]* I_hist[k-1] / N
            beta = beta
            system.update(beta, c_val, epsilon, p_rec, p_xi)
            u_vals[l_val+1] = system.u_value
            i_vals[l_val+1] = system.infected
            l_val += 1
    return i_vals, u_vals


def calibrate(
    data_to_fit,
    result_params,
    model_params,
    optimizer_params,
    alpha,
    score_type="regular neighbours",
    num_trials=1000,
):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    W = model_params['W']
    pops = model_params['pops']
    initial_infected = np.copy(model_params["initial_infected"])
    length = model_params["length"]
    c_array = result_params["c"]
    beta_val = result_params["beta"]
    score_array = result_params["score"]
    infected_array = result_params["infected"]
    suct_array = result_params["susceptibles"]
    u_array = result_params["u values"]
    time_slots = result_params["time slots"]
    optimizer = optimizer_params["optimizer"] = "momentum"
    mu = optimizer_params["mu"]
    learning_rate = optimizer_params["learning rate"]
    for index in range(1, time_slots):
        vt_val = 0.0
        data_to_fit_single = data_to_fit[index]
        if index == 1:
            initial_infected = initial_infected
            initial_sus = pops - initial_infected
            initial_u = np.full(length, 0.5)
        else:
            initial_infected = infected_array[index-2]
            initial_sus = suct_array[index-2]
            initial_u = u_array[index-2]
        opt_cvals = np.copy(c_array[index-1])
        score_old = score_array[index-1]
        for k in tqdm(range(num_trials), desc=f"index{index}"):
            system_change = [initial_infected,  initial_sus, initial_u]
            new_infected = initiate_system_single(
                model_params,
                opt_cvals,
                change=True,
                system_change=system_change,
                beta=beta_val
                )[1]
            data_to_fit_copy = np.copy(data_to_fit_single)  # + noise
            grad = new_infected[-1] - data_to_fit_copy
            if optimizer == optimizer:
                difference, vt_val = momentum(
                    grad,
                    vt_val,
                    mu,
                    learning_rate
                )
            else:
                difference = learning_rate*grad
            opt_cvals = np.copy(opt_cvals) - difference

            if score_type == "regular neighbours":
                score = regular_nearest(
                    new_infected[-1],
                    data_to_fit_copy,
                    opt_cvals,
                    beta_val,
                    W,
                    alpha)
            elif score_type == "regularL2":
                score = regular_l2(
                    new_infected[-1],
                    data_to_fit_copy,
                    opt_cvals)
            else:
                score = mean_squared(new_infected[-1], data_to_fit_copy)
            if k == num_trials-1:
                print(f"final score is {score_old}")
            if score < score_old:
                score_old = score
                c_array[index-1] = np.copy(opt_cvals)
                score_array[index-1] = score
                system_change = [initial_infected,  initial_sus, initial_u]
                susceptibles, infected, u_vals = initiate_system_single(
                    model_params,
                    opt_cvals,
                    change=True,
                    system_change=system_change,
                    beta=beta_val
                )
                infected_array[index-1] = infected[-1]
                suct_array[index-1] = susceptibles[-1]
                u_array[index-1] = u_vals[-1]
    return result_params
