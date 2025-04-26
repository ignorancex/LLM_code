"""
The "model.py" define functions which create and solve Pyomo models
"""

__author__ = "Babu Kumaran Nalini, Zhengjie You"
__copyright__ = "2021 TUM-EWK"
__credits__ = []
__license__ = "GPL v3.0"
__version__ = "1.0"
__maintainer__ = "Babu Kumaran Nalini"
__email__ = "babu.kumaran-nalini@tum.de"
__status__ = "Development"

import pyomo.core as pyen
from pyomo.opt import SolverFactory
from pyomo.environ import value as get_value
from pyomo.environ import *
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import time as tm
from datetime import datetime


def create_model(ems_local):
    """ create one optimization instance and parameterize it with the input data in ems model
    Args:
        - ems_local:  ems model which has been parameterized

    Return:
        - m: optimization model instance created according to ems model
    """
    # record the time
    t0 = tm.time()
    devices = ems_local['devices']

    # read data from excel file
    t = tm.time()
    time_interval = ems_local['time_data']['t_inval']  # x minutes for one time step
    
    # write in the time series from the data
    df_time_series = ems_local['fcst']
    time_series = pd.DataFrame.from_dict(df_time_series)

    # system
    # get the initial time step
    time_step_initial = ems_local['time_data']['isteps']
    time_step_end = ems_local['time_data']['nsteps']
    timesteps = np.arange(time_step_initial, time_step_end)

    # 15 min for every timestep/ timestep by one hour
    # create the concrete model
    p2e = time_interval / 60

    # create the model object m
    m = pyen.ConcreteModel()

    # create the parameter
    # print('Define Model ...\n')

    m.t = pyen.Set(ordered=True, initialize=timesteps)
    
    # battery
    bat_param = devices['bat']
    m.bat_cont_max = pyen.Param(initialize=bat_param['stocap'])
    m.bat_SOC_init = pyen.Param(initialize=bat_param['initSOC'])
    m.bat_power_max = pyen.Param(initialize=bat_param['maxpow'])
    m.bat_eta = pyen.Param(initialize=bat_param['eta'])

    # solar
    pv_param = devices['pv']
    m.pv_peak_power = pyen.Param(initialize=pv_param['maxpow'])
    m.solar = pyen.Param(m.t, initialize=1, mutable=True)

    # price
    m.ele_price_in, m.ele_price_out = (pyen.Param(m.t, initialize=1, mutable=True) for i in range(2))

    # lastprofil
    m.lastprofil_elec = pyen.Param(m.t, initialize=1, mutable=True)

    for t in m.t:
        m.ele_price_in[t] = time_series.loc[t]['ele_price_in']
        m.ele_price_out[t] = time_series.loc[t]['ele_price_out']
        m.lastprofil_elec[t] = time_series.loc[t]['load_elec']
        m.solar[t] = time_series.loc[t]['solar_power']

    # Variables
    m.PV_cap, m.elec_import, m.elec_export, m.bat_cont, m.sto_e_cont, m.bat_pow_pos, m.bat_pow_neg, \
     = (pyen.Var(m.t, within=pyen.NonNegativeReals) for i in range(7))
    m.sto_e_pow, m.costs = (pyen.Var(m.t, within=pyen.Reals) for i in range(2))

    # Constrains
    # battery
    def battery_e_cont_def_rule(m, t):
        if t > m.t[1]:
            return m.bat_cont[t] == m.bat_cont[t - 1] + (
                    m.bat_pow_pos[t] * m.bat_eta - m.bat_pow_neg[t] / m.bat_eta) * p2e
        else:
            return m.bat_cont[t] == m.bat_cont_max * m.bat_SOC_init / 100 + (m.bat_pow_pos[t] * m.bat_eta -
                                                                             m.bat_pow_neg[t] / m.bat_eta) * p2e

    m.bat_e_cont_def = pyen.Constraint(m.t,
                                       rule=battery_e_cont_def_rule,
                                       doc='battery_balance')

    def elec_balance_rule(m, t):
        return m.elec_import[t] + m.PV_cap[t] * m.solar[t] - \
               m.elec_export[t] - m.lastprofil_elec[t] - \
               (m.bat_pow_pos[t] - m.bat_pow_neg[t]) == 0

    m.elec_power_balance = pyen.Constraint(m.t, rule=elec_balance_rule, doc='elec_balance')

    def cost_sum_rule(m, t):
        return m.costs[t] == p2e * (m.elec_import[t] * m.ele_price_in[t]
                                    - m.elec_export[t] * m.ele_price_out[t])

    m.cost_sum = pyen.Constraint(m.t,
                                 rule=cost_sum_rule)

    # PV
    def pv_max_cap_rule(m, t):
        return m.PV_cap[t] <= m.pv_peak_power

    m.pv_max_cap_def = pyen.Constraint(m.t,
                                       rule=pv_max_cap_rule)

    # elec_import
    def elec_import_rule(m, t):
        return m.elec_import[t] <= 50 * 5000

    m.elec_import_def = pyen.Constraint(m.t,
                                        rule=elec_import_rule)

    # elec_export
    def elec_export_rule(m, t):
        return m.elec_export[t] <= 50 * 5000

    m.elec_export_def = pyen.Constraint(m.t,
                                        rule=elec_export_rule)

    if m.bat_cont_max > 0:
        def bat_e_cont_min_rule(m, t):
            return m.bat_cont[t] / m.bat_cont_max >= 0.1

        m.bat_e_cont_min = pyen.Constraint(m.t,
                                           rule=bat_e_cont_min_rule)

        def bat_e_cont_max_rule(m, t):
            return m.bat_cont[t] / m.bat_cont_max <= 0.9

        m.bat_e_cont_max = pyen.Constraint(m.t, rule=bat_e_cont_max_rule)

    def bat_e_max_pow_rule_1(m, t):
        return m.bat_pow_pos[t] <= min(m.bat_power_max, m.bat_cont_max)

    m.bat_e_pow_max_1 = pyen.Constraint(m.t,
                                        rule=bat_e_max_pow_rule_1)

    def bat_e_max_pow_rule_2(m, t):
        return m.bat_pow_neg[t] <= min(m.bat_power_max, m.bat_cont_max)

    m.bat_e_pow_max_2 = pyen.Constraint(m.t,
                                        rule=bat_e_max_pow_rule_2)

    # end state of storage and battery
    m.bat_e_cont_end = pyen.Constraint(expr=(m.bat_cont[m.t[-1]] >= 0.5 * m.bat_cont_max))

    def obj_rule(m):
        # Return sum of total costs over all cost types.
        # Simply calculates the sum of m.costs over all m.cost_types.
        return pyen.summation(m.costs)

    m.obj = pyen.Objective(
        sense=pyen.minimize,
        rule=obj_rule,
        doc='Sum costs by cost type')

    return m


def solve_model(m, solver, time_limit=100000, min_gap=0.001, troubleshooting=True):
    """ solve the optimization problem and save the results in instance m
    Args:
        - m: optimization model instance
        - solver: solver to be used, e.g. "glpk", "gurobi", "cplex"...
        - time_limit: time limit (in seconds) terminating the optimization
        - min_gap: solver will terminate (with an optimal result) when the gap between the lower and upper objective
          bound is less than min_gap times the absolute value of the upper bound.

    """
    optimizer = SolverFactory(solver)

    # optimizer.solve(m, load_solutions=True, options=solver_opt, tee=True)
    if solver == "glpk":
        solver_opt = dict()
        solver_opt['mipgap'] = min_gap
        optimizer.solve(m, load_solutions=True, options=solver_opt, tee=troubleshooting, timelimit=time_limit)
    elif solver == "gurobi":
        optimizer.set_options("timelimit="+str(time_limit))  # seconds
        optimizer.set_options("mipgap="+str(min_gap))  # default = 1e-3
        optimizer.solve(m, load_solutions=True, tee=troubleshooting)
    else:
        try:
            optimizer.solve(m, load_solutions=True, tee=troubleshooting)
        except RuntimeError:
            raise RuntimeError(
                'this solver is not available or the configuration is not adequate')
    return m


def extract_res(m, ems):
    """ extract the results from instance m and save it into ems model
    Args:
        - m: optimization model instance with results
        - ems: ems model to be filled with optimization results

    """

    timesteps = np.arange(ems['time_data']['isteps'], ems['time_data']['nsteps'])
    length = len(timesteps)

    # electricity variable
    elec_import, elec_export, lastprofil_elec, pv_power, bat_cont, \
    bat_power, pv_pv2demand, pv_pv2grid, bat_grid2bat, bat_power_pos, bat_power_neg \
    = (np.zeros(length) for i in range(11))

    bat_max_cont = get_value(m.bat_cont_max)

    i = 0

    # timesteps = sorted(get_entity(prob, 't').index)
    # demand, ext, pro, sto = get_timeseries(prob, timesteps

    for idx in timesteps:
        # electricity balance
        elec_import[i] = get_value(m.elec_import[idx])
        elec_export[i] = get_value(m.elec_export[idx])
        lastprofil_elec[i] = get_value(m.lastprofil_elec[idx])
        pv_power[i] = get_value(m.PV_cap[idx] * m.solar[idx])

        bat_cont[i] = get_value(m.bat_cont[idx])
        bat_power_pos[i] = get_value(m.bat_pow_neg[idx])
        bat_power_neg[i] = -get_value(m.bat_pow_pos[idx])
        pv_pv2demand[i] = min(pv_power[i], lastprofil_elec[i])
        pv_pv2grid[i] = max(0, min(pv_power[i] - pv_pv2demand[i] + bat_power_neg[i], elec_export[i]))
        bat_grid2bat[i] = min(elec_import[i], -bat_power_neg[i])
        SOC_elec = bat_cont / bat_max_cont * 100 if bat_max_cont > 0 else 0 * bat_cont
        i += 1
        
    data_input = {
                  'SOC_elec': list(SOC_elec),
                  'PV_power': list(pv_power), 'pv_pv2demand': list(pv_pv2demand), 'pv_pv2grid': list(pv_pv2grid),
                  'grid_import': list(elec_import),
                  'Last_elec': list(lastprofil_elec), 'grid_export': list(elec_export),
                  'bat_grid2bat': list(bat_grid2bat),
                  'bat_input_power': list(-bat_power_neg), 'bat_output_power': list(bat_power_pos),
                  'bat_SOC': list(SOC_elec)}

    ems['optplan'] = data_input

    return ems
