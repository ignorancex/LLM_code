import cvxpy as cp
import numpy as np
import pandas as pd

def setup_optimization_problem_for_ap1000(gen_df, loads, gen_variable_long, deadtime_matrix_ap1000, deadtime_matrix_ap300, D_points, first_run, remaining_downtime, commit_at_len_T, PHS_duration, G_nuclear_critical):

    """
    Sets up the unit commitment optimization problem for AP1000 reactors and other generators.
    
    Parameters:
        - gen_df (DataFrame): Generator information, including generator parameters and costs.
        - loads (DataFrame or Series): Load profile over the optimization horizon.
        - gen_variable_long (DataFrame): Time series of variable generator capacity factor (wind/solar) availability.
        - deadtime_matrix_ap1000 (DataFrame): Deadtime lookup table for AP1000 reactors
        - deadtime_matrix_ap300 (DataFrame): Deadtime lookup table for AP300 reactors
        - D_points (DataFrame): Pre-collated deadtime values by generator (indexed by r_id).
        - first_run (bool): If True, the initial SOC at t=0 is left unconstrained (only for the first block).
        - remaining_downtime (DataFrame): Hours of downtime still required from the previous run, carried forward 
          if not yet fulfilled.
        - commit_at_len_T (DataFrame): Commitment state, generator r_id, SOC value, and termination_status to transfer between runs.
        - PHS_duration: Duration (in hours) for storage energy capacity.
        - G_nuclear_critical (list): List of reactor IDs that have reached critical k_eff.
    
    Returns:
        dict containing:
            - prob (cvxpy.Problem): The optimization problem formulation. 
            - UCcost (Expression): Objective function (total system cost).
            - GEN (Variable): Generation dispatch by generator and time.
            - Rd, Up, St (Variable): Binary variables for ramp-down, ramp-up, and stabilization states.
            - NSE (Variable): Non-served energy per hour.
            - COMMIT, SHUT, START (Variable): Binary unit commitment, shutdown, and startup decisions.
            - gen_var_cf (DataFrame): Capacity factors for variable generators.
            - dynamic_downtime (Variable): Tracks downtime hours for each generator.
            - CHARGE, DISCHARGE, SOC (Variable): Pumped hydro charge, discharge, and state-of-charge.
    """

    
    """A. DEFINE GENERATOR SETS """
    G = gen_df["r_id"].tolist()                                 #set of all generators (above are all subsets of this)
    G_thermal = gen_df[gen_df["up_time"] > 0]["r_id"].tolist()  #thermal generators
    G_nonthermal = gen_df[gen_df["up_time"] == 0]["r_id"].tolist() #non-thermal generators
    G_var = gen_df[gen_df["is_variable"] == 1]["r_id"].tolist()    #variable/renewable generators
    G_phs = gen_df[gen_df["resource"] == "hydroelectric_pumped_storage" ]["r_id"].tolist()  #PHS ID
    G_wind = gen_df[gen_df["resource"] == "onshore_wind_turbine" ]["r_id"].tolist()         #Wind ID
    G_solar = gen_df[gen_df["resource"] == "solar_photovoltaic" ]["r_id"].tolist()          #Solar ID
    G_nonphs = list(set(G)-set(G_phs))  #Not PHS generators
    G_nt_nonvar = G_nonthermal + G_phs #non-variable and non-thermal
    G_ap1000 = gen_df[gen_df["resource"] == "ap1000" ]["r_id"].tolist()                     #AP1000 IDs
    
    # reactor ID that have not reached a threshold keff
    G_nuclear_noncritical=[item for item in G_thermal if item not in G_nuclear_critical]
    
    # length of a single UC
    T = loads["hour"]
    T_red = T[:-1]      #T-1 for specifc constraints
    
    # Generator capacity factor time series for variable generators
    gen_var_cf = gen_variable_long.merge(
        gen_df.loc[gen_df["is_variable"] == 1, ["r_id", "gen_full", "existing_cap_mw"]], 
        on="gen_full", how="inner")
    
    
    """B. DEFINE VARIABLES """
    # B.1. Core variables
    #B.1.1 Bin variables COMMIT, START and SHUT
    COMMIT = cp.Variable((len(G), len(T)), boolean=True) # commitment status (Bin=binary) for thermals
    START = cp.Variable((len(G), len(T)), boolean=True) # startup decision for thermals
    SHUT = cp.Variable((len(G), len(T)), boolean=True) # shutdown decision for thermal   

    #B.1.2. Bin variables for PMINSTABLE implement from Jenkins et al. nuclear flexibility paper
    Rd = cp.Variable((len(G), len(T)), boolean=True)  # Ramp down indicators
    St = cp.Variable((len(G), len(T)), boolean=True)  # Stabilization indicators
    Up = cp.Variable((len(G), len(T)), boolean=True)  # Ramp up indicators

    #B.1.3. Generation, auxillary generation and NSE variables
    GEN = cp.Variable((len(G), len(T)))         #generation across the fleet
    GENAUX=cp.Variable((len(G), len(T)))        #Auxillary variable for introducing ramp constraints
    NSE=cp.Variable(len(T))                     #NSE hourly
    curtailment = cp.Variable((len(G), len(T))) #hourly curtailemnt
    
    #B.1.4. For deadtime constraints
    dynamic_downtime = cp.Variable((len(G), len(T)), integer=True) #for downtime assignment
    
    # B.2. PHS variables for battery implement 
    SOC = cp.Variable(len(T)) #State-of-charge
    CHARGE = cp.Variable(len(T))
    DISCHARGE = cp.Variable(len(T))
    CHARGE_MODE = cp.Variable(len(T), boolean=True) #Bin variable records if battery is charging 1 or not 0

    #Required constants
    CapSolar=gen_df.loc[gen_df['resource'] == 'solar_photovoltaic', 'existing_cap_mw'].iloc[0] #Solar technical cap
    CapWind=gen_df.loc[gen_df['resource'] == 'onshore_wind_turbine', 'existing_cap_mw'].iloc[0] #Wind technical cap

    #Big-M Constant (bigger than the cap of largest generator)
    M=gen_df.loc[gen_df["resource"] == "ap1000", "existing_cap_mw"].iloc[0]+10 

    #All assumed time constants value
    PMINSTABLE = 10
    up_time = 6
    min_dn_time = 10

    #sets for vectorizing constraint sets
    thermal_idx = np.array(G_thermal, dtype=int) - 1
    nonthermal_idx = np.array(G_nonthermal, dtype=int) - 1
    solar_idx = np.array(G_solar, dtype=int) - 1
    wind_idx = np.array(G_wind, dtype=int) - 1
    
    # Filter once for thermal generator rows
    thermal_df = gen_df[gen_df["r_id"].isin(G_thermal)].set_index("r_id").loc[G_thermal]
    
    # Extract arrays
    existing_cap_array = thermal_df["existing_cap_mw"].values
    min_power_array = thermal_df["min_power"].values
    ramp_up_array = thermal_df["ramp_up_percentage"].values
    ramp_dn_array = thermal_df["ramp_dn_percentage"].values
    
    # Compute power limits and ramp rates (with shape (n_thermal, 1) for broadcasting)
    pmin_array = (existing_cap_array * min_power_array)[:, None]
    cap_rup = (existing_cap_array * ramp_up_array)[:, None]
    cap_rdn = (existing_cap_array * ramp_dn_array)[:, None]

    #var ids
    var_idx = np.array(G_var, dtype=int) - 1
    cap_var_array = gen_df.set_index("r_id").loc[G_var, "existing_cap_mw"].values[:, None]

    
    """C. DEFINE OBJECTIVE EXPRESSION"""
    # C.1 Sum of variable costs (G_nonvar and G_var: all) + start-up costs (G_thermal) for all generators and time 
    
    heat_rate = cp.Constant(np.array(gen_df["heat_rate_mmbtu_per_mwh"]))
    fuel_cost = cp.Constant(np.array(gen_df["fuel_cost"]))
    var_om_cost = cp.Constant(np.array(gen_df["var_om_cost_per_mwh"]))
    existing_cap = cp.Constant(np.array(gen_df["existing_cap_mw"]))
    start_cost = cp.Constant(np.array(gen_df["start_cost_per_mw"]))
    shut_cost = cp.Constant(np.array(gen_df["shut_cost_per_mw"]))

    #Define costs
    net_var_cost = cp.sum(GEN.T @ (cp.multiply(heat_rate, fuel_cost) + var_om_cost))
    net_startup_cost = cp.sum(START.T @ cp.multiply(existing_cap, start_cost))
    net_shutdown_cost = cp.sum(SHUT.T @ cp.multiply(existing_cap, shut_cost))

    #https://www.potomaceconomics.com/wp-content/uploads/2024/05/2023-State-of-the-Market-Report_Final_060624.pdf
    NSE_penalty=cp.sum(NSE) * 9000  
    
    curtailment_penalty = 0      #minimize curtailment
    curtailment_cost = curtailment_penalty * cp.sum(curtailment)

    #objective expression
    UCcost = net_var_cost + net_startup_cost + net_shutdown_cost + NSE_penalty + curtailment_cost
    
    
    """D. DEFINE CONSTRAINT EXPRESSIONS """
    constraints = []

    # D.1. SUPPLY-DEMAND BALANCE
    constraints.append(cp.sum(GEN, axis=0) + (DISCHARGE - CHARGE) + NSE == loads["demand"])
    constraints.append(NSE >= 0)
    
    # D.2.  CAPACITY/GENERATION CONSTRAINTS
    # D.2.1. thermal generators requiring commitment stay within power limits [P_min, P_max]
    constraints += [
        GEN[thermal_idx, :] >= cp.multiply(COMMIT[thermal_idx, :], (existing_cap_array * min_power_array)[:, None]),
        GEN[thermal_idx, :] <= cp.multiply(COMMIT[thermal_idx, :], existing_cap_array[:, None])
    ]
    
    # D.2.2. variable generation, accounting for hourly capacity factor (wind and solar)
    for i in G_var:
            constraints.append(
                GEN[i-1, :] <= gen_var_cf[gen_var_cf["r_id"] == i]["existing_cap_mw"] * \
                    gen_var_cf[gen_var_cf["r_id"] == i]["cf"])
    
    #D.2.3.1. curtailment penalty for G_var
    for j, i in enumerate(G_var):
        idx = i 
        cap = gen_var_cf.loc[gen_var_cf["r_id"] == i, "existing_cap_mw"].values[0]
        cf = gen_var_cf.loc[gen_var_cf["r_id"] == i, "cf"].values
        constraints += [
            curtailment[idx-1, :] == cap * cf - GEN[idx - 1, :]     #curt = aviail VRE-(went to dispatch+charging)
        ]
    
    #D.2.3.2. curtailment penalty for NOT G_var
    for i in G_thermal+G_phs:
        constraints.append(curtailment[i-1,:] == 0)
        
            
    # D.2.4. non-negativity of generation for all generators apart from PHS (specifically for variable gen, redundant for thermal gen)
    for i in G_nonphs:
        constraints.append(GEN[i-1, :] >= 0)
        constraints.append(GEN[i-1, :] <= gen_df[gen_df["r_id"] == i]["existing_cap_mw"])
        
        
    # D.3. COMMITMENT CONSTRAINTS (for G_thermal)
    # D.3.1. Minimum up time 
    for i in G_thermal:
        for t in range(len(T)): 
            start_expr = cp.sum(START[i-1, max(0, t-up_time):t]) 
            constraints.append(COMMIT[i-1, t] >= start_expr)

    # D.3.2. Minimum downtime which is 6 hrs if not constrained by reactivity 
    for i in G_thermal:
        match = D_points.loc[D_points['r_id'] == i, 'deadtime_value']
        dn_time = match.values[0] if not match.empty else min_dn_time  # default if i not in r_id
        for t in range(len(T)):
            constraints.append(
                1 - COMMIT[i-1, t] >= cp.sum(SHUT[i-1, max(0, t - dn_time):t])
            )
    
    # D.3.3. Commitment state
    constraints.append(
        COMMIT[thermal_idx, 1:] - COMMIT[thermal_idx, :-1] == START[thermal_idx, 1:] - SHUT[thermal_idx, 1:])

    # D.3.4 For non-thermals, COMMIT/START/SHUT should be 0 
    constraints += [
        COMMIT[nonthermal_idx, :] == 0,
        START[nonthermal_idx, :] == 0,
        SHUT[nonthermal_idx, :] == 0]

   
    """D.4. PMINSTABLE + RAMP UP/DOWN combined implementation from Jenkins et al. paper, but modified"""             
    # Constants
    M = 1e3          # Big-M matching max ramp capability
    
    # D.4.1 Auxiliary GEN offset from Pmin 
    constraints.append(
        GENAUX[thermal_idx, :] == GEN[thermal_idx, :] - cp.multiply(COMMIT[thermal_idx, :], pmin_array)
    )
    
    # D.4.2 Ramp-up constraint with bin acivation when gretaer than 1e-3
    constraints += [
        GENAUX[thermal_idx, 1:] - GENAUX[thermal_idx, :-1] >= 1e-3 - M * (1- Up[thermal_idx, :-1]),
        GENAUX[thermal_idx, 1:] - GENAUX[thermal_idx, :-1] <= cp.multiply(cap_rup, Up[thermal_idx, :-1]),
    ]
    
    # D.4.3 Ramp-down constraint with bin acivation when gretaer than 1e-3
    constraints += [
        GENAUX[thermal_idx, :-1] - GENAUX[thermal_idx, 1:] >= 1e-3 - M * (1- Rd[thermal_idx, :-1]),
        GENAUX[thermal_idx, :-1] - GENAUX[thermal_idx, 1:] <= cp.multiply(cap_rdn, Rd[thermal_idx, :-1]),
    ]
    
    
    # D.4.4. Exclusivity of states per priod for Rd, St, Up, only acvivates when gen is online (COMMIT=1)
    constraints.append(
        Rd[thermal_idx, :] + Up[thermal_idx, :] + St[thermal_idx, :] == COMMIT[thermal_idx, :]
    )

    # D.4.5. Enforce Min stable time after ramp down ends
    for t in range(len(T) - PMINSTABLE - 1):
        rampdown_end_expr = Rd[thermal_idx, t] - Rd[thermal_idx, t + 1]  # ∈ {0, 1}
        for k in range(1, PMINSTABLE + 1):
            tk = t + k
            constraints += [
                St[thermal_idx, tk] >= rampdown_end_expr,
                Up[thermal_idx, tk] <= 1 - rampdown_end_expr,
                Rd[thermal_idx, tk] <= 1 - rampdown_end_expr,
            ]

    #D.4.6 Ramp lim for G_var
    # Ramp-up: GEN[:, t] - GEN[:, t-1] <= capacity
    constraints.append(
        GEN[var_idx, 1:] - GEN[var_idx, :-1] <= cap_var_array
    )

    # Ramp-down: GEN[:, t-1] - GEN[:, t] <= capacity
    constraints.append(
        GEN[var_idx, :-1] - GEN[var_idx, 1:] <= cap_var_array
    )
            
    # D.5. PHS CONSTRAINTS [BATTERY]:
    one_way_efficiency = 0.84
    duration_hr = PHS_duration
    power_capacity = int(gen_df.loc[gen_df['resource'] == 'hydroelectric_pumped_storage', 'existing_cap_mw'].iloc[0])
    energy_capacity = power_capacity * duration_hr # MWh
    
    # D.5.1. PHS1. charge/discharge power constraints
    constraints.append(CHARGE >= 0)
    constraints.append(CHARGE <= power_capacity)
    constraints.append(DISCHARGE >= 0)
    constraints.append(DISCHARGE <= power_capacity)
    
    # D.5.2. PHS2. energy constraint
    constraints.append(SOC >= 0)
    constraints.append(SOC <= energy_capacity)
    
    # D.5.3. PHS3. state constraint
    for t in range(1,len(T)):  # T = CHARGE.shape[0]
        constraints.append(
            SOC[t] == SOC[t-1] + one_way_efficiency * CHARGE[t-1] - DISCHARGE[t-1] / one_way_efficiency
        )
        
    # D.5.4. SOC at BC
    constraints.append(CHARGE[-1] == 0)
    constraints.append(DISCHARGE[-1] == 0)

    # D.5.4.1 Transfer SOC from previous day
    #on day 1 the SOC at t=0 and t=72 is unconstrained, then based on the solution for day 1, the SOC for
    # the next day meaning SOC[0] is forced to continue using the state equation and SOC value at -1.
    if not first_run:
        # SOC_last_period is computed using the SOC transfer equation D.5.3.
        SOC_last_period = commit_at_len_T['soc_transfer'].iloc[-1]
        constraints.append(SOC[0] == SOC_last_period)

    # D.5.5. PHS5: CHARGE - DISCHARGE = GEN for PHS [inactive, added to supply demand constraint]
    for i in G_phs:
        constraints.append(GEN[i-1, :] == 0)
    
    # D.5.6. Add constraints linking CHARGE_MODE to CHARGE and DISCHARGE, so CHARGE/DISCHARGE dont overlap
    constraints += [
        CHARGE <= cp.multiply(CHARGE_MODE, power_capacity),
        DISCHARGE <= cp.multiply(1 - CHARGE_MODE, power_capacity)
    ]

    
    # D.6 REMIANING DOWNTIME ENFORCE remaining_downtime across time periods
    gen_array = remaining_downtime["generator"].astype(int).values
    rem_time_array = remaining_downtime["remaining_downtime"].astype(int).values

    #init set of all generators that are in downtime from previous dispatch
    constrained_by_downtime = set()

    #D.6.1. make COMMIT=0 for all gen that are in downtime observance
    for i, remaining_time in zip(gen_array, rem_time_array):
        if remaining_time > 0:
            constrained_by_downtime.add(i)
            constraints += [COMMIT[i-1, t] == 0 for t in range(min(remaining_time, len(T)))]
    
    #D.6.2. if NOT constrained by downtime then COMMIT state at t=0 is determine using COMMIT state during last dispatch ending 
    for idx, row in commit_at_len_T.iterrows():
        g = int(row['generator'])
        commitment_value = int(row['commitment'])
    
        # Skip constraint if generator is forced off due to downtime
        if g in constrained_by_downtime:
            continue
    
        # commitment continuity constraint
        constraints.append(COMMIT[g - 1, 0] >= commitment_value - SHUT[g - 1, 0])
        constraints.append(COMMIT[g - 1, 0] <= commitment_value + START[g - 1, 0])

    #D.6.3 dynamic_downtime assignent for transfer
    for i in G:
        # Check if r_id exists in D_points
        match = D_points.loc[D_points['r_id'] == i, 'deadtime_value']
        deadtime_val = match.values[0] if not match.empty else 0.0
    
        for t in range(len(T)):
            constraints.append(
                dynamic_downtime[i-1, t] == deadtime_val * SHUT[i-1, t]
            )

    """E. OPTIMIZATION SETUP """
    prob = cp.Problem(cp.Minimize(UCcost), constraints)
    
    return {
        "prob": prob,
        "UCcost": UCcost,
        "GEN": GEN,
        "Rd": Rd,
        "Up": Up,
        "St": St,
        "NSE":NSE,
        "COMMIT": COMMIT,
        "SHUT": SHUT,
        "START": START,
        "gen_var_cf": gen_var_cf,
        "dynamic_downtime": dynamic_downtime,
        "CHARGE": CHARGE,
        "DISCHARGE": DISCHARGE,
        "SOC":SOC

    }
