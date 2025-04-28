import math
import random
import numpy as np
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time
import copy

def cell_interaction_ABM(params,migration_rules,T_end=5.0,xn=200,yn=40,perc=0.75):

    """
    Simulate the Pulling or Adhesion ABM

    Args:
        params (tuple): A tuple containing reaction rate parameters, where
            params[0] (float): Rate of agent migration (rm).
            params[1] (float): Rate of agent interaction (rint). This is Ppull for the Pulling ABM and Padh for the Adhesion ABM
        migration_rules (callable): Function defining agent migration rules.
        T_end (float, optional): End time of the simulation. Default is 5.0.
        xn (int, optional): Number of lattice sites along the x-axis. Default is 200.
        yn (int, optional): Number of lattice sites along the y-axis. Default is 40.
        perc (float, optional): Initial occupancy percentage of lattice sites. Default is 0.75.

    Returns:
        A_out (numpy.ndarray): 2D array representing agent profiles over time and space.
        t_out (numpy.ndarray): Array of equispaced time points.
        x_out (numpy.ndarray): Array of lattice site positions along the x-axis.
        plot_list (list): List of snapshots of the ABM at specific time intervals.

    The function simulates the ABM with agent migration events. It tracks time, agent proportions, and snapshots of the ABM during the simulation.

    Note:
        - The `migration_rules` function should take input parameters like `A`, `params`, `loc`, `migration_loc`, `neighbor_loc`):. It should be 'migration_step_pulling' for the Pulling ABM and `migration_step_adhesion` for the Adhesion ABM

    Example:
        A_out, t_out, x_out, plot_list = cell_interaction_ABM((0.2, 0.1), migration_step_pulling, T_end=10.0)
    """
    
    rm, rint = params[:2]
    
    A = IC_initialization(yn,xn,[perc])
    
    #count number of occupied lattice sites.
    A_num = np.sum(A==1)

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    A_list = [A_num]
    A_profiles = [np.sum(A,axis=0)/yn]
    plot_list = [copy.deepcopy(A)]
    
    #number of snapshots saved so far in plot_list
    image_count = 1
    
    count = 0
    while t_list[-1] < T_end:

        a = rm*A_num
        
        tau = -np.log(np.random.uniform())/a
        t += tau

        #agent movement
        A = migration_trimolecular_reaction(A,
                                            params = (rint,),
                                            states = [1],
                                            migration_rules = migration_rules)

        #count number of occupied lattice sites
        A_num = np.sum(A==1)
        
        #only save every 100 steps
        if count%100 == 0:
            #append information to lists
            t_list.append(t)
            A_list.append(A_num)
            A_profiles.append(np.sum(A,axis=0)/yn)
        
            if (t_list[-2] < image_count*T_end/100 and t_list[-1] >= image_count*T_end/100): 
                plot_list.append(copy.deepcopy(A))
                image_count+=1
        count+=1

    #interpolation to equispace grid
    x_out = np.arange(xn)
    t_out = np.linspace(0,T_end,100)
    A_profiles = np.array(A_profiles)
    
    f = interpolate.interp2d(t_list,x_out,A_profiles.T)
    A_out = f(t_out,x_out)

    return A_out, t_out, x_out, plot_list

def cell_interaction_ABM_multistates(params,migration_rules,T_end=5.0,xn=200,yn=40):
    
    """
    Simulate the Pulling & Adhesion ABM.

    Args:
        params (tuple): A tuple containing model parameters for different agent types.
            - params[0] (float): Migration rate for adhesive agents (rmH).
            - params[1] (float): Migration rate for pulling agents  (rmP).
            - params[2] (float): Adhesion probability (radh).
            - params[3] (float): Pulling probability (rpull).
            - params[4] (float): Alpha parameter specifying proportion of adhesive agents.
        migration_rules (callable): Function defining agent migration rules.
        T_end (float, optional): End time of the simulation. Default is 5.0.
        xn (int, optional): Number of lattice sites along the x-axis. Default is 200.
        yn (int, optional): Number of lattice sites along the y-axis. Default is 40.

    Returns:
        A_out (numpy.ndarray): 2D array representing agent profiles over time and space.
        compartments_out (list of numpy.ndarray): List of agent compartment profiles over time and space.
        t_out (numpy.ndarray): Array of equispaced time points.
        x_out (numpy.ndarray): Array of lattice site positions along the x-axis.
        plot_list (list): List of snapshots of the ABM at specific time intervals.

    Example:
        A_out, compartments_out, t_out, x_out, plot_list = cell_interaction_ABM_multistates((0.1, 0.2, 0.3, 0.4, 0.5), migration_reaction_step_adhesion_pulling, T_end=10.0)
    """
    
    rmH, rmP, radh, rpull, alpha = params
    
    assert (alpha >= 0) and (alpha <= 1.0)
    
    A = IC_initialization(yn,xn,percs=[0.75*alpha ,.75*(1-alpha)])
    
    #count number of occupied lattice sites.
    compartment_nums = [np.sum(A==i) for i in np.arange(1,3)]
    A_num = np.sum(compartment_nums)

    #initialize time
    t = 0

    #track time, agent proportions, and snapshots of ABM in these lists
    t_list = [t]
    A_list = [A_num]
    A_profiles = [np.sum(A>0,axis=0)/yn]
    
    compartment_list = [compartment_nums]
    compartment_profiles = [[np.sum(A==i,axis=0)/yn for i in np.arange(1,3)]]
    
    plot_list = [np.copy(A)]
    
    #number of snapshots saved so far
    image_count = 1
    
    count = 0
    while t_list[-1] < T_end and A_num != 0:

        a = rmH*compartment_nums[0] + rmP*compartment_nums[1]
        
        tau = -np.log(np.random.uniform())/a
        t += tau

        Action = a*np.random.uniform()

        if Action <= rmH*compartment_nums[0]:
            #H agent movement
            A = migration_trimolecular_reaction(A,
                                                states = [1],
                                                params = (radh, rpull), 
                                                migration_rules = migration_rules)
            
        elif Action <= rmH*compartment_nums[0] + rmP*compartment_nums[1]:
            #H agent movement
            A = migration_trimolecular_reaction(A,
                                                states = [2],
                                                params = (radh, rpull), 
                                                migration_rules = migration_rules)

        #count number of occupied lattice sites
        compartment_nums = [np.sum(A==i) for i in np.arange(1,3)]
        A_num = np.sum(compartment_nums)
        
        #only save every 100 steps
        if count%100 == 0:
            #append information to lists
            t_list.append(t)
            A_list.append(A_num)
            A_profiles.append(np.sum(A>0,axis=0)/yn)
            compartment_list.append(compartment_nums)
            compartment_profiles.append([np.sum(A==i,axis=0)/yn 
                                         for i in np.arange(1,3)])
            if (t_list[-2] < image_count*T_end/100 and t_list[-1] >= image_count*T_end/100): 
                plot_list.append(np.copy(A))
                image_count+=1
            
        count+=1

            
    #interpolation to equispace grid
    x_out = np.arange(xn)
    t_out = np.linspace(0,T_end,100)
    A_profiles = np.array(A_profiles)

    f = interpolate.interp2d(t_list,x_out,A_profiles.T)
    A_out = f(t_out,x_out)
    
    compartments_out = []
    compartment_profiles = np.array(compartment_profiles)
    compartment_profiles = [compartment_profiles[:,i,:] for i in np.arange(2)]
    
    for compartment_profile in compartment_profiles:
        f = interpolate.interp2d(t_list,x_out,compartment_profile.T)
        compartments_out.append(f(t_out,x_out))
        
    return A_out, compartments_out, t_out, x_out, plot_list


def migration_step(A,loc,migration_loc):
    """
    Perform a migration step

    Args:
        A (numpy.ndarray): An array representing the state of agents, where 0 represents an empty space and 1 represents an agent.
        loc (tuple): The location of the current agent.
        migration_loc (tuple): The location to which the agent is migrating.
        
    Returns:
        A (numpy.ndarray): The updated state of agents after the migration step.
    """
    state = A[loc]
    
    if A[migration_loc] == 0:
            #agent moves into migration_loc
            A[migration_loc] = state
            A[loc] = 0                                
    return A

def migration_step_pulling(A,params,loc,migration_loc,neighbor_loc):
    
    """
    Perform a single step of the Pulling ABM.

    Args:
        A (numpy.ndarray): An array representing the state of agents, where 0 represents an empty space and 1 represents an agent.
        params (list): A list containing a single parameter, Ppull, which is the probability of a pulling event.
        loc (tuple): The location of the current agent.
        migration_loc (tuple): The location to which the agent is migrating.
        neighbor_loc (tuple): The location of the neighboring agent.

    Returns:
        A (numpy.ndarray): The updated state of agents after the migration step.

    The function performs Rules A and B based on A[migration_loc] and A[neighbor_loc]
    """
    
    assert len(params)==1
    Ppull = params[0]
    
    #Can only move if chosen migration location is empty
    if A[migration_loc] == 0:
        
        #move to migration location
        A[migration_loc] = 1
        
        if A[neighbor_loc] == 1 and np.random.uniform() < Ppull:
            #Rule B -- successful pulling event
            A[loc] = 1
            A[neighbor_loc] = 0
        else:
            #Either Rule A (if A[neighbor_loc] == 0)
            #or Rule B -- unsuccessful pulling event
            A[loc] = 0            
                    
    return A

def migration_step_adhesion(A,params,loc,migration_loc,neighbor_loc):
    
    """
    Perform a single step of the Adhesion ABM.

    Args:
        A (numpy.ndarray): An array representing the state of agents, where 0 represents an empty space and 1 represents an agent.
        params (list): A list containing a single parameter, Padh, which is the probability of an adhesion event.
        loc (tuple): The location of the current agent.
        migration_loc (tuple): The location to which the agent is migrating.
        neighbor_loc (tuple): The location of the neighboring agent.

    Returns:
        A (numpy.ndarray): The updated state of agents after the migration step.

    The function performs Rules A and B based on A[migration_loc] and A[neighbor_loc]
    """
    
    assert len(params)==1
    Padh = params[0]
    
    #Can only move if chosen migration location is empty
    if A[migration_loc] == 0:
        if A[neighbor_loc] == 1 and np.random.uniform() < Padh:
            #Rule D adhesion event -- migration event aborted
            pass
        else:
            #Rule C or Rule D with unsuccessful adhesion event
            A[migration_loc] = 1
            A[loc] = 0            

    return A


def migration_trimolecular_reaction(A, states, params, migration_rules):

    """
    Perform a trimolecular reaction involving agent migration and interaction.

    Args:
        A (numpy.ndarray): 2D array representing the lattice with agents.
        states (list): List of agent states involved in the reaction. [1] denotes adhesive agent, 2 denotes pulling agents
        params (tuple): A tuple containing reaction-specific parameters.
        migration_rules (callable): Function defining the interaction step.

    Returns:
        A (numpy.ndarray): Updated lattice array after the reaction.
        
    """
    
    assert type(params) == tuple
    
    yn, xn = A.shape

    # Select Random agent
    for i,state in enumerate(states):
        agent_loc_tmp = A == state
        if i == 0:
            agent_loc = agent_loc_tmp
        else:
            agent_loc = np.logical_or(agent_loc,agent_loc_tmp)
    agent_loc = np.where(agent_loc)
    agent_ind = np.random.permutation(len(agent_loc[0]))[0]
    loc = (agent_loc[0][agent_ind],agent_loc[1][agent_ind])
    
    ### Determine direction
    dir_select = np.ceil(np.random.uniform(high=4.0))
    
    #downward
    if dir_select == 1 and loc[0] < yn-1:
        dy = 1
        dx = 0
        
        if loc[0] > 0:
            #perform interaction-based migration in interior
            reaction = "interaction"
        else:
            #perform simple migration at boundary (no neighboring lattice site)
            reaction = "migration"
    
    #upward
    elif dir_select == 2 and loc[0] > 0:
        dy = -1
        dx = 0
        
        if loc[0] < yn-1: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
        
    #rightward    
    elif dir_select == 3 and loc[1] < xn-1:
        dy = 0
        dx = 1
        
        if loc[1] > 0: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
            
    #leftward        
    elif dir_select == 4 and loc[1] > 0: #left
        dy = 0
        dx = -1
        
        if loc[1] < xn-1: #interior
            reaction = "interaction"
        else:
            reaction = "migration"
            
    else:
        dy = 0
        dx = 0
        reaction = "aborted"
        
    migration_loc = (loc[0]+dy,loc[1]+dx)
    neighbor_loc  = (loc[0]-dy,loc[1]-dx)

    if reaction == "interaction":
        A = migration_rules(A,params,loc,migration_loc,neighbor_loc)       
    elif reaction == "migration":
        A = migration_step(A,loc,migration_loc)
    elif reaction == "aborted":
        pass

    return A

def migration_reaction_step_adhesion_pulling(A,params,loc,migration_loc,neighbor_loc):
    
    """
    Perform a migration reaction step with adhesion and pulling interactions.

    Args:
        A (numpy.ndarray): 2D array representing the lattice with agents.
        params (tuple): A tuple containing parameters for adhesion and pulling.
            - params[0] (float): Adhesion probability (Pa).
            - params[1] (float): Pulling probability (Pp).
        loc (tuple): Current location of the agent.
        migration_loc (tuple): New location for the migrating agent.
        neighbor_loc (tuple): Location of the neighboring agent.

    Returns:
        A (numpy.ndarray): Updated lattice array after the migration reaction step.

    Example:
        new_lattice = migration_reaction_step_adhesion_pulling(my_lattice, (0.2, 0.3), (3, 4), (3, 5), (3,3))
    """
    
    ### values of 1 denote adhesive agents, values of 2 denote pulling agents
    
    Pa, Pp = params
    
    #can only move into empty space
    if A[migration_loc] == 0: 
            
        state = A[loc]
        state_neighbor = A[neighbor_loc]

        #Rule A or C
        if state_neighbor == 0:
            A[migration_loc] = state
            A[loc] = 0            

        #Rule F (P-H-0)
        elif state == 1 and state_neighbor == 2:
            A[migration_loc] = state
            A[loc] = 0

        #Rule D (H-H-0)
        elif state == 1 and state_neighbor == 1:
            event = np.random.uniform()

            if event < Pa:
                #migration aborted due to adhesion
                pass
            else:
                #migration event unsuccessful
                A[migration_loc] = state
                A[loc] = 0

        #Rule B (P-P-0)
        elif state == 2 and state_neighbor == 2:
            event = np.random.uniform()

            if event < Pp:
                #Successful pulling event
                A[migration_loc] = state
                A[loc] = state_neighbor
                A[neighbor_loc] = 0
            else:
                #Unsuccessful pulling event
                A[migration_loc] = state
                A[loc] = 0

        #Rule E (H-P-0)
        elif state == 2 and state_neighbor == 1:
            event = np.random.uniform()

            if event < Pa:
                #migration aborted due to adhesion
                pass
            elif event < Pa + Pp:
                #Successful pulling event
                A[migration_loc] = state
                A[loc] = state_neighbor
                A[neighbor_loc] = 0
            else:
                #Both adhesion and pulling unsuccessful
                A[migration_loc] = state
                A[loc] = 0
                
    return A

def IC_initialization(yn,xn,percs=[0.5]):
    
    """
    Initialize an ABM lattice with specified agent distributions.

    Args:
        yn (int): Number of lattice sites along the y-axis.
        xn (int): Number of lattice sites along the x-axis.
        percs (list, optional): List of occupancy percentages for agent types. Default is [0.5].

    Returns:
        A (numpy.ndarray): A 2D array representing the initialized ABM lattice.

    Example:
        initial_lattice = IC_initialization(40, 200, [0.6, 0.3]) will place adhesive agents (1) in 60% of the lattice sites and pulling agents (2) in 30% of the lattice sites and leave the remaining 10% of lattice sites empty
    """
    
    A = np.zeros((yn,xn))

    #will initialize the middle 20% of the lattice
    xn_interior = np.arange(int(.4*xn),int(.6*xn))[1:]

    for i in xn_interior:
        #Place agents in perc% of y-locations for each x location in interior
        perm = np.random.permutation(yn)

        index_lb = 0
        for j in np.arange(len(percs)):
            index_ub = int(index_lb + percs[j]*yn)
            yn_perm = perm[index_lb : index_ub]
            A[yn_perm,i] = j+1
            index_lb = index_ub


        
    return A

def simulate_nonlinear_migration_ABM(params, migration_rules, interaction_string, T_end = 1000.0, perc=0.75, n=5):
    
    """
    Simulate and save the Pulling or Adhesion ABM

    Args:
        params (tuple): A tuple containing parameters for the simulation, including:
            - params[0] (float): Migration probability parameter (Pm).
            - params[1] (float): Pulling or Adhesion strength parameter (Pint).
        migration_rules (callable): A function defining the interaction rate between agents.
        interaction_string (str): A string indicating the type of interaction ('pulling' or 'adhesion').
        T_end (float, optional): The end time of the simulation. Default is 1000.0.
        perc (float, optional): A parameter controlling the percentage of sites initially occupied. Default is 0.75.
        n (int, optional): The number of simulations to perform. Default is 5.
    
    Returns:
        None

    The function runs `n` simulations of the specified ABM, and saves the data to a file.

    Example:
        To simulate the Adhesion ABM with Pm=0.2 and Padh=0.5 for 10 simulations, you can call:
        simulate_nonlinear_migration_ABM((0.2, 0.5), migration_rules, 'adhesion', n=10)
    """
    
    Pm   = round(params[0],3)
    Pint = round(params[1],3)
    
    #Initialize list containing all individual simulations
    Cs = []
    #Initialize dictionary storing all saved information
    data = {}
    #Used to track computation time
    computationTimeAll0 = time.time()
    
    for i in np.arange(n):
        
        C_out, t_out, x_out, plot_list = cell_interaction_ABM(params,
                                                              migration_rules = migration_rules,
                                                              T_end=T_end,
                                                              perc=perc)
        #store computed information
        Cs.append(C_out)
        data[f'c{i}'] = C_out
    
    #record time of all simulations
    computationTimeAllFinal = time.time() - computationTimeAll0
    
    #convert to np arrays, get mean data
    Cs = np.array(Cs)    
    C_mean = np.mean(Cs,axis=0)

    T,X = np.meshgrid(t_out,x_out)

    if interaction_string == "pulling":
        Pint_string = "Ppull"
    if interaction_string == "adhesion":
        Pint_string = "Padh"
    
    data['Pm'] = Pm
    data[Pint_string] = Pint

    data['x'] = x_out[:,None]
    data['X'] = X
    data['t'] = t_out[:,None].T
    data['T'] = T
    data['C'] = C_mean
    data['time_all'] = computationTimeAllFinal
    np.save(f"../../data/simple_{interaction_string}_mean_{n}_Pm_{Pm}_{Pint_string}_{Pint}.npy",data)  
    
def simulate_heterogeneous_ABM(params, migration_rules, T_end = 1000.0, n=5):
    
    """
    Simulate the Pulling & Adhesion ABM.

    Args:
        params (tuple): A tuple containing model parameters.
            - params[0] (float): Rate of migration for adhesive agents (PmH).
            - params[1] (float): Rate of migration for pulling agents (PmP).
            - params[2] (float): Probability of adhesion (Padh).
            - params[3] (float): Probability of pulling (Ppull).
            - params[4] (float): Alpha parameter specifying proportion of adhesive agents.
        migration_rules (callable): Function defining agent migration rules.
        T_end (float, optional): End time of the simulation. Default is 1000.0.
        n (int, optional): Number of simulation iterations. Default is 5.

    Returns:
        None

    This function performs multiple ABM simulations and saves the mean data to a file.

    Example:
        simulate_heterogeneous_ABM((0.1, 0.2, 0.3, 0.4, 0.5), migration_reaction_step_adhesion_pulling, T_end=500.0, n=10)
    """
    
    PmH, PmP, Padh, Ppull, alpha = params
    
    PmH = round(PmH,3)
    PmP = round(PmP,3)
    Padh = round(Padh,3)
    Ppull = round(Ppull,3)
    alpha = round(alpha,3)
    
    #Initialize list containing all individual simulations
    Cs = []
    #Initialize dictionary storing all saved information
    data = {}
    #Used to track computation time
    computationTimeAll0 = time.time()
    
    for i in np.arange(n):
        
        #simulate ABM
        C_out, compartments_out ,t_out, x_out, plot_list = cell_interaction_ABM_multistates(params,
                                         migration_rules = migration_rules,
                                         T_end=T_end)
        
        #store computed information
        Cs.append(C_out)
        compartments.append(compartments_out)
        data[f'c{i}'] = C_out
    
    #record time of all simulations
    computationTimeAllFinal = time.time() - computationTimeAll0
    
    #convert to np arrays, get mean data
    Cs = np.array(Cs)    
    C_mean = np.mean(Cs,axis=0)
    compartments_mean = np.mean(np.array(compartments),axis=0)

    T,X = np.meshgrid(t_out,x_out)


    data['PmH'] = PmH
    data['PmP'] = PmP
    data['Ppull'] = Ppull
    data['Padh'] = Padh
    data['alpha'] = alpha
    data['time_all'] = computationTimeAllFinal
    
    data['compartments'] = compartments_mean

    data['x'] = x_out[:,None]
    data['X'] = X
    data['t'] = t_out[:,None].T
    data['T'] = T
    data['C'] = C_mean

    np.save(f"../../data/adhesion_pulling_mean_{n}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}.npy",data)    