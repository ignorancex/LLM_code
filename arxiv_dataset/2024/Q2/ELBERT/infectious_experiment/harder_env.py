'''
New Lending Env with the following modification from APPO's code
  1. Add vaccine disable probability (P(recovered->suspectible)) so that recovered is not an absorbing state anymore 
'''

from infectious_experiment.environments.infectious_disease import *

########## Env Parameters ##########
INFECTION_PROBABILITY = 0.5
INFECTED_EXIT_PROBABILITY = 0.005
NUM_TREATMENTS = 1
BURNIN = 1
GRAPH_NAME = 'karate'

GRAPHS = {'karate': nx.karate_club_graph()}

############################ new env Parameters ############################
DISABLE_PROBILITY = 0.2

############################ create new env ############################

def create_GeneralInfectiousDiseaseEnv():
  '''
  Directly create the new version of the env, without any argument 
  (all env parameters are hard-wired in the file)
  '''

  graph = GRAPHS[GRAPH_NAME]
    # Randomly initialize a node to infected
  initial_health_state = [0 for _ in range(graph.number_of_nodes())]
  initial_health_state[0] = 1
    
  transition_matrix = np.array([
    [0, 0, 0],
    [0, 1 - INFECTED_EXIT_PROBABILITY, INFECTED_EXIT_PROBABILITY],
    [DISABLE_PROBILITY, 0, 1 - DISABLE_PROBILITY]])

  env_params = Params(
      population_graph=graph,
      transition_matrix=transition_matrix,
      treatment_transition_matrix=np.array([[0, 0, 1],
                                            [0, 1, 0],
                                            [0, 0, 1]]),
      state_names=['susceptible', 'infected', 'recovered'],
      healthy_index=0,
      infectious_index=1,
      healthy_exit_index=1,
      infection_probability=INFECTION_PROBABILITY,
      initial_health_state=copy.deepcopy(initial_health_state),
      initial_health_state_seed=100,
      num_treatments=NUM_TREATMENTS,
      max_treatments=1,
      burn_in=BURNIN)

  env = InfectiousDiseaseEnv(env_params)
  return env