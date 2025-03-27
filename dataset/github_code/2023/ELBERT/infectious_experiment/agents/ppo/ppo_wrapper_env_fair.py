import copy

import gym
import numpy as np
from gym import spaces

from networkx.algorithms import community

# the following should be in the env_param_dict
# from infectious_experiment.config_fair import EP_TIMESTEPS,ZETA_1, ZETA_0

class PPOEnvWrapper_fair(gym.Wrapper):
    '''
    the reward (after applying this wrapper) will be of the form:
    [main_reward, [r_u_0,r_u_1,r_u_2], [r_b_0, r_b_1, r_b_2]] (assuming there are 3 groups)
    '''
    def __init__(self,
                 env,
                 reward_fn,
                 env_param_dict):
        super(PPOEnvWrapper_fair, self).__init__(env)

        self.env = env

        shape = self.format_observation(
            self.env.observation_space.sample()).shape

        self.observation_space = spaces.Box(
            low=np.inf,
            high=np.inf,
            shape=shape,
        )

        self.action_space = spaces.Discrete(
            n=shape[0] + 1,
        )

        self.reward_fn = reward_fn()

        self.timestep = 0
        self.ep_timesteps = env_param_dict['ep_timesteps']
        self.zeta_0 = env_param_dict['zeta_0']
        self.zeta_1 = env_param_dict['zeta_1']

        communities_generator = community.girvan_newman(self.env.state.population_graph)
        self.communities = tuple(sorted(c) for c in next(communities_generator))
        self.num_communities = len(self.communities)

        self.num_groups = self.num_communities       # for using non env specific implementation

        # Map individuals in the graph to a community
        self.communities_map = {
            individual: comm_i for comm_i, comm in enumerate(self.communities) for individual in comm
        }

        # Keep track of how many vaccines go to each community (cummulative)
        self.num_vaccines_per_community = np.zeros(self.num_communities)           # Not used by reward
        # Keep track of previous health states to compute newly infected number
        self.prev_health_states = copy.deepcopy(self.env.state.health_states)
        # Newly infected in each community (cummulative)
        self.num_newly_infected_per_community = np.zeros(self.num_communities)     # Not used by reward

        # only for APPO
        self.delta = 0
        self.delta_delta = 0


    def format_observation(self, obs):
        """Formats health state observations into a numpy array.
        The health-states are one-hot encoded as row vectors, and then stacked
        together vertically to create a |population| x |health states| array.

        The population is padded on top and bottom with "recovered" indivduals,
        which don't affect the disease spread but make convolutions simpler.
        Args:
          obs: An observation dictionary.
        Returns:
          A numpy array suitable for passing to a DQN agent.
        """
        vecs = []
        initial_params = self.env.initial_params
        num_states = len(initial_params.state_names)
        for state in obs['health_states']:
            vecs.append(np.zeros((num_states, 1), dtype=float))
            vecs[-1][state] = 1.0
        return np.hstack(vecs).T

    def process_action(self, action):
        if action == self.action_space.n - 1:
            return None
        return np.array([action])

    def reset(self):
        self.timestep = 0
        # Keep track of how many vaccines go to each community
        self.num_vaccines_per_community = np.zeros(self.num_communities)
        # Keep track of previous health states to compute newly infected number
        self.prev_health_states = copy.deepcopy(self.env.state.health_states)
        # Newly infected in each community
        self.num_newly_infected_per_community = np.zeros(self.num_communities)

        # only for APPO
        self.delta = 0
        self.delta_delta = 0

        return self.format_observation(self.env.reset())

    def step(self, action):

        self.prev_health_states = copy.deepcopy(self.env.state.health_states)

        action = self.process_action(action)
        obs, _, done, info = self.env.step(action)

        # fairness signal:
        r_U = np.zeros(self.num_communities) # number of vaccine per group at the current step
        r_B = np.zeros(self.num_communities) # number of newly infected individuals per group at the current step

        # Update the number of vaccines in each community
        if action is not None:
            # NOTE: it seems that this code only works when the number of vaccines <= 1 per step. 
            comm_i = self.communities_map[action[0]]
            self.num_vaccines_per_community[comm_i] += 1

            r_U[comm_i] += 1

        # Compute newly infected
        for i, (health_state, prev_health_state) in enumerate(zip(self.env.state.health_states, self.prev_health_states)):
            # 1 stands for infectious
            if health_state == 1 and health_state != prev_health_state:
                comm_i = self.communities_map[i]
                self.num_newly_infected_per_community[comm_i] += 1

                r_B[comm_i] += 1

        r = self.reward_fn(health_states=self.env.state.health_states,
                           num_vaccines_per_community=self.num_vaccines_per_community,
                           num_newly_infected_per_community=self.num_newly_infected_per_community,
                           eta0=self.zeta_0,
                           eta1=self.zeta_1)
        
        old_delta = self.delta
        self.delta = self.reward_fn.calc_delta(num_vaccines_per_community=self.num_vaccines_per_community,
                                               num_newly_infected_per_community=self.num_newly_infected_per_community)
        self.delta_delta = self.delta - old_delta
        
        self.timestep += 1
        if self.timestep == self.ep_timesteps:
            done = True

        return self.format_observation(obs), [r,r_U,r_B], done, info

