import numpy as np

from hmmlearn import hmm
import json
import gym
from typing import Tuple, Dict
import importlib.resources
import json
import logging

class EnvModeTrajectory():
    NORMAL = "NORMAL" #steps are all simulated
    BACKTESTING = "BACKTESTING" #steps are taken from real world trajectory (only a single trajectory
    PRESAMPLED = "PRESAMPLED" #presampled

class EnvMode():
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"

class FinancialMarketsEnv(gym.Env):

    def __init__(self, terminal_time_step=12, include_cash_asset: bool=True,
                 include_unobservable_market_state_information_for_evaluation: bool=True,
                 data_set_name: str="model_parameter_data_set_G_markov_states_2_4",#"model_parameter_data_set_A_currency_USD_markov_states_2",
                 one_sided_spread_fee: float= 0.000,
                 short_selling_yearly_fee: float= 0.03,
                 seed=1,
                 env_mode_trajectory=EnvModeTrajectory.NORMAL,
                 env_mode=EnvMode.TRAINING,
                 total_number_predetermined_eval_trajectory=None,
                 total_number_predetermined_training_trajectory=None,
                 reset_periodicity_eval_env=None,
                 list_reordering=None,
                 **kwargs):

        self.current_hidden_market_state = None
        self.current_predicted_market_state = None
        self.current_sampled_output = None
        self.current_startprob = None


        self.current_time_step = 0
        self.terminal_time_step = terminal_time_step

        self.model_choice = data_set_name

        self.amount_states, amount_features, means_, covars_, transmat_, self.initial_startprob_ = \
            self.initializing_hmm_model_parameters(list_reordering)

        self.dict_ordered_returns_backtesting, self.dict_ordered_statistics_mean, self.dict_ordered_statistics_cov = \
            self.initializing_backtesting_parameters(list_reordering)

        self.model = hmm.GaussianHMM(n_components=self.amount_states, covariance_type="full", random_state=seed)

        self.model.startprob_ = self.initial_startprob_
        self.model.transmat_ = transmat_
        self.model.means_ = means_
        self.model.covars_ = covars_

        self.include_cash_asset = include_cash_asset
        self.include_unobservable_market_state_information_for_evaluation = include_unobservable_market_state_information_for_evaluation

        self.amount_assets = amount_features

        if self.include_cash_asset:
            self.amount_assets+=1

        self.np_predetermined_eval_trajectories, self.total_number_predetermined_trajectory_available\
            = self.initialize_predetermined_eval_trajectories(list_reordering)

        self.is_backtesting_mode = False
        self.environment_seed = seed
        self.set_env_mode_trajectory(env_mode_trajectory, seed=seed)
        self.set_env_mode(env_mode)
        self.reset_periodicity_eval_env = reset_periodicity_eval_env
        self.trajectories_counter = 0

        self.current_predetermined_eval_trajectory_id = None
        self.np_rng = np.random.default_rng(seed)

        if total_number_predetermined_eval_trajectory is not None:
            self.total_number_predetermined_eval_trajectory = total_number_predetermined_eval_trajectory
        else:
            self.total_number_predetermined_eval_trajectory = self.total_number_predetermined_trajectory_available

        if total_number_predetermined_training_trajectory is not None:
            self.total_number_predetermined_training_trajectory = total_number_predetermined_training_trajectory
        else:
            self.total_number_predetermined_training_trajectory = self.total_number_predetermined_trajectory_available

        self.ask_spread_vector = np.ones((self.amount_assets), dtype=float)*one_sided_spread_fee  #this is for how much we can buy an asset
        self.ask_spread_vector[0] = 0.0 # buying cash does not cost anything (but we have to pay the bid spread of the asset we go out of)

        self.bid_spread_vector = np.ones((self.amount_assets),
                                         dtype=float) * one_sided_spread_fee  # this is for how much we can buy an asset
        self.bid_spread_vector[
            0] = 0.0  # selling cash does not cost anything (but we have to pay the ask spread of the asset we go into in)

        self.short_selling_yearly_fee_vector = np.ones((self.amount_assets), dtype=float)*short_selling_yearly_fee

        # portfolio specific state information
        self.current_state_portfolio_allocation = np.zeros((self.amount_assets), dtype=float)  # numpy vector
        self.current_state_portfolio_allocation[0] = 1.0 # setting initial allocation to 100% cash
        self.current_state_portfolio_wealth = 100  # None # float

        # Portfolio allocation which has to sum up to 1, i.e. action_space is a simplex
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.amount_assets,), dtype=float)

        #Allowing for short selling
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amount_assets,), dtype=float)

        # Box is for R valued observation space while MultiDiscrete is for multidimensional discrete action space
        amount_observations = self.amount_assets + self.amount_assets + 1
        # +1 for portfolio wealth
        # +self.amount_assets for currently held portfolio allocation
        if include_unobservable_market_state_information_for_evaluation:
            amount_observations += 1
        # +1 for the hidden state //give back portfolio state and portfolio wealth

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(amount_observations,), dtype=float)


    def set_backtesting_mode(self, is_backtesting_mode):
        if is_backtesting_mode==True:
            self.set_env_mode_trajectory(EnvModeTrajectory.BACKTESTING)
        else:
            self.set_env_mode_trajectory(EnvModeTrajectory.NORMAL)

    def set_environment_seed(self, seed):
        # necessary to controll the hmm model sampling
        np.random.seed(seed)
        # this controls the internal random number generator
        self.np_rng = np.random.default_rng(seed)

    def set_env_mode_trajectory(self, env_mode_trajectory, seed=None):
        """
        This was later added to allow for multiple evn modes
        :return:
        """
        if seed is not None:
            self.set_environment_seed(seed=seed)
        self.env_mode_trajectory = env_mode_trajectory

    def set_env_mode(self, env_mode):
        self.env_mode = env_mode

    def initializing_hmm_model_parameters(self, list_reordering=None)-> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:

        #https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_text("financial_markets_gym.envs.models", f"{self.model_choice}.json") as file:
            parameter_dict = json.load(file)

        means_ = np.array(parameter_dict.get("model.means_"))

        amount_states = means_.shape[0]
        amount_features = means_.shape[1]

        transmat_ = np.array(parameter_dict.get("model.transmat_"))
        covars_ = np.array(parameter_dict.get("model.covars_"))
        startprob_ = np.array(parameter_dict.get("model.startprob_"))

        #in case that the input order of the assets needs to be changed:
        #means_ and covariance matrix
        # Reorder the columns
        if list_reordering is not None:
            means_ = means_[:, list_reordering]
            list_covars = []
            for i in range(2):
                tmp_covar = covars_[i]
                list_covars.append(tmp_covar[np.ix_(list_reordering, list_reordering)])
            covars_=np.array(list_covars)
        return amount_states, amount_features, means_, covars_, transmat_, startprob_

    def initializing_backtesting_parameters(self, list_reordering=None):# -> Dict[str: np.ndarray]:

        try:
            backtesting_file = f"{self.model_choice.replace('parameter', 'backtesting')}.json"
            with importlib.resources.open_text("financial_markets_gym.envs.models", f"{backtesting_file}") as file:
                backtesting_dict = json.load(file)

            tmp_dict_ordered_returns_backtesting = {}
            for key, value in backtesting_dict.get("data_ordered_returns").items():
                tmp_array = np.array(value)
                if list_reordering is not None:
                    tmp_array = tmp_array[:, list_reordering]
                tmp_dict_ordered_returns_backtesting[key] =  tmp_array

            tmp_dict_ordered_statistics_mean = {}
            for key, value in backtesting_dict.get("data_statistics_mean").items():
                tmp_array = np.array(value)
                if list_reordering is not None:
                    tmp_array = tmp_array[:, list_reordering]
                tmp_dict_ordered_statistics_mean[key] = tmp_array

            tmp_dict_ordered_statistics_cov = {}
            for key, value in backtesting_dict.get("data_statistics_cov").items():
                tmp_array = np.array(value)
                if list_reordering is not None:
                    tmp_array = tmp_array[:, list_reordering]
                tmp_dict_ordered_statistics_cov[key] = tmp_array

        except FileNotFoundError as e:
            backtesting_file = f"{self.model_choice.replace('parameter', 'backtesting')}.json"

            print(f"WARNING: No backtesting file {backtesting_file} was found.")
            tmp_dict_ordered_returns_backtesting = None
            tmp_dict_ordered_statistics_mean = None
            tmp_dict_ordered_statistics_cov = None

        return tmp_dict_ordered_returns_backtesting, tmp_dict_ordered_statistics_mean, tmp_dict_ordered_statistics_cov

    @staticmethod
    def get_financial_kpi_data(num_dim,list_reordering=None):
        """
        Reads out additional financial kpi_data if available
        :return:
        """
        cash_asset = True
        # TODO remodelled for static use
        #data_set_name = self.model_choice
        #file_name = "model_meta_info_" + data_set_name.replace("model_parameter_", "") + ".json"

        file_name = f"model_meta_info_data_set_G_markov_states_2_{int(num_dim-1)}.json"
        try:

            with importlib.resources.open_binary("financial_markets_gym.envs.models", file_name) as file:
                dict_data = json.load(file)
                if 'additional_data' not in dict_data:
                    return {}
                else:
                    tmp_dict = {}
                    for key, value in dict_data.get("additional_data").items():
                        if list_reordering is not None:
                            value = [value[i] for i in list_reordering]

                        if cash_asset:
                            value.insert(0, 0.0) # cash has an assumed value of zero in all present KPIs
                        tmp_dict[key]=value
                    return tmp_dict
        except FileNotFoundError as e:
            logging.warning(f'The file {file_name} does not exist. It is not possible to use the predetermined mode')
            return None


    def initialize_predetermined_eval_trajectories(self, list_reordering):
        """
        This is optional and enables the env to also go into the PREDETERMINED TRAJECTORY MODE
        :return:
        """

        data_set_name = self.model_choice
        file_name = "predetermined_trajectories_" + data_set_name.replace("model_parameter_", "") + ".csv"

        try:
            # df_trajectories.to_csv(file_name, index=False)
            # df_trajectories.to_csv(f'{file_name}.gz', index=False, compression='gzip')
            with importlib.resources.open_binary("financial_markets_gym.envs.models", file_name) as file:
                # data_out = pd.read_csv(fo, compression="gzip")
                np_array = np.genfromtxt(file, delimiter=',', skip_header=1)
            number_trajectories = int(np.max(np_array[:, 0])) + 1 #+1 due to zero indexing

            if list_reordering is not None:
                list_reordering = [0, 1]+[entry+2 for entry in list_reordering] # the first two rows need to remain unchanged
                np_array = np_array[:, list_reordering]

            assert number_trajectories > 0
            logging.info(f'Predet file {file_name} was successfully processed with {number_trajectories}')
            return np_array, number_trajectories

        except FileNotFoundError as e:
            logging.warning(f'The file {file_name} does not exist. It is not possible to use the predetermined mode')
            return None, None

    def get_np_asset_return_data(self, np_array, trajectory_id, time_step):

        # Define conditions for the first and second fields
        condition_1 = np_array[:, 0] == trajectory_id  # Check the value for the first field(trajectory_id)
        condition_2 = np_array[:, 1] == time_step  # Check the value for the first field(time_steps)
        # Combine conditions using logical AND (&)
        combined_condition = condition_1 & condition_2

        # Use boolean indexing to get the rows that satisfy the conditions, we want to exclude the first two rows (trajectory_id and time_steps)
        filtered_rows = np_array[combined_condition][:, 2:]
        return filtered_rows

    def render(self):
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        # calculate trading fees:
        trading_delta_vector = action - self.current_state_portfolio_allocation # increases will be positive, decreases negative
        trading_delta_vector_buys = np.maximum(trading_delta_vector, np.zeros_like(trading_delta_vector))
        trading_delta_vector_sells = -np.minimum(trading_delta_vector, np.zeros_like(trading_delta_vector)) # we define it positive

        trading_fee = np.dot(trading_delta_vector_buys, self.ask_spread_vector) + \
                      np.dot(trading_delta_vector_sells, self.bid_spread_vector)

        # short selling fees:
        short_positions = -np.minimum(action, np.zeros_like(action)) # we define the short position positive
        short_fee = np.dot(short_positions, self.short_selling_yearly_fee_vector/12.0)

        if self.env_mode_trajectory == EnvModeTrajectory.BACKTESTING: #self.is_backtesting_mode:
            sampled_output = self.dict_ordered_returns_backtesting.get(str(self.current_time_step))
            sampled_output = np.expand_dims(sampled_output, axis=0)
            hidden_state = np.array([0]) #dummy value
        elif self.env_mode_trajectory == EnvModeTrajectory.NORMAL:
            sampled_output, hidden_state = self.model.sample(1, random_state=int(self.np_rng.integers(low=0, high=10_000_000)))
        elif self.env_mode_trajectory == EnvModeTrajectory.PRESAMPLED:
            sampled_output = self.get_np_asset_return_data(np_array=self.np_predetermined_eval_trajectories,
                                          trajectory_id=self.current_predetermined_eval_trajectory_id,
                                          time_step=self.current_time_step)
            hidden_state = np.array([0]) #dummy value
        else:
            raise ValueError(f'Unknown env_mode_trajectory')

        self.current_predicted_market_state = self.model.predict(sampled_output).item()


        if self.include_cash_asset:
            cash_return = [0.0]
            sampled_output = np.concatenate(([cash_return], sampled_output), axis=1)

        self.current_sampled_output = sampled_output.flatten()
        self.current_hidden_market_state = hidden_state.item()


        reward = np.dot(action, self.current_sampled_output) - trading_fee - short_fee

        market_movement_adjustment = np.ones_like(self.current_sampled_output)+self.current_sampled_output

        self.current_state_portfolio_allocation = (action*market_movement_adjustment)/np.sum(action*market_movement_adjustment)
        self.current_state_portfolio_wealth *= (1+reward)

        # setting the startprob for the next step
        self.model.startprob_ = self.model.transmat_[self.current_hidden_market_state]

        observation = self.create_observation(real_hidden_state=self.current_hidden_market_state,
                                              current_state_portfolio_wealth=self.current_state_portfolio_wealth,
                                              current_state_portfolio_allocation=self.current_state_portfolio_allocation,
                                              sampled_output=self.current_sampled_output)

        if self.current_time_step >= self.terminal_time_step:
            done = True
        else:
            done = False


        self.current_time_step += 1

        return observation, reward, done, {}


    def create_observation(self, real_hidden_state: int, current_state_portfolio_wealth:float,
                                              current_state_portfolio_allocation: np.ndarray,
                                              sampled_output: np.ndarray) -> np.ndarray:

        if self.include_unobservable_market_state_information_for_evaluation:
            return np.concatenate(([real_hidden_state], [current_state_portfolio_wealth],
                                   current_state_portfolio_allocation, sampled_output))
        else:
            return np.concatenate(([current_state_portfolio_wealth],
                                   current_state_portfolio_allocation, sampled_output))

    def decompose_environment_observation(self, np_observation: np.ndarray) -> Tuple:

        if self.include_unobservable_market_state_information_for_evaluation:
            real_hidden_state = int(np_observation[0])
            portfolio_wealth = np_observation[1]
            current_state_portfolio_allocation = np_observation[2:2 + self.amount_assets]
            sampled_output = np_observation[2 + self.amount_assets:]

            return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            portfolio_wealth = np_observation[0]
            current_state_portfolio_allocation = np_observation[1:1 + self.amount_assets]
            sampled_output = np_observation[1 + self.amount_assets:]

            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    @staticmethod
    def check_all_elements_included(np_observation, last_read_element_index):
        if np_observation.ndim == 2:
            assert np_observation.shape[1] == last_read_element_index
        elif np_observation.ndim == 1:
            assert np_observation.shape[0] == last_read_element_index

    @staticmethod
    def static_decompose_environment_observation_dict(np_observation: np.ndarray,
                                                      include_unobservable_market_state_information_for_evaluation: bool = False) -> Tuple:
        if np_observation.ndim == 2:
            # Batch case
            amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation[0, :].flatten(),
                                                                        include_unobservable_market_state_information_for_evaluation)
            if include_unobservable_market_state_information_for_evaluation:
                real_hidden_state = int(np_observation[:, 0])
                portfolio_wealth = np_observation[:, 1]
                current_state_portfolio_allocation = np_observation[:, 2:2 + amount_assets]
                final_index_element = 2 + amount_assets + amount_assets
                prev_observed_returns = np_observation[:, 2 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)
                tmp_dict = {"real_hidden_state": real_hidden_state,
                            "portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, prev_observed_returns
                return tmp_dict
            else:
                portfolio_wealth = np_observation[:, 0]
                current_state_portfolio_allocation = np_observation[:, 1:1 + amount_assets]
                final_index_element = 1 + amount_assets+amount_assets
                prev_observed_returns = np_observation[:, 1 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)
                tmp_dict = {"portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return portfolio_wealth, current_state_portfolio_allocation, sampled_output
                return tmp_dict
        elif np_observation.ndim == 1:
            amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation,
                                                                        include_unobservable_market_state_information_for_evaluation)

            if include_unobservable_market_state_information_for_evaluation:
                real_hidden_state = int(np_observation[0])
                portfolio_wealth = np_observation[1]
                current_state_portfolio_allocation = np_observation[2:2 + amount_assets]
                final_index_element = 2 + amount_assets + amount_assets
                prev_observed_returns = np_observation[2 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)

                tmp_dict = {"real_hidden_state": real_hidden_state,
                            "portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, prev_observed_returns
                return tmp_dict
            else:
                portfolio_wealth = np_observation[0]
                current_state_portfolio_allocation = np_observation[1:1 + amount_assets]
                final_index_element = 1 + amount_assets + amount_assets
                prev_observed_returns = np_observation[1 + amount_assets:final_index_element]
                FinancialMarketsEnv.check_all_elements_included(np_observation, final_index_element)

                tmp_dict = {"portfolio_wealth": portfolio_wealth,
                            "current_state_portfolio_allocation": current_state_portfolio_allocation,
                            "prev_observed_returns": prev_observed_returns}
                # return portfolio_wealth, current_state_portfolio_allocation, sampled_output
                return tmp_dict

    @staticmethod
    def static_decompose_environment_observation(np_observation: np.ndarray,
                                                 include_unobservable_market_state_information_for_evaluation: bool = False) -> Tuple:
        amount_assets = FinancialMarketsEnv.calculate_amount_assets(np_observation,
                                                                    include_unobservable_market_state_information_for_evaluation)

        if include_unobservable_market_state_information_for_evaluation:
            real_hidden_state = int(np_observation[0])
            portfolio_wealth = np_observation[1]
            current_state_portfolio_allocation = np_observation[2:2 + amount_assets]
            sampled_output = np_observation[2 + amount_assets:]

            return real_hidden_state, portfolio_wealth, current_state_portfolio_allocation, sampled_output
        else:
            portfolio_wealth = np_observation[0]
            current_state_portfolio_allocation = np_observation[1:1 + amount_assets]
            sampled_output = np_observation[1 + amount_assets:]

            return portfolio_wealth, current_state_portfolio_allocation, sampled_output

    @staticmethod
    def calculate_amount_assets(np_observation: np.ndarray,
                                include_unobservable_market_state_information_for_evaluation: bool = False) -> int:
        additional_fields = 0
        # portfolio_wealth
        additional_fields += 1
        if include_unobservable_market_state_information_for_evaluation:
            additional_fields += 1

        # since we have two full amount_asset measures

        return int((np_observation.shape[0] - additional_fields) / 2)

    def reset(self):

        if self.env_mode == EnvMode.EVALUATION:
            if self.reset_periodicity_eval_env is not None:
                if self.trajectories_counter % self.reset_periodicity_eval_env == 0:
                    self.set_environment_seed(self.environment_seed)

        self.current_state_portfolio_allocation = np.zeros((self.amount_assets), dtype=float)  # numpy vector
        self.current_state_portfolio_allocation[0] = 1.0 # setting initial allocation to 100% cash
        self.current_state_portfolio_wealth = 100  # None # float

        self.current_hidden_market_state = None
        self.model.startprob_ = self.initial_startprob_

        sampled_output, hidden_state = self.model.sample(1, random_state=int(self.np_rng.integers(low=0, high=10_000_000)))

        self.current_predicted_market_state = self.model.predict(sampled_output).item()

        if self.include_cash_asset:
            cash_return = [0.0]
            sampled_output = np.concatenate(([cash_return], sampled_output), axis=1)

        if self.env_mode_trajectory == EnvModeTrajectory.PRESAMPLED:
            if self.env_mode == EnvMode.EVALUATION:
                self.current_predetermined_eval_trajectory_id = int(
                    self.np_rng.integers(low=0, high=self.total_number_predetermined_eval_trajectory, size=1)
                )
            elif self.env_mode == EnvMode.TRAINING:
                self.current_predetermined_eval_trajectory_id = int(
                    self.np_rng.integers(low=0, high=self.total_number_predetermined_training_trajectory, size=1)
                )
            else:
                raise NotImplementedError

        self.current_sampled_output = sampled_output.flatten()
        self.current_hidden_market_state = hidden_state.item()

        # setting the startprob for the next step
        self.model.startprob_ = self.model.transmat_[self.current_hidden_market_state]

        # setting initial time step
        self.current_time_step = 0

        #increasing the counter
        self.trajectories_counter += 1

        return self.create_observation(real_hidden_state=self.current_hidden_market_state,
                                              current_state_portfolio_wealth=self.current_state_portfolio_wealth,
                                              current_state_portfolio_allocation=self.current_state_portfolio_allocation,
                                              sampled_output=self.current_sampled_output)
