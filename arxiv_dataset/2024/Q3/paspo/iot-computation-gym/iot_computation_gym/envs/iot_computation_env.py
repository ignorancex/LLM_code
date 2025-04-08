import numpy as np
import json
import gym
from typing import Tuple, Dict
import importlib.resources
import json

from itertools import groupby

MODELLED_COMPUTING_STEPS_PER_SECOND = 10**5 #this is different from the cycle
#DIV_FACTOR_SECONDS = 10**9
import numpy as np
import json
import gym
from typing import Tuple, Dict
import importlib.resources
import json

class EnvMode():
    NORMAL = "NORMAL"
    EVAL_BACKTESTING = "BACKTESTING"
    EVAL_PRESAMPLED = "EVAL_PRESAMPLED"

from itertools import groupby
import logging


MODELLED_COMPUTING_STEPS_PER_SECOND = 10 ** 5  # this is different from the cycle
# DIV_FACTOR_SECONDS = 10**9

# LENGTH_TIME_STEP = DIV_FACTOR_SECONDS

# one timestep is has the length of LENGTH_TIME_STEP/(DIV_FACTOR_SECONDS) seconds

STATE_TIME_REP = "milliseconds"
STATE_CYCLE_REP = "megacycles"  # 1_000_000


class Task():
    static_task_id = 0 #static variable

    def __init__(self, task_bits, task_cycles, task_total_delta_time_max, task_generated_at_time):
        self.task_id = Task.static_task_id
        Task.static_task_id += 1
        self.task_bits = task_bits
        self.task_cycles = task_cycles
        self.task_total_delta_time_max = task_total_delta_time_max

        self.task_generated_at_time = task_generated_at_time

    def __str__(self):
        return f"({self.task_bits}, {self.task_cycles}, {self.task_total_delta_time_max}, {self.task_id}, {self.task_generated_at_time})"

    def __repr__(self):
        return f"({self.task_bits}, {self.task_cycles}, {self.task_total_delta_time_max}, {self.task_id}, {self.task_generated_at_time})"

class SubTask():
    def __init__(self, sub_task_bits, sub_task_cycles, original_task_total_delta_time_max, original_task_id):
        self.original_task_id = original_task_id
        self.original_task_total_delta_time_max = original_task_total_delta_time_max

        self.sub_task_bits = sub_task_bits
        self.sub_task_cycles = sub_task_cycles

        self.transmission_time = None
        self.server_specific_computation_time = None
        self.computation_time_begin = None
        self.computation_time_completed = None

    def set_transmission_time(self, transmission_time: float):
        self.transmission_time = transmission_time

    def set_computation_time(self, computation_time):
        self.server_specific_computation_time = computation_time

    def set_computation_time_begin(self, computation_time_begin):
        self.computation_time_begin = computation_time_begin

    def set_computation_time_completed(self, computation_time_completed):
        self.computation_time_completed = computation_time_completed

    def get_computation_time_completed(self):
        return self.computation_time_completed

    def __str__(self):
        return f"({self.sub_task_bits}, {self.sub_task_cycles}, " \
               f"{self.original_task_total_delta_time_max}, {self.original_task_id}, " \
               f"{self.server_specific_computation_time}, {self.computation_time_completed})"

    def __repr__(self):

        return f"({self.sub_task_bits}, {self.sub_task_cycles}, " \
               f"{self.original_task_total_delta_time_max}, {self.original_task_id}, " \
               f"{self.server_specific_computation_time}, {self.computation_time_completed})"

class IOTComputationEnv(gym.Env):

    logger=None

    def __init__(self, environment_parameters_set_name, seed=1, **kwargs):

        self.environment_parameters_set_name = environment_parameters_set_name
        self.initialize_environment_parameters()

        self.logger = IOTComputationEnv.get_static_logger()
        self._rng = np.random.default_rng(seed)

        self.server_config = self.environment_parameters.get("server_config")
        self.user_config = self.environment_parameters.get("user_config")

        self.list_servers = []
        for server_param in self.server_config.values():
            self.list_servers.append(Server(**server_param, rng=self._rng))

        self.current_time = 0.0  # in sec

        #register user task schedule in event_queue
        self.list_upcoming_task_queue = []
        self.fill_list_upcoming_task_queue()

        #env_mode
        self.env_mode = EnvMode.NORMAL

        # Unrestricted action spaces
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.list_servers),), dtype=np.float32)

        #observation_size = len(self.list_servers)
        self.number_tasks_to_show_queue = 3

        dict_observation_spaces = {
            "servers_queue_waiting_time": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.list_servers),), dtype=np.float32),
            "task_queue_list_cycles": gym.spaces.Box(low=0, high=np.inf, shape=(self.number_tasks_to_show_queue,), dtype=np.float32),
            "task_queue_list_bits": gym.spaces.Box(low=0, high=np.inf, shape=(self.number_tasks_to_show_queue,), dtype=np.float32),
            "task_queue_total_delta_time_max": gym.spaces.Box(low=0, high=np.inf, shape=(self.number_tasks_to_show_queue,), dtype=np.float32)
        }

        self.observation_space = gym.spaces.Dict(dict_observation_spaces)

    @staticmethod
    def get_static_logger():
        if IOTComputationEnv.logger is None:
            IOTComputationEnv.logger = logging.getLogger("IOTComputationEnvLogger")
        return IOTComputationEnv.logger
        
    def initialize_environment_parameters(self):
        thousand_seperator = ","
        decimal_seperator = "."
        # https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
        with importlib.resources.open_text("iot_computation_gym.envs.parameters", f"{self.environment_parameters_set_name}.json") as file:
            parameter_dict = json.load(file)

        for outer_key, outer_value in parameter_dict.get("server_config").items():
            for inner_key, inner_value in outer_value.items():
                if decimal_seperator in inner_value:
                    outer_value[inner_key] = float(inner_value)
                else:
                    outer_value[inner_key] = int(inner_value.replace(thousand_seperator, ""))


        for outer_key, outer_value in parameter_dict.get("user_config").items():
            for inner_key, inner_value in outer_value.items():
                if decimal_seperator in inner_value:
                    outer_value[inner_key] = float(inner_value)
                else:
                    outer_value[inner_key] = int(inner_value.replace(thousand_seperator, ""))

        self.environment_parameters = parameter_dict

    def calculate_acceptable_time_delta_max(self, task_bit_size, task_required_cycles, acceptable_leeway_factor):
        # Step 1 calculate fastest possible time if no queue would be there
        list_individual_server_bit_transfer_per_second = [server.bit_transfer_per_second for server in self.list_servers]
        list_individual_server_cycle_capacity = [server.server_max_frequency for server in self.list_servers]

        #under the assumption that all transfer values are equal
        #assert all_equal(self.list_individual_server_bit_transfer_per_second)
        minimum_transmission_time = task_bit_size/list_individual_server_bit_transfer_per_second[0] #assuming we transport all data to each server
        minimum_computation_time = task_required_cycles/sum(list_individual_server_cycle_capacity) #if all servers are working on the same problem, we can add up their capacity, i.e. their slope

        minimum_required_time = minimum_transmission_time+minimum_computation_time

        return acceptable_leeway_factor*minimum_required_time

    def fill_list_upcoming_task_queue(self):

        tmp_task_list = []
        for user_count, user_param in enumerate(self.user_config.values()):
            #self.list_users.append(User(**user_param))
            #print(f"USER {user_count}")
            #print(f"{user_param}")
            average_events_in_interval = user_param.get("average_number_task_spawning_per_interval")
            interval_length_in_sec = user_param.get("interval_length_in_sec")
            task_computing_size_cycles = int(self._rng.integers(
                low=user_param.get("task_computing_size_lower_limit_cycles"),
                high=user_param.get("task_computing_size_upper_limit_cycles")+1))
            task_data_size_bits = int(self._rng.integers(
                low=user_param.get("task_data_size_lower_limit_bits"),
                high=user_param.get("task_data_size_upper_limit_bits")+1))
            task_computing_time_delta_max = self.calculate_acceptable_time_delta_max(
                task_bit_size=task_data_size_bits,
                task_required_cycles=task_computing_size_cycles,
                acceptable_leeway_factor=3.0)

            cumulative_normalized_time_stamp = 0
            average_time_between_events = 1 / average_events_in_interval
            while cumulative_normalized_time_stamp < 1:
                tmp_interval_length = float(self._rng.exponential(scale=average_time_between_events, size=1))
                cumulative_normalized_time_stamp += tmp_interval_length
                if cumulative_normalized_time_stamp < 1:
                    task_generated_at_time = cumulative_normalized_time_stamp*interval_length_in_sec
                    tmp_task = Task(task_bits=task_data_size_bits, task_cycles=task_computing_size_cycles,
                                    task_total_delta_time_max=task_computing_time_delta_max,
                                    task_generated_at_time=task_generated_at_time)
                    tmp_task_list.append(tmp_task)

        #ensuring correct order
        tmp_task_list.sort(key=lambda x: x.task_generated_at_time)
        self.list_upcoming_task_queue = tmp_task_list

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        #process events until we git the event "task_generation"
        assert len(self.list_upcoming_task_queue)>0
        fifo_task = self.list_upcoming_task_queue.pop(0)
        #self.logger.info(action)
        #print(fifo_task)

        #progressing time
        self.set_current_time(fifo_task.task_generated_at_time)
        for server in self.list_servers:
            server.set_current_time(fifo_task.task_generated_at_time)
            server.process_new_current_time()

        latest_time_sub_task_completed = self.split_and_assign_task_to_servers(np_action=action,
                                                       task_to_split=fifo_task)

        reward = self.calculate_reward(
            maximum_allowed_time= fifo_task.task_generated_at_time + fifo_task.task_total_delta_time_max,
            latest_time_task_completed=latest_time_sub_task_completed)

        #are there tasks left?
        if len(self.list_upcoming_task_queue)==0:
            done=True
        else:
            done=False

        observation=self.create_observation()

        return observation, reward, done, {}

    def set_env_mode(self, env_mode, seed=None):
        """
        This was later added to allow for multiple evn modes
        :return:
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
            for server in self.list_servers:
                server.set_rng(self._rng)

            
        self.env_mode = env_mode

    def calculate_reward(self, maximum_allowed_time, latest_time_task_completed):
        if maximum_allowed_time>latest_time_task_completed:
            return 1.0
        else:
            return 0.0

    def split_and_assign_task_to_servers(self, np_action, task_to_split: Task)-> float:
        """
        This will return the lastest point in time a subtask is completed
        :param np_action:
        :param task_to_split:
        :return:
        """

        list_action = np_action.tolist()

        #round properly
        number_digits_rounding = 6
        list_action = [round(entry, number_digits_rounding) for entry in list_action]

        assert len(list_action) == len(self.list_servers)

        list_time_sub_task_completed_per_server = []

        for allocation_server_pair in (zip(list_action, self.list_servers)):
            sub_task_action_weight = allocation_server_pair[0]
            corresponding_server = allocation_server_pair[1]

            #splitting
            sub_task_bits = int(task_to_split.task_bits*sub_task_action_weight) #simplification by parsing into int
            sub_task_cycles = int(task_to_split.task_cycles*sub_task_action_weight) #simplification by parsing into int

            tmp_sub_task = SubTask(sub_task_bits=sub_task_bits, sub_task_cycles=sub_task_cycles,
                                   original_task_id=task_to_split.task_id,
                                   original_task_total_delta_time_max=task_to_split.task_total_delta_time_max)

            time_sub_task_completed_server = corresponding_server.add_sub_task_to_server_queue(tmp_sub_task)
            list_time_sub_task_completed_per_server.append(time_sub_task_completed_server)

        latest_time_task_completed = max(list_time_sub_task_completed_per_server)
        return latest_time_task_completed

    def reset(self):
        Task.static_task_id = 0
        Server.static_server_id = 0

        for server in self.list_servers:
            server.reset_server()

        self.current_time = 0.0
        #register user task schedule in event_queue
        self.list_upcoming_task_queue = []
        self.fill_list_upcoming_task_queue()

        return self.create_observation()

    def set_current_time(self, current_time):
        self.current_time = current_time

    def create_observation(self):
        list_waiting_time_obs = []
        for server in self.list_servers:
            list_waiting_time_obs.append(server.current_waiting_time_until_computation(str_time=STATE_TIME_REP))

        list_cycles = []
        list_bits = []
        list_total_delta_time_max = []
        for i in range(self.number_tasks_to_show_queue):
            if i<len(self.list_upcoming_task_queue):
                task = self.list_upcoming_task_queue[i]
                list_cycles.append(task.task_cycles)
                list_bits.append(task.task_bits)
                list_total_delta_time_max.append(task.task_total_delta_time_max)
            else:
                list_cycles.append(0)
                list_bits.append(0)
                list_total_delta_time_max.append(0)

        # Create a numpy array filled with zeros of size 3
        np_cycles = np.array(list_cycles, dtype=np.float32)
        np_bits = np.array(list_bits, dtype=np.float32)
        np_total_delta_time_max = np.array(list_total_delta_time_max, dtype=np.float32)

        observation = {
            "servers_queue_waiting_time": np.array(list_waiting_time_obs, dtype=np.float32),
            "task_queue_list_cycles": np_cycles,
            "task_queue_list_bits": np_bits,
            "task_queue_total_delta_time_max": np_total_delta_time_max
        }

        return observation # The correct output is a one-dimensional array

class Server():
    static_server_id = 0  # static variable

    BIT_TRANSFER_PER_SECOND = 1_000_000_000

    def __init__(self, server_max_frequency_lower_limit_hz, server_max_frequency_upper_limit_hz, rng: np.random.Generator):
        self._rng = rng
        # hz = cycles/seconds

        # ~ 5.000.000.000
        # average cycle 90.000.000 ~ 55 tasks per second per server

        self.server_max_frequency_lower_limit_hz = server_max_frequency_lower_limit_hz  # 2 * 10 ** 9
        self.server_max_frequency_upper_limit_hz = server_max_frequency_upper_limit_hz  # 8 * 10 ** 9

        self.server_max_frequency = self.sample_server_max_frequency()

        self.bit_transfer_per_second = Server.BIT_TRANSFER_PER_SECOND

        self.current_time = 0
        self.current_active_subtask_queue = []  # fifo queue

        self.server_id = f'server_{Server.static_server_id}'
        Server.static_server_id += 1

    def set_rng(self, rng):
        self._rng = rng
    
    def reset_server(self):
        self.current_time = 0
        self.current_active_subtask_queue = []  # fifo queue

    def sample_server_max_frequency(self):

        if self.server_max_frequency_lower_limit_hz == self.server_max_frequency_upper_limit_hz:
            server_max_frequency = self.server_max_frequency_lower_limit_hz
        else:
            server_max_frequency = self._rng.integers(low=self.server_max_frequency_lower_limit_hz,
                                                high=self.server_max_frequency_upper_limit_hz)
        return server_max_frequency


    def set_current_time(self, current_time):
        self.current_time = current_time

    def add_sub_task_to_server_queue(self, sub_task) -> float:
        """
        Returns determinsticly calculated time of completed task
        :param subtask:
        :return:
        """

        transmission_time = self.calculate_transmission_time(sub_task)
        sub_task.set_transmission_time(transmission_time=transmission_time)

        computation_time = self.calculate_computation_time(required_cycles=sub_task.sub_task_cycles)
        sub_task.set_computation_time(computation_time=computation_time)

        time_stamp_computation_begin = self.calculate_time_stamp_for_calculation_begin(transmission_time=transmission_time)
        sub_task.set_computation_time_begin(time_stamp_computation_begin)
        time_stamp_computation_completed = time_stamp_computation_begin + computation_time
        sub_task.set_computation_time_completed(time_stamp_computation_completed)

        self.current_active_subtask_queue.append(sub_task)

        return time_stamp_computation_completed


    def calculate_transmission_time(self, subtask):
        return subtask.sub_task_bits/self.bit_transfer_per_second


    def calculate_computation_time(self, required_cycles, available_cycles=None):
        if available_cycles is None:
            available_cycles=self.server_max_frequency_upper_limit_hz
        return required_cycles/available_cycles

    def calculate_time_stamp_for_calculation_begin(self, transmission_time):

        if len(self.current_active_subtask_queue) == 0:  # empty queue
            return self.current_time+transmission_time
        else:
            index_last_element = len(self.current_active_subtask_queue)-1
            delta_time_until_last_computation_is_completed = \
                self.current_active_subtask_queue[index_last_element].get_computation_time_completed()-self.current_time

            assert delta_time_until_last_computation_is_completed >= 0 #should always be positive

            if transmission_time<delta_time_until_last_computation_is_completed:
                return self.current_active_subtask_queue[index_last_element].get_computation_time_completed()
            else:
                return self.current_time+transmission_time

    def process_new_current_time(self):

        while len(self.current_active_subtask_queue)>0 and \
                self.current_active_subtask_queue[0].get_computation_time_completed()<self.current_time:
            _ = self.current_active_subtask_queue.pop(0)

    def current_waiting_time_until_computation(self, str_time="milliseconds"):
        """
        Without the consideration of transmission time
        :param str_time:
        :return:
        """
        if str_time=="milliseconds":
            time_factor = 1_000
        else:
            raise ValueError

        if len(self.current_active_subtask_queue) == 0:
            return 0
        else:
            index_last_element = len(self.current_active_subtask_queue) - 1
            return (self.current_active_subtask_queue[index_last_element].get_computation_time_completed())*time_factor
