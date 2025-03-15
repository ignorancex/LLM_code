"""
Environments for training vehicles to reduce capacity drops in a bottleneck.

This environment was used in:

E. Vinitsky, K. Parvate, A. Kreidieh, C. Wu, Z. Hu, A. Bayen, "Lagrangian
Control through Deep-RL: Applications to Bottleneck Decongestion," IEEE
Intelligent Transportation Systems Conference (ITSC), 2018.
"""

from copy import deepcopy

import numpy as np
from gym.spaces.box import Box

from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from flow.core.params import InFlows, NetParams
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core import rewards
from flow.envs.base_env import Env

MAX_LANES = 4  # base number of largest number of lanes in the network
EDGE_LIST = ["1", "2", "3", "4", "5"]  # Edge 1 is before the toll booth
EDGE_BEFORE_TOLL = "1"  # Specifies which edge number is before toll booth
TB_TL_ID = "2"
EDGE_AFTER_TOLL = "2"  # Specifies which edge number is after toll booth
NUM_TOLL_LANES = MAX_LANES

TOLL_BOOTH_AREA = 10  # how far into the edge lane changing is disabled
RED_LIGHT_DIST = 50  # how close for the ramp meter to start going off

EDGE_BEFORE_RAMP_METER = "2"  # Specifies which edge is before ramp meter
EDGE_AFTER_RAMP_METER = "3"  # Specifies which edge is after ramp meter
NUM_RAMP_METERS = MAX_LANES

RAMP_METER_AREA = 80  # Area occupied by ramp meter

MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK = 3  # Average waiting time at fast track
MEAN_NUM_SECONDS_WAIT_AT_TOLL = 15  # Average waiting time at toll

BOTTLE_NECK_LEN = 280  # Length of bottleneck
NUM_VEHICLE_NORM = 20

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}

# Keys for RL experiments
ADDITIONAL_RL_ENV_PARAMS = {
    # velocity to use in reward functions
    "target_velocity": 30,
    # if an RL vehicle exits, place it back at the front
    "add_rl_if_exit": True,
}

# Keys for VSL style experiments
ADDITIONAL_VSL_ENV_PARAMS = {
    # number of controlled regions for velocity bottleneck controller
    "controlled_segments": [("1", 1, True), ("2", 1, True), ("3", 1, True),
                            ("4", 1, True), ("5", 1, True)],
    # whether lanes in a segment have the same action or not
    "symmetric": False,
    # which edges are observed
    "observed_segments": [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1)],
    # whether the inflow should be reset on each rollout
    "reset_inflow": False,
    # the range of inflows to reset on
    "inflow_range": [1000, 2000],
    # whether to subtract a penalty when vehicles congest
    "congest_penalty": True,
    # initial inflow
    "start_inflow": 1900,
    # the lane changing mode. 1621 for LC on for humans, 0 for it off.
    "lc_mode": 1621,
    # # how many seconds the outflow reward should sample over
    "num_sample_seconds": 20,
    # whether the reward function should be over speed
    "speed_reward": False,
    # whether the reward should be high when the exiting vehicles come from a uniform distribution over entering lanes
    "fair_reward": False,
    # how many seconds you should look back in tracking the history of which lane exiting vehicles are from
    "exit_history_seconds": 60,
}

START_RECORD_TIME = 0.0  # Time to start recording
PERIOD = 10.0


class BottleneckEnv(Env):
    """Abstract bottleneck environment.

    This environment is used as a simplified representation of the toll booth
    portion of the bay bridge. Contains ramp meters, and a toll both.

    Additional
        Vehicles are rerouted to the start of their original routes once
        they reach the end of the network in order to ensure a constant
        number of vehicles.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        """Initialize the BottleneckEnv class."""
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, scenario, simulator)
        env_add_params = self.env_params.additional_params
        # tells how scaled the number of lanes are
        self.scaling = scenario.net_params.additional_params.get("scaling", 1)
        self.edge_dict = dict()
        self.cars_waiting_for_toll = dict()
        self.cars_before_ramp = dict()
        self.toll_wait_time = np.abs(
            np.random.normal(MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                             4 / self.sim_step, NUM_TOLL_LANES * self.scaling))
        self.fast_track_lanes = range(
            int(np.ceil(1.5 * self.scaling)), int(np.ceil(2.6 * self.scaling)))

        self.tl_state = ""
        self.next_period = START_RECORD_TIME / self.sim_step

        # values for the ALINEA ramp meter algorithm
        self.n_crit = env_add_params.get("n_crit")
        self.q_max = env_add_params.get("q_max")
        self.q_min = env_add_params.get("q_min")
        self.feedback_coeff = env_add_params.get("feedback_coeff")
        self.q = self.q_max  # ramp meter feedback controller
        self.feedback_update_time = env_add_params.get("feedback_update", 30)
        self.feedback_timer = 0.0
        self.cycle_time = 8
        self.prev_cycle_time = self.cycle_time
        # self.ramp_state = np.linspace(0,
        #                               self.cycle_offset * self.scaling * MAX_LANES,
        #                               self.scaling * MAX_LANES)
        self.green_time = 4
        self.ramp_state = np.array([0, -self.green_time] *
                                   (self.scaling * MAX_LANES // 2)).astype(np.float64)

        self.smoothed_num = np.zeros(10)  # averaged number of vehs in '4'
        self.outflow_index = 0
        self.waiting_queue = []

    def additional_command(self):
        """Build a dict with vehicle information.

        The dict contains the list of vehicles and their position for each edge
        and for each edge within the edge.
        """
        super().additional_command()

        # build a dict containing the list of vehicles and their position for
        # each edge and for each lane within the edge
        empty_edge = [[] for _ in range(MAX_LANES * self.scaling)]

        self.edge_dict = {k: deepcopy(empty_edge) for k in EDGE_LIST}
        for veh_id in self.k.vehicle.get_ids():
            try:
                edge = self.k.vehicle.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict[edge] = deepcopy(empty_edge)
                lane = self.k.vehicle.get_lane(veh_id)  # integer
                pos = self.k.vehicle.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except Exception:
                pass

        if not self.env_params.additional_params['disable_tb']:
            self.apply_toll_bridge_control()
        if not self.env_params.additional_params['disable_ramp_metering']:
            self.ramp_meter_lane_change_control()
            self.alinea()

        # compute the outflow
        veh_ids = self.k.vehicle.get_ids_by_edge('4')
        self.smoothed_num[self.outflow_index] = len(veh_ids)
        self.outflow_index = \
            (self.outflow_index + 1) % self.smoothed_num.shape[0]

        if self.time_counter > self.next_period:
            self.next_period += PERIOD / self.sim_step

    def ramp_meter_lane_change_control(self):
        """Control the lane changing behavior.

        Specify/Toggle the lane changing behavior of the vehicles
        depending on factors like whether or not they are before
        the toll.
        """
        cars_that_have_left = []
        for veh_id in self.cars_before_ramp:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_RAMP_METER:
                color = self.cars_before_ramp[veh_id]['color']
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = self.cars_before_ramp[veh_id][
                        'lane_change_mode']
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            del self.cars_before_ramp[veh_id]

        for lane in range(NUM_RAMP_METERS * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_RAMP_METER][lane]

            for veh_id, pos in cars_in_lane:
                if pos > RAMP_METER_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        if self.simulator == 'traci':
                            # Disable lane changes inside Toll Area
                            lane_change_mode = \
                                self.k.kernel_api.vehicle.getLaneChangeMode(
                                    veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (0, 255, 255))
                        self.cars_before_ramp[veh_id] = {
                            'lane_change_mode': lane_change_mode,
                            'color': color
                        }

    def alinea(self):
        """Utilize the ALINEA algorithm for toll booth metering control.

        This acts as an implementation of the ramp metering control algorithm
        from the article:

        Spiliopoulou, Anastasia D., Ioannis Papamichail, and Markos
        Papageorgiou. "Toll plaza merging traffic control for throughput
        maximization." Journal of Transportation Engineering 136.1 (2009):
        67-76.
        """
        self.ramp_state += self.sim_step
        self.feedback_timer += self.sim_step
        if self.feedback_timer > self.feedback_update_time:
            self.feedback_timer = 0
            # now implement the integral controller update
            # find all the vehicles in an edge
            q_update = self.feedback_coeff * (
                self.n_crit - np.average(self.smoothed_num))
            self.q = np.clip(
                self.q + q_update, a_min=self.q_min, a_max=self.q_max)
            # convert q to cycle time, we keep track of the previous cycle time to let the cycle coplete
            self.prev_cycle_time = self.cycle_time
            self.cycle_time = 7200 * self.scaling * MAX_LANES / self.q

        # now apply the cycle time to compute if the light should be green or not
        if np.all(self.ramp_state > self.prev_cycle_time):
            self.prev_cycle_time = self.cycle_time
            self.ramp_state = np.array([0, -self.green_time] *
                                       (self.scaling * MAX_LANES // 2)).astype(np.float64)

        # step through, if the value of tl_state is below self.green_time
        # we should be green, otherwise we should be red
        time_mask = (self.ramp_state >= 0)
        tl_mask = (self.ramp_state <= self.green_time)
        tl_mask = tl_mask & time_mask
        colors = ['G' if val else 'r' for val in tl_mask]
        self.k.traffic_light.set_state('3', ''.join(colors))

    def apply_toll_bridge_control(self):
        """Apply control to the toll bridge."""
        cars_that_have_left = []
        for veh_id in self.cars_waiting_for_toll:
            if self.k.vehicle.get_edge(veh_id) == EDGE_AFTER_TOLL:
                lane = self.k.vehicle.get_lane(veh_id)
                color = self.cars_waiting_for_toll[veh_id]["color"]
                self.k.vehicle.set_color(veh_id, color)
                if self.simulator == 'traci':
                    lane_change_mode = \
                        self.cars_waiting_for_toll[veh_id]["lane_change_mode"]
                    self.k.kernel_api.vehicle.setLaneChangeMode(
                        veh_id, lane_change_mode)
                if lane not in self.fast_track_lanes:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_TOLL / self.sim_step,
                            1 / self.sim_step))
                else:
                    self.toll_wait_time[lane] = max(
                        0,
                        np.random.normal(
                            MEAN_NUM_SECONDS_WAIT_AT_FAST_TRACK /
                            self.sim_step, 1 / self.sim_step))

                cars_that_have_left.append(veh_id)

        for veh_id in cars_that_have_left:
            del self.cars_waiting_for_toll[veh_id]

        traffic_light_states = ["G"] * NUM_TOLL_LANES * self.scaling

        for lane in range(NUM_TOLL_LANES * self.scaling):
            cars_in_lane = self.edge_dict[EDGE_BEFORE_TOLL][lane]

            for veh_id, pos in cars_in_lane:
                if pos > TOLL_BOOTH_AREA:
                    if veh_id not in self.cars_waiting_for_toll:
                        # Disable lane changes inside Toll Area
                        if self.simulator == 'traci':
                            lane_change_mode = self.k.kernel_api.vehicle.\
                                getLaneChangeMode(veh_id)
                            self.k.kernel_api.vehicle.setLaneChangeMode(
                                veh_id, 512)
                        else:
                            lane_change_mode = None
                        color = self.k.vehicle.get_color(veh_id)
                        self.k.vehicle.set_color(veh_id, (255, 0, 255))
                        self.cars_waiting_for_toll[veh_id] = \
                            {'lane_change_mode': lane_change_mode,
                             'color': color}
                    else:
                        if pos > 50:
                            if self.toll_wait_time[lane] < 0:
                                traffic_light_states[lane] = "G"
                            else:
                                traffic_light_states[lane] = "r"
                                self.toll_wait_time[lane] -= 1

        new_tl_state = "".join(traffic_light_states)

        if new_tl_state != self.tl_state:
            self.tl_state = new_tl_state
            self.k.traffic_light.set_state(
                node_id=TB_TL_ID, state=new_tl_state)

    def get_bottleneck_density(self, lanes=None):
        """Return the density of specified lanes.

        If no lanes are specified, this function calculates the
        density of all vehicles on all lanes of the bottleneck edges.
        """
        bottleneck_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        if lanes:
            veh_ids = [
                veh_id for veh_id in bottleneck_ids
                if str(self.k.vehicle.get_edge(veh_id)) + "_" +
                str(self.k.vehicle.get_lane(veh_id)) in lanes
            ]
        else:
            veh_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

    # Dummy action and observation spaces
    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1, ),
            dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        reward = self.k.vehicle.get_outflow_rate(10 * self.sim_step) / \
            (2000.0 * self.scaling)
        return reward

    def get_state(self):
        """See class definition."""
        return np.asarray([1])

    def reset(self):
        self.waiting_queue = []
        self.q = self.q_max  # ramp meter feedback controller
        self.feedback_timer = 0.0
        self.cycle_time = 8
        self.prev_cycle_time = self.cycle_time
        self.green_time = 4
        self.ramp_state = np.array([0, -self.green_time] *
                                   (self.scaling * MAX_LANES // 2)).astype(np.float64)

        self.smoothed_num = np.zeros(10)  # averaged number of vehs in '4'
        return super().reset()


class BottleNeckAccelEnv(BottleneckEnv):
    """BottleNeckAccelEnv.

    Environment used to train vehicles to effectively pass through a
    bottleneck.

    States
        An observation is the edge position, speed, lane, and edge number of
        the AV, the distance to and velocity of the vehicles
        in front and behind the AV for all lanes. Additionally, we pass the
        density and average velocity of all edges. Finally, we pad with
        zeros in case an AV has exited the system.
        Note: the vehicles are arranged in an initial order, so we pad
        the missing vehicle at its normal position in the order

    Actions
        The action space consist of a list in which the first half
        is accelerations and the second half is a direction for lane
        changing that we round

    Rewards
        The reward is the two-norm of the difference between the speed of
        all vehicles in the network and some desired speed. To this we add
        a positive reward for moving the vehicles forward, and a penalty to
        vehicles that lane changing too frequently.

    Termination
        A rollout is terminated once the time horizon is reached.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        """Initialize BottleNeckAccelEnv."""
        for p in ADDITIONAL_RL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        super().__init__(env_params, sim_params, scenario, simulator)
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self.rl_id_list = deepcopy(self.initial_vehicles.get_rl_ids())
        self.max_speed = self.k.scenario.max_speed()

    @property
    def observation_space(self):
        """See class definition."""
        num_edges = len(self.k.scenario.get_edge_list())
        num_rl_veh = self.num_rl
        num_obs = 2 * num_edges + 4 * MAX_LANES * self.scaling \
            * num_rl_veh + 4 * num_rl_veh

        return Box(low=-3.0, high=3.0, shape=(num_obs, ), dtype=np.float32)

    def get_state(self):
        """See class definition."""
        headway_scale = 1000

        rl_ids = self.k.vehicle.get_rl_ids()

        # rl vehicle data (absolute position, speed, and lane index)
        rl_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list.index(veh_id)
            if rl_id_num != id_counter:
                rl_obs = np.concatenate(
                    (rl_obs, np.zeros(4 * (rl_id_num - id_counter))))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1

            # get the edge and convert it to a number
            edge_num = self.k.vehicle.get_edge(veh_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                edge_num = int(edge_num) / 6
            rl_obs = np.concatenate((rl_obs, [
                self.k.vehicle.get_x_by_id(veh_id) / 1000,
                (self.k.vehicle.get_speed(veh_id) / self.max_speed),
                (self.k.vehicle.get_lane(veh_id) / MAX_LANES), edge_num
            ]))

        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(rl_obs.shape[0] / 4)
        if diff > 0:
            rl_obs = np.concatenate((rl_obs, np.zeros(4 * diff)))

        # relative vehicles data (lane headways, tailways, vel_ahead, and
        # vel_behind)
        relative_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list.index(veh_id)
            if rl_id_num != id_counter:
                pad_mat = np.zeros(
                    4 * MAX_LANES * self.scaling * (rl_id_num - id_counter))
                relative_obs = np.concatenate((relative_obs, pad_mat))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1
            num_lanes = MAX_LANES * self.scaling
            headway = np.asarray([1000] * num_lanes) / headway_scale
            tailway = np.asarray([1000] * num_lanes) / headway_scale
            vel_in_front = np.asarray([0] * num_lanes) / self.max_speed
            vel_behind = np.asarray([0] * num_lanes) / self.max_speed

            lane_leaders = self.k.vehicle.get_lane_leaders(veh_id)
            lane_followers = self.k.vehicle.get_lane_followers(veh_id)
            lane_headways = self.k.vehicle.get_lane_headways(veh_id)
            lane_tailways = self.k.vehicle.get_lane_tailways(veh_id)
            headway[0:len(lane_headways)] = (
                np.asarray(lane_headways) / headway_scale)
            tailway[0:len(lane_tailways)] = (
                np.asarray(lane_tailways) / headway_scale)
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (
                        self.k.vehicle.get_speed(lane_leader) / self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_followers != '':
                    vel_behind[i] = (self.k.vehicle.get_speed(lane_follower) /
                                     self.max_speed)

            relative_obs = np.concatenate((relative_obs, headway, tailway,
                                           vel_in_front, vel_behind))

        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(relative_obs.shape[0] / (4 * MAX_LANES))
        if diff > 0:
            relative_obs = np.concatenate((relative_obs,
                                           np.zeros(4 * MAX_LANES * diff)))

        # per edge data (average speed, density
        edge_obs = []
        for edge in self.k.scenario.get_edge_list():
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(veh_ids) > 0:
                avg_speed = (sum(self.k.vehicle.get_speed(veh_ids)) /
                             len(veh_ids)) / self.max_speed
                density = len(veh_ids) / self.k.scenario.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        return np.concatenate((rl_obs, relative_obs, edge_obs))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        num_rl = self.k.vehicle.num_rl_vehicles
        lane_change_acts = np.abs(np.round(rl_actions[1::2])[:num_rl])
        return (rewards.desired_velocity(self) + rewards.rl_forward_progress(
            self, gain=0.1) - rewards.boolean_action_penalty(
                lane_change_acts, gain=1.0))

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.num_rl
        ub = [max_accel, 1] * self.num_rl

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    def _apply_rl_actions(self, actions):
        """
        See parent class.

        Takes a tuple and applies a lane change or acceleration. if a lane
        change is applied, don't issue any commands
        for the duration of the lane change and return negative rewards
        for actions during that lane change. if a lane change isn't applied,
        and sufficient time has passed, issue an acceleration like normal.
        """
        num_rl = self.k.vehicle.num_rl_vehicles
        acceleration = actions[::2][:num_rl]
        direction = np.round(actions[1::2])[:num_rl]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = sorted(self.k.vehicle.get_rl_ids(),
                               key=self.k.vehicle.get_x_by_id)

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = [
            self.time_counter <= self.env_params.additional_params[
                'lane_change_duration'] + self.k.vehicle.get_last_lc(veh_id)
            for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.

        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.
        """
        super().additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge='1',
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass


class DesiredVelocityEnv(BottleneckEnv):
    """DesiredVelocityEnv.

    Environment used to train vehicles to effectively pass through a
    bottleneck by specifying the velocity that RL vehicles should attempt to
    travel in certain regions of space.

    States
        An observation is the number of vehicles in each lane in each
        segment

    Actions
        The action space consist of a list in which each element
        corresponds to the desired speed that RL vehicles should travel in
        that region of space

    Rewards
        The reward is the outflow of the bottleneck plus a reward
        for RL vehicles making forward progress
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        """Initialize DesiredVelocityEnv."""
        super().__init__(env_params, sim_params, scenario, simulator)

        # default (edge, segment, controlled) status
        add_env_params = self.env_params.additional_params
        default = [(str(i), 1, True) for i in range(1, 6)]
        self.segments = add_env_params.get("controlled_segments", default)

        # number of segments for each edge
        self.num_segments = [segment[1] for segment in self.segments]

        # whether an edge is controlled
        self.is_controlled = [segment[2] for segment in self.segments]

        self.num_controlled_segments = [
            segment[1] for segment in self.segments if segment[2]
        ]

        # sum of segments
        self.total_segments = int(
            np.sum([segment[1] for segment in self.segments]))
        # sum of controlled segments
        segment_list = [segment[1] for segment in self.segments if segment[2]]
        self.total_controlled_segments = int(np.sum(segment_list))

        # list of controlled edges for comparison
        self.controlled_edges = [
            segment[0] for segment in self.segments if segment[2]
        ]

        additional_params = env_params.additional_params

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.slices = {}
        for edge, num_segments, _ in self.segments:
            edge_length = self.k.scenario.edge_length(edge)
            self.slices[edge] = np.linspace(0, edge_length, num_segments + 1)

        # get info for observed segments
        self.obs_segments = additional_params.get("observed_segments", [])

        # number of segments for each edge
        self.num_obs_segments = [segment[1] for segment in self.obs_segments]

        # for convenience, construct the relevant positions defining
        # segments within edges
        # self.slices is a dictionary mapping
        # edge (str) -> segment start location (list of int)
        self.obs_slices = {}
        for edge, num_segments in self.obs_segments:
            edge_length = self.k.scenario.edge_length(edge)
            self.obs_slices[edge] = np.linspace(0, edge_length,
                                                num_segments + 1)

        # self.symmetric is True if all lanes in a segment
        # have same action, else False
        self.symmetric = additional_params.get("symmetric")

        # action index tells us, given an edge and a lane,the offset into
        # rl_actions that we should take.
        self.action_index = [0]
        for i, (edge, segment, controlled) in enumerate(self.segments[:-1]):
            if self.symmetric:
                self.action_index += [
                    self.action_index[i] + segment * controlled
                ]
            else:
                num_lanes = self.k.scenario.num_lanes(edge)
                self.action_index += [
                    self.action_index[i] + segment * controlled * num_lanes
                ]

        self.action_index = {}
        action_list = [0]
        index = 0
        for (edge, num_segments, controlled) in self.segments:
            if controlled:
                if self.symmetric:
                    self.action_index[edge] = action_list[index]
                    action_list += [action_list[index] + num_segments * controlled]
                else:
                    num_lanes = self.k.scenario.num_lanes(edge)
                    self.action_index[edge] = action_list[index]
                    action_list += [
                        action_list[index] +
                        num_segments * controlled * num_lanes
                    ]
                index += 1

        # inflow to keep track of for observations
        self.inflow = add_env_params["start_inflow"]

        # mapping from vehicle id to lane
        self.exit_history_seconds = add_env_params["exit_history_seconds"]
        self.id_to_lane_dict = {}
        self.exit_counter = np.zeros((int(self.exit_history_seconds / self.sim_step), self.scaling * MAX_LANES))

    @property
    def observation_space(self):
        """See class definition."""
        num_obs = 0
        # density and velocity for rl and non-rl vehicles per segment
        # Last elements are the outflow and inflows
        for segment in self.obs_segments:
            num_obs += 4 * segment[1] * self.k.scenario.num_lanes(segment[0])
        num_obs += 2

        # If we have a fair reward, we also return the current statistics
        if self.env_params.additional_params["fair_reward"]:
            num_obs += 4

        return Box(low=-3.0, high=3.0, shape=(num_obs, ), dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        if self.symmetric:
            action_size = self.total_controlled_segments
        else:
            action_size = 0.0
            for segment in self.segments:  # iterate over segments
                if segment[2]:  # if controlled
                    num_lanes = self.k.scenario.num_lanes(segment[0])
                    action_size += num_lanes * segment[1]
        add_params = self.env_params.additional_params
        max_accel = add_params.get("max_accel")
        max_decel = add_params.get("max_decel")
        return Box(
            low=-max_decel*self.sim_step, high=max_accel*self.sim_step,
            shape=(int(action_size), ), dtype=np.float32)

    def get_state(self):
        """See class definition."""

        # update the id to lane mapping
        new_ids = self.k.vehicle.get_departed_ids()
        self.id_to_lane_dict.update({veh_id: self.k.vehicle.get_lane(veh_id) for veh_id in new_ids})
        exit_ids = self.k.vehicle.get_arrived_ids()

        # track which vehicles from which lanes exited
        exit_lanes = [self.id_to_lane_dict[exit_id] for exit_id in exit_ids if exit_id in self.id_to_lane_dict]
        self.exit_counter = np.roll(self.exit_counter, 1, axis=0)
        unique, unique_counts = np.unique(exit_lanes, return_counts=True)

        # count up how many vehicles exited from a given lane
        for lane, unique_count in zip(unique, unique_counts):
            try:
                self.exit_counter[0, lane] = unique_count
            except:
                import ipdb; ipdb.set_trace()

        # state space is number of vehicles in each segment in each lane,
        # number of rl vehicles in each segment in each lane
        # mean speed in each segment, and mean rl speed in each
        # segment in each lane
        num_vehicles_list = []
        num_rl_vehicles_list = []
        vehicle_speeds_list = []
        rl_speeds_list = []
        for i, edge in enumerate(EDGE_LIST):
            num_lanes = self.k.scenario.num_lanes(edge)
            num_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            num_rl_vehicles = np.zeros((self.num_obs_segments[i], num_lanes))
            vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            rl_vehicle_speeds = np.zeros((self.num_obs_segments[i], num_lanes))
            ids = self.k.vehicle.get_ids_by_edge(edge)
            lane_list = self.k.vehicle.get_lane(ids)
            pos_list = self.k.vehicle.get_position(ids)
            for i, id in enumerate(ids):
                segment = np.searchsorted(self.obs_slices[edge],
                                          pos_list[i]) - 1
                if id in self.k.vehicle.get_rl_ids():
                    rl_vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_rl_vehicles[segment, lane_list[i]] += 1
                else:
                    vehicle_speeds[segment, lane_list[i]] \
                        += self.k.vehicle.get_speed(id)
                    num_vehicles[segment, lane_list[i]] += 1

            # normalize

            num_vehicles /= NUM_VEHICLE_NORM
            num_rl_vehicles /= NUM_VEHICLE_NORM
            num_vehicles_list += num_vehicles.flatten().tolist()
            num_rl_vehicles_list += num_rl_vehicles.flatten().tolist()
            vehicle_speeds_list += vehicle_speeds.flatten().tolist()
            rl_speeds_list += rl_vehicle_speeds.flatten().tolist()

        unnorm_veh_list = np.asarray(num_vehicles_list) * NUM_VEHICLE_NORM
        unnorm_rl_list = np.asarray(num_rl_vehicles_list) * NUM_VEHICLE_NORM

        # compute the mean speed if the speed isn't zero
        num_rl = len(num_rl_vehicles_list)
        num_veh = len(num_vehicles_list)
        mean_speed = np.nan_to_num([
            vehicle_speeds_list[i] / unnorm_veh_list[i]
            if int(unnorm_veh_list[i]) else 0 for i in range(num_veh)
        ])
        mean_speed_norm = mean_speed / 50
        mean_rl_speed = np.nan_to_num([
            rl_speeds_list[i] / unnorm_rl_list[i]
            if int(unnorm_rl_list[i]) else 0 for i in range(num_rl)
        ]) / 50
        outflow = np.asarray(
            self.k.vehicle.get_outflow_rate(20 * self.sim_step) / 3000.0)
        obs = np.concatenate((num_vehicles_list, num_rl_vehicles_list,
                               mean_speed_norm, mean_rl_speed, [outflow],
                               [self.inflow / 3000.0]))

        # Give the vehicles the relevant statistics on ratios of exited vehicles
        if self.env_params.additional_params["fair_reward"]:
            exit_ratios = np.sum(self.exit_counter, axis=0)

            # put a count of one in all the lanes with zero counts so far so the entropy doesn't blow up
            exit_ratios[exit_ratios == 0] = 1

            # convert to probabilities
            exit_ratios = exit_ratios / np.sum(exit_ratios)
            obs = np.concatenate((obs, exit_ratios))
        return obs

    def _apply_rl_actions(self, rl_actions):
        """
        RL actions are split up into 3 levels.

        * First, they're split into edge actions.
        * Then they're split into segment actions.
        * Then they're split into lane actions.
        """
        for rl_id in self.k.vehicle.get_rl_ids():
            edge = self.k.vehicle.get_edge(rl_id)
            lane = self.k.vehicle.get_lane(rl_id)
            if edge:
                # If in outer lanes, on a controlled edge, in a controlled lane
                if edge[0] != ':' and edge in self.controlled_edges:
                    pos = self.k.vehicle.get_position(rl_id)

                    if not self.symmetric:
                        num_lanes = self.k.scenario.num_lanes(edge)
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[int(lane) + bucket * num_lanes +
                                            self.action_index[edge]]
                    else:
                        # find what segment we fall into
                        bucket = np.searchsorted(self.slices[edge], pos) - 1
                        action = rl_actions[bucket + self.action_index[edge]]

                    self.k.vehicle.apply_acceleration(rl_id, action)

    def compute_reward(self, rl_actions, **kwargs):
        """Outflow rate over last ten seconds normalized to max of 1."""
        if self.env_params.evaluate:
            if self.time_counter == self.env_params.horizon:
                reward = self.k.vehicle.get_outflow_rate(500)
            else:
                return 0
        else:
            add_params = self.env_params.additional_params
            # reward is the mean AV speed
            if add_params["speed_reward"]:
                rl_ids = self.k.vehicle.get_rl_ids()
                mean_vel = np.mean(self.k.vehicle.get_speed(rl_ids)) / 60.0
                reward = mean_vel
            # reward is how close the entry lanes of the exiting vehicles are to a uniform distribution
            elif add_params["fair_reward"]:
                reward = 0
                if len(self.k.vehicle.get_arrived_ids()) > 0:
                    exit_ratios = np.sum(self.exit_counter, axis=0)

                    # put a count of one in all the lanes with zero counts so far so the entropy doesn't blow up
                    exit_ratios[exit_ratios == 0] = 1

                    exit_ratios = exit_ratios / np.sum(exit_ratios)
                    # the reward is the entropy of the exiting distributions
                    reward = -np.sum(exit_ratios * np.log(exit_ratios))

            # reward is the outflow over "num_sample_seconds" seconds
            else:
                reward = self.k.vehicle.get_outflow_rate(int(add_params["num_sample_seconds"] / self.sim_step)) / 2000.0 - \
                         self.env_params.additional_params["life_penalty"]

            if add_params["congest_penalty"]:
                num_vehs = len(self.k.vehicle.get_ids_by_edge('4'))
                if num_vehs > add_params["congest_penalty_start"] * self.scaling:
                    penalty = (num_vehs - add_params["congest_penalty_start"] * self.scaling) / 10.0
                    reward -= penalty

                num_vehs = len(self.k.vehicle.get_ids_by_edge('3'))
                if num_vehs > add_params["congest_penalty_start"] * self.scaling:
                    penalty = (num_vehs - add_params["congest_penalty_start"] * self.scaling) / 10.0
                    reward -= penalty
        return reward

    def reset(self, new_inflow_rate=None):
        """Reset the environment with a new inflow rate.

        The diverse set of inflows are used to generate a policy that is more
        robust with respect to the inflow rate. The inflow rate is update by
        creating a new scenario similar to the previous one, but with a new
        Inflow object with a rate within the additional environment parameter
        "inflow_range", which is a list consisting of the smallest and largest
        allowable inflow rates.

        **WARNING**: The inflows assume there are vehicles of type
        "av" and "human" within the VehicleParams object.
        """
        add_params = self.env_params.additional_params
        if add_params.get("reset_inflow"):
            inflow_range = add_params.get("inflow_range")
            if new_inflow_rate:
                flow_rate = new_inflow_rate
            else:
                flow_rate = np.random.uniform(
                    min(inflow_range), max(inflow_range)) * self.scaling
            self.inflow = flow_rate
            print('New flow rate is ', flow_rate)
            for _ in range(100):
                try:

                    vehicles = VehicleParams()
                    if not np.isclose(add_params.get("av_frac"), 1):
                        vehicles.add(
                            veh_id="human",
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=add_params.get("lc_mode"),
                            ),
                            num_vehicles=1)
                        vehicles.add(
                            veh_id="av",
                            acceleration_controller=(RLController, {}),
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=0,
                            ),
                            num_vehicles=1)
                    else:
                        vehicles.add(
                            veh_id="av",
                            acceleration_controller=(RLController, {}),
                            lane_change_controller=(SimLaneChangeController, {}),
                            routing_controller=(ContinuousRouter, {}),
                            car_following_params=SumoCarFollowingParams(
                                speed_mode=31,
                            ),
                            lane_change_params=SumoLaneChangeParams(
                                lane_change_mode=add_params.get("lc_mode"),
                            ),
                            num_vehicles=1)

                    inflow = InFlows()
                    if not np.isclose(add_params.get("av_frac"), 1.0):
                        inflow.add(
                            veh_type="av",
                            edge="1",
                            vehs_per_hour=flow_rate * add_params.get("av_frac"),
                            departLane="random",
                            departSpeed=10.0)
                        inflow.add(
                            veh_type="human",
                            edge="1",
                            vehs_per_hour=flow_rate * (1 - add_params.get("av_frac")),
                            departLane="random",
                            departSpeed=10.0)
                    else:
                        inflow.add(
                            veh_type="av",
                            edge="1",
                            vehs_per_hour=flow_rate,
                            departLane="random",
                            departSpeed=10.0)

                    # all other network parameters should match the previous
                    # environment (we only want to change the inflow)
                    additional_net_params = {
                        "scaling": self.scaling,
                        "speed_limit": self.net_params.
                            additional_params['speed_limit']
                    }
                    net_params = NetParams(
                        inflows=inflow,
                        no_internal_links=False,
                        additional_params=additional_net_params)

                    # recreate the scenario object
                    self.scenario = self.scenario.__class__(
                        name=self.scenario.orig_name,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.initial_config,
                        traffic_lights=self.scenario.traffic_lights)
                    # restart the sumo instance
                    self.restart_simulation(
                        sim_params=self.sim_params,
                        render=self.sim_params.render)
                    self.k.vehicle = deepcopy(self.initial_vehicles)
                    self.k.vehicle.kernel_api = self.k.kernel_api
                    self.k.vehicle.master_kernel = self.k

                    observation = super().reset()

                    # reset the timer to zero
                    self.time_counter = 0

                    # update the vehicle to lane dict
                    self.id_to_lane_dict = {}
                    new_ids = self.k.vehicle.get_departed_ids()
                    self.id_to_lane_dict.update({veh_id: self.k.vehicle.get_lane(veh_id) for veh_id in new_ids})
                    self.id_to_lane_dict.update({veh_id: self.k.vehicle.get_lane(veh_id) for veh_id in
                                                 self.k.vehicle.get_ids()})

                    return observation

                except Exception as e:
                    print('error on reset ', e)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        # update the vehicle to lane dict
        self.id_to_lane_dict = {}
        new_ids = self.k.vehicle.get_departed_ids()
        self.id_to_lane_dict.update({veh_id: self.k.vehicle.get_lane(veh_id) for veh_id in new_ids})
        self.id_to_lane_dict.update({veh_id: self.k.vehicle.get_lane(veh_id) for veh_id in
                                     self.k.vehicle.get_ids()})

        return observation
