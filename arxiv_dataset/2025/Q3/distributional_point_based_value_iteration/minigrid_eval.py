import time
import pickle
import hashlib
import minigrid
from minigrid.core.world_object import WorldObj, Wall, Door, Key, Goal
import networkx as nx
import numpy as np
import seaborn as sns
import gymnasium as gym
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
from minigrid.wrappers import ViewSizeWrapper, ReseedWrapper
from typing import Dict, Any, Tuple, List, ValuesView, KeysView, ItemsView
from enum import Enum, unique
from collections import Counter

from agents.pbvi import PBVI
from agents.dpbvi import DPBVI


@unique
class Action(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6


@unique
class Position(Enum):
    TOP_LEFT = 0
    TOP_CENTER = 1
    TOP_RIGHT = 2
    CENTER_LEFT = 3
    CENTER = 4
    CENTER_RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM_CENTER = 7
    BOTTOM_RIGHT = 8


@unique
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


@unique
class KeyPosition(Enum):
    TOP_LEFT = 0
    CENTER_LEFT = 1
    BOTTOM_LEFT = 2
    ACQUIRED = 3


@unique
class DoorPosition(Enum):
    TOP_CENTER = 0
    CENTER = 1
    BOTTOM_CENTER = 2


@unique
class TileState(Enum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2


@unique
class GoalPosition(Enum):
    TOP_RIGHT = 0
    CENTER_RIGHT = 1
    BOTTOM_RIGHT = 2


@unique
class ObservationObject(Enum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10


class State:
    def __init__(self, player_pos: Position, player_dir: Direction, key_pos: KeyPosition, door_pos: DoorPosition, door_state: TileState, goal_pos: GoalPosition):
        assert 0 <= player_pos.value < 9, 'player_pos must be in [0, 8]'
        assert 0 <= player_dir.value < 4, 'player_dir must be in [0, 3]'
        assert 0 <= key_pos.value < 4, 'key_pos must be in [0, 3]' # HACK: assuming key cant be dropped #9, 'key_pos must be in [0, 8]'
        assert 0 <= door_pos.value < 3, 'door_pos must be in [0, 2]'
        assert 0 <= door_state.value < 3, 'door_state must be in [0, 2]'
        assert 0 <= goal_pos.value < 3, 'goal_pos must be in [0, 2]'
        self._player_pos = player_pos
        self._player_dir = player_dir
        self._key_pos = key_pos
        self._door_pos = door_pos
        self._door_state = door_state
        self._goal_pos = goal_pos

    def __hash__(self) -> int:
        return hash((self._player_pos, self._player_dir, self._key_pos, self._door_pos, self._door_state, self._goal_pos))

    def __eq__(self, other: 'State') -> bool:
        return (self._player_pos == other.player_pos
                and self._player_dir == other.player_dir
                and self._key_pos == other.key_pos
                and self._door_pos == other.door_pos
                and self._door_state == other.door_state
                and self._goal_pos == other.goal_pos)

    @property
    def player_pos(self) -> Position:
        return self._player_pos

    @property
    def player_dir(self) -> Direction:
        """
        0 is right, 1 is down, 2 is left, 3 is up
        :return:
        """
        return self._player_dir

    @property
    def key_pos(self) -> KeyPosition:
        """
        0 - 2: key is not acquired. 3: key is acquired
        0 is top left, 1 is center left, 2 is bottom left
        :return: key state (0-3)
        """
        return self._key_pos

    @property
    def door_pos(self) -> DoorPosition:
        """
        0 is top center, 1 is center, 2 is bottom center
        :return: door position state
        """
        return self._door_pos

    @property
    def door_state(self) -> TileState:
        """
        0 is open, 1 is locked
        :return: door state
        """
        return self._door_state

    @property
    def goal_pos(self) -> GoalPosition:
        """
        0 is top right, 1 is center right, 2 is bottom right
        :return: goal state
        """
        return self._goal_pos


class Observation:
    def __init__(self, obs_: np.ndarray, direction: Direction):
        self._raw_obs = obs_
        self._direction = direction
        # build readable observation
        # left
        self._far_left_obj_obs = ObservationObject(obs_[0, 0, 0])
        self._far_left_state_obs = TileState(obs_[0, 0, 2])
        self._center_left_obj_obs = ObservationObject(obs_[0, 1, 0])
        self._center_left_state_obs = TileState(obs_[0, 1, 2])
        self._near_left_obj_obs = ObservationObject(obs_[0, 2, 0])
        self._near_left_state_obs = TileState(obs_[0, 2, 2])
        # center
        self._far_center_obj_obs = ObservationObject(obs_[1, 0, 0])
        self._far_center_state_obs = TileState(obs_[1, 0, 2])
        self._center_obj_obs = ObservationObject(obs_[1, 1, 0])
        self._center_state_obs = TileState(obs_[1, 1, 2])
        self._near_center_obj_obs = ObservationObject(obs_[1, 2, 0])
        self._near_center_state_obs = TileState(obs_[1, 2, 2])
        # right
        self._far_right_obj_obs = ObservationObject(obs_[2, 0, 0])
        self._far_right_state_obs = TileState(obs_[2, 0, 2])
        self._center_right_obj_obs = ObservationObject(obs_[2, 1, 0])
        self._center_right_state_obs = TileState(obs_[2, 1, 2])
        self._near_right_obj_obs = ObservationObject(obs_[2, 2, 0])
        self._near_right_state_obs = TileState(obs_[2, 2, 2])

    def __hash__(self) -> int:
        hasher = hashlib.sha256()
        flattened_obs = self._raw_obs.flatten()
        flattened_obs = np.concatenate([flattened_obs, np.array([self._direction.value])])
        flattened_obs = tuple(flattened_obs)
        flattened_obs = str(flattened_obs).encode('utf-8')
        hasher.update(flattened_obs)
        hex_digest = hasher.hexdigest()
        hashcode = int(hex_digest, 16)
        return hashcode

    def __eq__(self, other: 'Observation') -> bool:
        return np.array_equal(self._raw_obs, other.raw_obs) and self._direction == other.direction

    @property
    def raw_obs(self) -> np.ndarray:
        return self._raw_obs

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def far_left_obj_obs(self) -> ObservationObject:
        return self._far_left_obj_obs

    @property
    def far_left_state_obs(self) -> TileState:
        return self._far_left_state_obs

    @property
    def center_left_obj_obs(self) -> ObservationObject:
        return self._center_left_obj_obs

    @property
    def center_left_state_obs(self) -> TileState:
        return self._center_left_state_obs

    @property
    def near_left_obj_obs(self) -> ObservationObject:
        return self._near_left_obj_obs

    @property
    def near_left_state_obs(self) -> TileState:
        return self._near_left_state_obs

    @property
    def far_center_obj_obs(self) -> ObservationObject:
        return self._far_center_obj_obs

    @property
    def far_center_state_obs(self) -> TileState:
        return self._far_center_state_obs

    @property
    def center_obj_obs(self) -> ObservationObject:
        return self._center_obj_obs

    @property
    def center_state_obs(self) -> TileState:
        return self._center_state_obs

    @property
    def near_center_obj_obs(self) -> ObservationObject:
        return self._near_center_obj_obs

    @property
    def near_center_state_obs(self) -> TileState:
        return self._near_center_state_obs

    @property
    def far_right_obj_obs(self) -> ObservationObject:
        return self._far_right_obj_obs

    @property
    def far_right_state_obs(self) -> TileState:
        return self._far_right_state_obs

    @property
    def center_right_obj_obs(self) -> ObservationObject:
        return self._center_right_obj_obs

    @property
    def center_right_state_obs(self) -> TileState:
        return self._center_right_state_obs

    @property
    def near_right_obj_obs(self) -> ObservationObject:
        return self._near_right_obj_obs

    @property
    def near_right_state_obs(self) -> TileState:
        return self._near_right_state_obs


class MinigridUtilities:
    def __init__(self, env_: minigrid.minigrid_env.MiniGridEnv, initial_state_: State):
        self._env = env_
        self._initial_state = initial_state_
        self._prev_state = initial_state_

    def extract_state_from_env(self) -> State:
        agent_dir = Direction(self._env.agent_dir)
        agent_pos = Position(self._env.agent_pos[0] - 1 + (self._env.agent_pos[1] - 1) * 3)
        if isinstance(self._env.carrying, Key):
            key_pos = KeyPosition.ACQUIRED
        else:
            key_pos = KeyPosition.TOP_LEFT # assume key is always in the key start state unless picked up - simplification
        goal_pos = GoalPosition.BOTTOM_RIGHT # assume goal is always bottom right - simplification
        door_pos = DoorPosition.CENTER # assume door is always center - simplification
        door_state = TileState.LOCKED if self._env.grid.get(2, 2).is_locked else TileState.OPEN # assume door is either locked or open - simplification
        state = State(agent_pos, agent_dir, key_pos, door_pos, door_state, goal_pos)
        return state

    @property
    def initial_state(self) -> State:
        return self._initial_state

    @property
    def prev_state(self) -> State:
        return self._prev_state

    @prev_state.setter
    def prev_state(self, state: State):
        self._prev_state = state


class StateSpace:
    def __init__(self, transition_ctr: Counter, state_obs_ctr: Counter):
        unique_states = set()
        for state, _, next_state in transition_ctr.keys():
            unique_states.add(state)
            unique_states.add(next_state)
        num_unique = len(unique_states)
        for state, _ in state_obs_ctr.keys():
            unique_states.add(state)
        assert len(unique_states) == num_unique, 'State space mismatch'
        self._idx_state_dict = dict()
        self._state_idx_dict = dict()
        for idx, state in enumerate(unique_states):
            self._idx_state_dict[idx] = state
            self._state_idx_dict[state] = idx

    def get_state_index(self, state: State) -> int:
        return self._state_idx_dict[state]

    def get_state(self, idx: int) -> State:
        return self._idx_state_dict[idx]

    @property
    def state_indices(self) -> KeysView[int]:
        return self._idx_state_dict.keys()

    @property
    def state_index_pairs(self) -> ItemsView[int, State]:
        return self._idx_state_dict.items()

    @property
    def cardinality(self) -> int:
        return len(self._idx_state_dict)

class TransitionModel:
    def __init__(self, transition_ctr: Counter, state_space_: StateSpace):
        self._transition_ctr = transition_ctr
        self._state_space = state_space_
        self._transition_model = np.zeros((state_space_.cardinality, len(Action), state_space_.cardinality), dtype=np.float32)
        for (state, action, next_state), count in transition_ctr.items():
            state_idx = state_space_.get_state_index(state)
            action_idx = action
            next_state_idx = state_space_.get_state_index(next_state)
            self._transition_model[state_idx, action_idx, next_state_idx] = count
        # normalize
        trans_sum = self._transition_model.sum(axis=-1, keepdims=True)
        self._transition_model = np.where(trans_sum > 0, self._transition_model / trans_sum, 0)
        trans_sum = self._transition_model.sum(axis=-1)
        assert np.allclose(np.where(trans_sum > 0, trans_sum, 1.0), 1.0), 'Transition model not normalized' # filter out the zeros - some transitions arent recorded

    @property
    def transition_model(self) -> np.ndarray:
        return self._transition_model


class ObservationSpace:
    def __init__(self, state_obs_ctr: Counter):
        self._state_obs_ctr = state_obs_ctr
        self._observation_space = set()
        for state, obs in state_obs_ctr.keys():
            self._observation_space.add(obs)
        self._idx_obs_dict = dict()
        self._obs_idx_dict = dict()
        for idx, obs in enumerate(self._observation_space):
            self._idx_obs_dict[idx] = obs
            self._obs_idx_dict[obs] = idx

    def get_observation_index(self, obs: Observation) -> int:
        return self._obs_idx_dict[obs]

    def get_observation(self, idx: int) -> Observation:
        return self._idx_obs_dict[idx]

    @property
    def observation_indices(self) -> KeysView[int]:
        return self._idx_obs_dict.keys()

    @property
    def observation_index_pairs(self) -> ItemsView[int, Observation]:
        return self._idx_obs_dict.items()

    @property
    def cardinality(self) -> int:
        return len(self._idx_obs_dict)


class SensorModel:
    def __init__(self, state_obs_ctr: Counter, observation_space_: ObservationSpace, state_space_: StateSpace):
        self._state_obs_ctr = state_obs_ctr
        self._observation_space = observation_space_
        self._state_space = state_space_
        self._sensor_model = np.zeros((state_space_.cardinality, observation_space_.cardinality), dtype=np.float32)
        for (state, obs), count in state_obs_ctr.items():
            state_idx = state_space_.get_state_index(state)
            obs_idx = observation_space_.get_observation_index(obs)
            self._sensor_model[state_idx, obs_idx] = count
        # normalize
        sum_obs = self._sensor_model.sum(axis=-1, keepdims=True)
        self._sensor_model = np.where(sum_obs > 0, self._sensor_model / sum_obs, 0)
        sum_obs = self._sensor_model.sum(axis=-1)
        assert np.allclose(np.where(sum_obs > 0, sum_obs, 1.0), 1.0), 'Sensor model not normalized'

    @property
    def sensor_model(self) -> np.ndarray:
        return self._sensor_model


class RewardFunction:
    def __init__(self, state_space_: StateSpace, transition_ctr: Counter, transition_model_: TransitionModel):
        self._reward_function = np.zeros((state_space_.cardinality, len(Action)), dtype=np.float32)
        for (state, action, next_state), count in transition_ctr.items():
            if next_state.player_pos == Position.BOTTOM_RIGHT or state.player_pos == Position.BOTTOM_RIGHT:
                transition_prob = transition_model_.transition_model[state_space_.get_state_index(state), action, state_space_.get_state_index(next_state)]
                self._reward_function[state_space_.get_state_index(state), action] = 1.0 * transition_prob

    @property
    def reward_function(self) -> np.ndarray:
        return self._reward_function

if __name__ == '__main__':
    """
    State: position of agent, direction of agent, position of key (includes whether acquired), position of door, position of goal
    (grid is 3x3 - 0 is top left, 2 is top right, 6 is bottom left, 8 is bottom right)
    Actions: left, right, forward, pickup, drop, toggle, done
    Observations: 
    1. 3 x 3 x 3 grid of what the agent sees (it cannot see through doors or walls)
        - Obs Column x Obs Row x Tile Encoding (relative to agent facing direction)
            - 0 x 0: far left
            - 0 x 1: center left
            - 0 x 2: near left
            - 1 x 0: far center
            - 1 x 1: center
            - 1 x 2: near center (where agent is)
            - 2 x 0: far right
            - 2 x 1: center right
            - 2 x 2: near right
        - Each tile is encoded with a three-tuple
            - Object Index: unseen, empty, wall, floor, door, key, ball, box, goal, lava, agent
            - Color Index: red, green, blue, purple, yellow, grey
            - (door) State: open, closed, locked
    2. direction (agent facing): right, down, left, up
    """

    keyboard_play = False
    env = gym.make('MiniGrid-DoorKey-5x5-v0', render_mode='rgb_array' if keyboard_play else 'human', max_steps=500)
    initial_state = State(Position.BOTTOM_LEFT, Direction.LEFT, KeyPosition.TOP_LEFT, DoorPosition.CENTER, TileState.LOCKED, GoalPosition.BOTTOM_RIGHT)
    env_utils = MinigridUtilities(env.unwrapped, initial_state)
    env = ViewSizeWrapper(env, agent_view_size=3) # simplify the observation space
    env = ReseedWrapper(env, seeds=[7])
    if keyboard_play:
        keys_to_actions = {
            'a': 0,  # left
            's': 1,  # right
            'w': 2,  # forward
            'd': 3,  # pickup
            'q': 4,  # drop
            'e': 5,  # toggle
            ' ': 6,  # done
        }
        transition_model_counter = Counter()
        state_obs_counter = Counter()
        def play_callback(obs_t: Dict[str, Any], obs_tp1: Dict[str, Any], action: int, reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]):
            current_state = env_utils.extract_state_from_env()
            transition_model_counter[(env_utils.prev_state, action, current_state)] += 1
            obs = Observation(obs_tp1['image'], Direction(obs_tp1['direction']))
            state_obs_counter[(current_state, obs)] += 1
            env_utils.prev_state = current_state
        play(env, callback=play_callback, keys_to_action=keys_to_actions, noop=6, wait_on_player=True)
        pkl_dict = {
            'transition_ctr': transition_model_counter,
            'state_obs_ctr': state_obs_counter
        }
        with open('minigrid_counters.pkl', 'wb') as f:
            pickle.dump(pkl_dict, f)
    else:
        with open('minigrid_counters.pkl', 'rb') as f:
            pkl_dict = pickle.load(f)
        transition_model_counter = pkl_dict['transition_ctr']
        state_obs_counter = pkl_dict['state_obs_ctr']

        # mini validation
        for state, obs in state_obs_counter.keys():
            found_ct = 0
            for state_other, obs_other in state_obs_counter.keys():
                if state == state_other:
                    found_ct += 1
            assert found_ct == 1, f'Found {found_ct} for state {state}'
        for state, action, next_state in transition_model_counter.keys():
            found_ct = 0
            for state_other, action_other, next_state_other in transition_model_counter.keys():
                if state == state_other and action == action_other:
                    found_ct += 1
            assert found_ct == 1, f'Found {found_ct} for state {state}, action {action}'

        state_space = StateSpace(transition_model_counter, state_obs_counter)
        transition_model = TransitionModel(transition_model_counter, state_space)
        observation_space = ObservationSpace(state_obs_counter)
        sensor_model = SensorModel(state_obs_counter, observation_space, state_space)
        reward_function = RewardFunction(state_space, transition_model_counter, transition_model)
        terminal_states = np.zeros(state_space.cardinality, dtype=np.bool_)
        goal_state_idx = state_space.get_state_index(State(Position.BOTTOM_RIGHT, Direction.DOWN, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT))
        terminal_states[goal_state_idx] = True

        # states to plan for
        states_to_plan_for = [
            initial_state,
            State(Position.BOTTOM_LEFT, Direction.UP, KeyPosition.TOP_LEFT, DoorPosition.CENTER, TileState.LOCKED, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_LEFT, Direction.UP, KeyPosition.TOP_LEFT, DoorPosition.CENTER, TileState.LOCKED, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_LEFT, Direction.UP, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.LOCKED, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_LEFT, Direction.RIGHT, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.LOCKED, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_LEFT, Direction.RIGHT, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER, Direction.RIGHT, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_RIGHT, Direction.RIGHT, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT),
            State(Position.CENTER_RIGHT, Direction.DOWN, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT),
            State(Position.BOTTOM_RIGHT, Direction.DOWN, KeyPosition.ACQUIRED, DoorPosition.CENTER, TileState.OPEN, GoalPosition.BOTTOM_RIGHT),
        ]

        expected_action_seq = [
            Action.RIGHT,
            Action.FORWARD,
            Action.PICKUP,
            Action.RIGHT,
            Action.TOGGLE,
            Action.FORWARD,
            Action.FORWARD,
            Action.RIGHT,
            Action.FORWARD,
        ]

        # for i, (state, action) in enumerate(zip(states_to_plan_for, expected_action_seq)):
        #     if i < len(states_to_plan_for) - 1:
        #         print(f'State {i}: {state} -> Action: {action} -> Next State: {states_to_plan_for[i+1]} - Transition Prob: {transition_model.transition_model[state_space.get_state_index(state), action.value, state_space.get_state_index(states_to_plan_for[i+1])]}')

        # convert to numpy array
        beliefs = np.zeros((len(states_to_plan_for), state_space.cardinality), dtype=np.float32)
        for idx, state in enumerate(states_to_plan_for):
            state_idx = state_space.get_state_index(state)
            beliefs[idx, state_idx] = 1.0

        gamma = 0.9
        max_iterations = 10000
        convergence_eps = 1e-3  # convergence threshold for PBVI and DPBVI
        pbvi = PBVI(transition_model.transition_model, sensor_model.sensor_model, reward_function.reward_function, terminal_states, gamma=gamma)
        policy, beliefs = pbvi.plan(beliefs, steps_before_belief_expansion=max_iterations, max_itrs=max_iterations, convergence_eps=convergence_eps)
        dist_support = (0, 5)
        dpbvi = DPBVI(transition_model.transition_model, sensor_model.sensor_model, reward_function.reward_function, terminal_states, gamma=gamma, dist_support=dist_support)
        policy, beliefs = dpbvi.plan(beliefs, steps_before_belief_expansion=max_iterations, max_itrs=max_iterations, convergence_eps=convergence_eps)
        print(f'All Values Close: {np.allclose(pbvi.values, dpbvi.values)}')
        denom = np.where(np.abs(pbvi.cached_values) > 1e-8, np.abs(pbvi.cached_values), 1e-8)  # avoid division by zero
        max_abs_cached_val_rel_error = (np.abs(pbvi.cached_values - dpbvi.cached_values) / denom).max(axis=1)  # take max across beliefs
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = sns.lineplot(x=np.arange(max_abs_cached_val_rel_error.shape[0]), y=max_abs_cached_val_rel_error, ax=ax)
        ax.set_title('Maximum Value Relative Error per Iteration', fontsize=20)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Max Relative Error', fontsize=16)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        plt.show()

        def update_belief(b: np.ndarray, sensor_model_: SensorModel, transition_fn_: TransitionModel, obs_: Observation, obs_space_: ObservationSpace) -> np.ndarray:
            obs_idx = obs_space_.get_observation_index(obs_)
            b_prime = np.zeros_like(b)
            for s_prime in range(b.shape[0]):
                for s in range(b.shape[0]):
                    b_prime[s_prime] += b[s] * transition_fn_.transition_model[s, :, s_prime].sum(-1) # sum over actions
                b_prime[s_prime] *= sensor_model_.sensor_model[s_prime, obs_idx] # scale by prob of observation
            b_prime /= b_prime.sum() # normalize
            return b_prime

        belief = beliefs[-1].copy()
        obs, info = env.reset()
        while True:
            b_idx = np.isclose(beliefs, belief).all(-1).nonzero()[0][0]
            action = policy[b_idx]
            obs, reward, terminated, truncated, info = env.step(action)
            obs = Observation(obs['image'], Direction(obs['direction']))
            belief = update_belief(belief, sensor_model, transition_model, obs, observation_space)
            done = terminated or truncated
            if done:
                break
            time.sleep(0.5)
