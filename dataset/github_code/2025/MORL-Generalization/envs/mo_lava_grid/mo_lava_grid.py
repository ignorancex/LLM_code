import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import random
from copy import deepcopy

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Lava, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)
import operator
from functools import reduce
from typing import Optional

from morl_generalization.algos.dr import DREnv

class Goal(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("goal", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Lava(WorldObj):
    def __init__(self):
        super().__init__("lava", "grey")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))

GOAL_IDX_TO_COLOR = {
    0: "green",
    1: "yellow",
    2: "blue",
}

class MOLavaGridDR(MiniGridEnv):
    """
    ## Description
    Multi-objective version of the Minigrid Lava Gap environment. 
    (https://minigrid.farama.org/environments/minigrid/LavaGridEnv/)
    Unlike the original Lava Gap environment, the episode does not terminate when the agent falls
    into lava.

    ## Parameters
    - size: maze is size*size big (including walls)
    - agent_start_pos: agent starting position
    - agent_start_dir: agent starting direction
    - n_goals: number of goals
    - weightages: weightages of goals
    - goal_pos: position to place goals, must be a list of (y, x) tuples
    - n_lava: max number of lava tiles to add
    - bit_map: (size-1)*(size-1) list to indicate maze configuration (0 for empty, 1 for lava)

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Lava Damage (-5 if agent falls into lava)
    - 1: Time Penalty (-3 for every step the agent has taken)

    ## Termination
    The episode ends if any one of the following conditions is met:
    - 1: The agent reaches the goal. +100*n_goals*goal_weightage will be added to all dimensions in the vectorial reward.
    - 2: Timeout (see max_steps).
    """

    def __init__(
        self,
        size=13,
        max_steps=256,
        agent_start_pos=(2, 2),
        agent_start_dir=0,
        n_goals=3, # number of goals
        weightages=None, # weightages of goals
        n_lava=60,
        bit_map=None,
        goal_pos=None,
        is_rgb=False,
        tile_size=8,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.n_goals = n_goals  # Set the number of goals
        self.is_rgb = is_rgb # use RGB img observations

        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Reduce lava if there are too many
        self.n_lava = min(int(n_lava), (size-2)**2 - 1 - n_goals) # -1 for agent start position, -n_goals for goal positions

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True, # Set this to True for maximum speed
            highlight=False, # Fully observable
            **kwargs,
        )

        if weightages is None:
            weightages = [1/n_goals] * n_goals
        assert self.n_goals <= 3, "maximum number of goals is 3"
        assert len(weightages) == self.n_goals, "number of weightages must match number of goals"
        assert sum(weightages) == 1, "weightages must sum to 1"
        if goal_pos is not None:
            assert len(goal_pos) == len(set(goal_pos)), "number of unique goal positions must match number of goals"
            assert all([0 < x < size-1 and 0 < y < size-1 for x, y in goal_pos]), "goal positions must be within the grid"
        self.collected_goals = []
        self.goal_weightages = weightages
        self.immutable_goal_weightages = deepcopy(weightages)
        self.goal_to_goal_idx = {}

        self._gen_grid(self.width, self.height, goal_pos=goal_pos)
        # Default configuration (if provided)
        if bit_map is not None:
            bit_map = np.array(bit_map)
            assert bit_map.shape == (size-4, size-4), "invalid bit map configuration"
            indices = np.argwhere(bit_map)
            for y, x in indices:
                assert not self._reject_fn((x+2, y+2)), "position already occupied, cannot place lava"
                self.put_obj(Lava(), x+2, y+2)

        # Observation space
        if not is_rgb:
            imgShape = (self.width - 2) * (self.height - 2) + 1 + self.n_goals # +n_goals for goal weightages, +1 for agent direction
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(imgShape,),
                dtype='float32'
            )
        else:
            image_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    3,
                    self.width * tile_size,
                    self.height * tile_size,
                ),
                dtype="uint8",
            )
            goal_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.n_goals,),
                dtype="float32",
            )
            self.observation_space = spaces.Dict(
                {
                    "image": image_space,
                    "goal": goal_space,
                }
            )
        # lava damage, time penalty
        self.reward_space = spaces.Box(
            low=np.array([-max_steps, -max_steps]),
            high=np.array([100, 100]),
            shape=(2,),
            dtype=np.float32,
        )
        self.reward_dim = 2

        # Only 3 actions permitted: turn left, turn right, move forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)

    @staticmethod
    def _gen_mission():
        return "visit all goals while balancing lava damage and time penalty"

    def _gen_grid(self, width, height, goal_pos=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # place lava around the inner border
        for i in range(1, width-1):
            self.grid.set(i, 1, Lava())
            self.grid.set(i, height-2, Lava())

        for j in range(1, height-1):
            self.grid.set(1, j, Lava())
            self.grid.set(width-2, j, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place multiple goal squares at random positions
        if goal_pos is None:
            self.goal_positions = self._randomly_place_multiple_goals()
        else:
            for i, pos in enumerate(goal_pos):
                self.put_obj(Goal(GOAL_IDX_TO_COLOR[i]), pos[0], pos[1])
                self.goal_to_goal_idx[(pos[0], pos[1])] = i
            self.goal_positions = goal_pos

    # prevent lava from being place on start and goal positions
    def _reject_fn(self, pos):
        if pos == self.agent_start_pos or pos in self.goal_positions:
            return True
        return False

    def _randomly_place_multiple_goals(self):
        """Randomly place the goal objects avoiding the agent start position."""
        available_positions = [
            (x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1)
            if (x, y) != self.agent_start_pos
        ]

        # Sample unique goal positions
        goal_positions = random.sample(available_positions, self.n_goals)

        # Place goal objects in the grid
        for i, goal_pos in enumerate(goal_positions):
            self.put_obj(Goal(GOAL_IDX_TO_COLOR[i]), goal_pos[0], goal_pos[1])
            self.goal_to_goal_idx[(goal_pos[0], goal_pos[1])] = i

        return goal_positions

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n or action < 0:
            raise ValueError("Invalid action!")
        
        self.step_count += 1

        vec_reward = np.zeros(2, dtype=np.float32) 
        vec_reward[1] = -3 # -3 for time penalty
        terminated = False
        truncated = False
        
        # Get the contents of the current cell (penalize if the agent stays in lava cells)
        current_cell = self.grid.get(*self.agent_pos)
        in_lava = current_cell is not None and current_cell.type == "lava"

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            if in_lava: 
                # stay in lava
                vec_reward[0] = -20

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            if in_lava: 
                # stay in lava
                vec_reward[0] = -20

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                goal_idx = fwd_cell.cur_pos # Get the goal's index
                if goal_idx not in self.collected_goals:
                    self.collected_goals.append(goal_idx)
                    assert not (self.goal_weightages[self.goal_to_goal_idx[goal_idx]] == 0 \
                        and self.immutable_goal_weightages[self.goal_to_goal_idx[goal_idx]] != 0), "goal already collected"
                    vec_reward += self.n_goals * 100 * self.goal_weightages[self.goal_to_goal_idx[goal_idx]]
                    self.goal_weightages[self.goal_to_goal_idx[goal_idx]] = 0
                    if len(self.collected_goals) == self.n_goals:
                        terminated = True
            if fwd_cell is not None and fwd_cell.type == "lava": 
                # walk into lava
                vec_reward[0] = -20
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.is_rgb:
            return self.rgb_observation(), vec_reward, terminated, truncated, {}
        info = {
            "time": self.step_count,
            "goal": self.collected_goals,
        }
        info["original_scalar_reward"] = 0.9 * vec_reward[0] + 0.1 * vec_reward[1]
        
        return self.observation(), vec_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # Step count since episode start
        self.step_count = 0
        self.collected_goals = []

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        if self.render_mode == "human":
            self.render()

        if self.is_rgb:
            return self.rgb_observation(), {}
        self.goal_weightages = deepcopy(self.immutable_goal_weightages) # reset goal weightages back to original
        return self.observation(), {}
    
    def _resample_n_lava(self):
        n_lava = np.random.randint(0, self.n_lava)

        return n_lava
    
    def _randomize_goal_weightages(self):
        """Randomly assign weightages to each goal, ensuring they sum to 1."""
        random_weights = np.random.rand(self.n_goals).astype(np.float32)
        self.goal_weightages = np.array(random_weights / np.sum(random_weights))
        self.immutable_goal_weightages = deepcopy(self.goal_weightages)

    def _reset_agent_start_pos(self):
        """Randomly reset the agent's position."""
        self.agent_start_pos = (0,0) # set it to an invalid position first so it is not include in reject_fn
        available_positions = [
            (x, y) for x in range(1, self.width-1) for y in range(1, self.height-1)
            if not self._reject_fn((x, y))
        ]
        self.agent_start_pos = random.choice(available_positions)
        self.agent_start_dir = random.randint(0, 3)
    
    def reset_random(self):
        """Use domain randomization to create the environment."""
        # Create an empty grid with goals and agent only
        self._reset_agent_start_pos()
        self._randomize_goal_weightages()
        self._gen_grid(self.width, self.height)
        
        # Create a list of all possible positions
        available_positions = [
            (x, y) for x in range(1, self.width-1) for y in range(1, self.height-1)
            if not self._reject_fn((x, y))
        ]

        # Randomly sample `n_lava` unique positions
        n_lava = self._resample_n_lava()
        selected_positions = random.sample(available_positions, n_lava)

        # Place the Lava objects at the randomly selected positions
        for x, y in selected_positions:
            self.put_obj(Lava(), x, y)
        
        # Double check that the agent and goal doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_start_pos)
        assert start_cell is None or start_cell.can_overlap(), "agent's initial position is invalid"

    # use only object color to encode the grid, reduce dimensionality by 3x (original has (obj_type, obj_color, obj_state))
    # MAKE SURE EACH OBJECT HAS A UNIQUE COLOR: grey used by lava, (green, yellow, blue) used by goals, red used by empty, purple used by agent
    def observation(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        sub_grid = full_grid[1:self.width-1, 1:self.height-1, 1] # ignore the walls
        sub_grid[env.agent_pos[0]-1][env.agent_pos[1]-1] = COLOR_TO_IDX['purple'] # agent color

        sub_grid = np.concatenate(([env.agent_dir], self.goal_weightages, sub_grid.flatten()))

        # full_grid = full_grid.flatten()
        # full_grid = np.concatenate([full_grid, np.array(self.goal_weightages)])
        return sub_grid.astype(np.float32) / 1.
    
    def rgb_observation(self):
        env = self.unwrapped
        rgb_img = self.grid.render(
            self.tile_size,
            env.agent_pos,
            env.agent_dir,
        )
        rgb_img = np.moveaxis(rgb_img, -1, 0) # move channel axis to the front
        obs = {
            "image": rgb_img,
            "goal": self.goal_weightages
        }
        return obs
    

if __name__ == "__main__":
    from gymnasium.envs.registration import register
    from mo_utils.evaluation import seed_everything
    import matplotlib.pyplot as plt

    seed_everything(101)

    register(
        id="MOLavaGridDR",
        entry_point="envs.mo_lava_grid.mo_lava_grid:MOLavaGridDR",
    )
    env = gym.make(
        "MOLavaGridDR", 
        render_mode="human",
        # is_rgb=True,
    )

    terminated = False
    env.unwrapped.reset_random()
    env.reset()
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        print(r, terminated, truncated, obs.shape)
        # plt.figure()
        # plt.imshow(obs, vmin=0, vmax=255)
        # plt.show()
        env.render()
        if terminated or truncated:
            env.unwrapped.reset_random()
            env.reset()