import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from ragen.utils import NoLoggerWarnings
from ragen.env.sokoban.room_utils import generate_room
from ragen.utils import set_seed
import re
import copy
from typing_extensions import override

from ragen.env.base import BaseDiscreteActionEnv

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):

    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(self, **kwargs):
        BaseDiscreteActionEnv.__init__(self)
        self.cur_seq = []
        self.action_sequence = []
        self.search_depth = kwargs.pop('search_depth', 300)
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.reward = 0
        self._valid_actions = []


    # @staticmethod
    # @override
    # def formulate_output(env_feedback: str, done: bool = False):
    #     """
    #     No environment feedback for Sokoban
    #     NOTE hard coded for sokoban easy now
    #     """

    #     return ""



    def extract_action(self, text):
        """
        Extract action from text.
        - 0: Still (Invalid Action)
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
        """
        DIRECTION_MAP = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return self.INVALID_ACTION
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return self.INVALID_ACTION


    def reset(self, mode='tiny_rgb_array', seed=None):
        self._reset_tracking_variables()
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(mode, next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
            return self.render(mode)
        

    def finished(self):
        return self.num_env_steps >= self.max_steps or self.success()

    def success(self):
        return self.boxes_on_target == self.num_boxes
    

    # def step(self, action: int or list):
    #     actions = [action] if isinstance(action, int) else action
            
    #     for act in actions:
    #         with NoLoggerWarnings():
    #             if act != self.INVALID_ACTION:
    #                 self._valid_actions.append(act)
    #             _, reward, done, _ = GymSokobanEnv.step(self, action, observation_mode='tiny_rgb_array')
    #         if done:
    #             break
            
    #     obs = self.render()
    #     return obs, reward, done, _

    def step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).
        """
        assert not self.success()

        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {"action_is_effective": False}
        prev_player_position = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action, observation_mode='tiny_rgb_array')
        
        # # NOTE re-define reward for sokoban
        # reward = -1 # format reward
        # if done:
        #     reward = 1 # success reward
            
        obs = self.render()
        return obs, reward, done, {"action_is_effective": not np.array_equal(prev_player_position, self.player_position)}
     

    def render(self, mode='tiny_rgb_array'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img


        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
    
        
    def copy(self):
        new_self = SokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    



    # def _reverse_action_sequence(self, action_sequence):
    #     def reverse_action(action):
    #         return (action % 2 + 1) % 2 + 2 * (action // 2) # 0 <-> 1, 2 <-> 3
    #     return [reverse_action(action) + 1 for action in action_sequence[::-1]] # action + 1 to match the action space
            
    def set_state(self, rendered_state):
        # from the rendered state, set the room state and player position
        self.room_state = np.where(rendered_state == 6, 5, rendered_state)
        self.player_position = np.argwhere(self.room_state == 5)[0]
        
        



GUIDE = """
### Sokoban Puzzle Instructions

In Sokoban, your goal is to move all the boxes to the target spots on the grid. This requires careful planning and strategic moves. Here's how it works:

---

#### Symbols and Their Meaning
- **Walls (`#`)**: These block movement. You can't move through or push anything into walls.
- **Floor (`_`)**: Open spaces where you can walk and move boxes.
- **Targets (`O`)**: The spots where boxes need to go.
- **Boxes (`X`)**: These are what you need to push onto the targets.
- **Player (`P`)**: That's you! You'll move around the grid to push boxes.
- **Box on Target (`√`)**: A box successfully placed on a target.
- **Player on Target (`S`)**: You standing on a target.

---

#### Your Goal
Push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on targets, you win!

---

#### Rules to Remember
1. **You Can Only Push Boxes**: You can't pull them, so plan ahead to avoid getting stuck.
2. **No Moving Through Walls**: You can't walk through or push boxes into walls (`#`).
3. **Avoid Traps**: Don't push boxes into corners or against walls where they can't be moved again.

---

#### Controls
Use these outputs to move the player:
- `1`: Move **up**.
- `2`: Move **down**.
- `3`: Move **left**.
- `4`: Move **right**.

#### Rewards
- **Move**: Each step you take costs 0.1.
- **Push Box to Target**: Each box placed on a target gives you 1.0.
- **Achieve Goal**: When all boxes are on targets, you get a reward of 10.0.

---

#### Example Map
Here's an example of a Sokoban puzzle:

# 	 # 	 # 	 # 	 # 	 # 	 # 	 
# 	 _ 	 _ 	 # 	 # 	 # 	 # 	 
# 	 _ 	 # 	 # 	 # 	 O 	 # 	 
# 	 _ 	 _ 	 _ 	 O 	 _ 	 # 	 
# 	 _ 	 X 	 X 	 _ 	 _ 	 # 	 
# 	 _ 	 O 	 _ 	 X 	 P 	 # 	 
# 	 # 	 # 	 # 	 # 	 # 	 # 	 

Each puzzle will have a different layout, but the rules and goal remain the same.

---

#### Tips for Beginners
1. **Move Boxes Step by Step**: Push them one at a time toward the targets.
2. **Think Ahead**: Avoid pushing a box into a spot where you can’t move it again.

Enjoy the challenge!
"""

if __name__ == '__main__':
    # print(GUIDE)
    import matplotlib.pyplot as plt
    env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=30)
    print(env.reset(seed=1010))
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)
    print()

    env2 = env.copy()
    print(env2.reset(seed=1010))
    np_img = env2.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban2.png', np_img)