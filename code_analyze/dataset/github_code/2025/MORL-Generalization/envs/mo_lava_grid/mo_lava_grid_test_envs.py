import gymnasium as gym
from envs.mo_lava_grid.mo_lava_grid import MOLavaGridDR

class MOLavaGridCorridor(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(6, 7), (4, 5), (8, 5)]
        weightages = [0.6, 0.1, 0.3]
        agent_pos = (2, 6) # middle row left column
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, agent_start_pos=agent_pos, weightages=weightages, **kwargs)

class MOLavaGridIslands(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [1, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 0, 0]
        ]
        goal_pos = [(9, 3), (10, 10), (3, 9)]
        agent_pos = (2, 3) 
        # default weightages (equal)
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, agent_start_pos=agent_pos,  **kwargs)

class MOLavaGridMaze(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0]
        ]
        goal_pos = [(10, 2), (2, 10), (10, 10)]
        weightages = [0.05, 0.05, 0.9]
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)

class MOLavaGridSnake(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0]
        ]
        goal_pos = [(7, 10), (8, 2), (10, 10)]
        weightages = [0.2, 0.3, 0.5]
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)

class MOLavaGridRoom(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0]
        ]
        goal_pos = [(9, 9), (3, 3), (9, 3)]
        weightages = [0.5, 0.3, 0.2]
        agent_pos = (6, 6) # center of the grid
        agent_dir = 3 # face upwards
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_dir=agent_dir, agent_start_pos=agent_pos, **kwargs)

class MOLavaGridLabyrinth(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(6, 6), (8, 8), (10, 10)]
        weightages = [0.5, 0.05, 0.45]
        agent_pos = (2, 10) # bottom left
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, **kwargs)

class MOLavaGridSmiley(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(5, 3), (8, 3), (6, 7)]
        weightages = [0.4, 0.4, 0.2]
        agent_pos = (2, 10) # bottom left
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, **kwargs)

class MOLavaGridCheckerBoard(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0]
        ]
        goal_pos = [(9, 3), (9, 9), (3, 9)]
        weightages = [0.3, 0.1, 0.6]
        agent_pos = (6, 4) # bottom left
        agent_dir = 1 # face downwards
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, agent_start_dir=agent_dir, **kwargs)

class MOLavaGridCorridor10x10(MOLavaGridDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
        ]
        goal_pos = [(3, 7), (5, 7), (7, 7)]
        weightages = [0.6, 0.1, 0.3]
        super().__init__(size=10, bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)  
                         
def register_lava_grid():
    try:
        gym.envs.register(
            id="MOLavaGridDR-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid:MOLavaGridDR",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridCorridor-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridCorridor",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridCorridor10x10-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridCorridor10x10",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridIslands-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridIslands",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridMaze-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridMaze",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridSnake-v0", 
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridSnake",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridRoom-v0", 
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridRoom",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridLabyrinth-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridLabyrinth",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridSmiley-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridSmiley",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGridCheckerBoard-v0",
            entry_point="envs.mo_lava_grid.mo_lava_grid_test_envs:MOLavaGridCheckerBoard",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
