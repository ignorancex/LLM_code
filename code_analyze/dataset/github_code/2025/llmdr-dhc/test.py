'''create test set and test model'''
import os
import random
import pickle
import multiprocessing as mp
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
import configs
import copy
import warnings
warnings.simplefilter("ignore", UserWarning)
from openai import OpenAI
import json

detection_interval = 4
resolution_interval = 16
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()


# 함수들
directiondict = {
    'stay': 0, 'north': 1, 'south': 2, 'west': 3, 'east': 4
}
reverse_directiondict = {v: k for k, v in directiondict.items()}


def PIBT(env, sorted_agents, action_priorities, agents_pos, envmap):
    """
    PIBT 알고리즘: 각 에이전트의 이동 우선순위를 기반으로 충돌 없는 이동 경로 설정
    """
    Reserved = set()
    Moves = {}

    for agent in sorted_agents:
        if agent not in Moves:
            PIBT_H(env, agent, Reserved, Moves, action_priorities, agents_pos, envmap)

    return Moves


def PIBT_H(env, agent, Reserved, Moves, action_priorities, agents_pos, envmap):
    """
    PIBT의 재귀적 탐색 (PIBT-H)
    - 우선순위에 따라 정렬된 액션 리스트를 활용하여 가능한 이동을 시도.
    """

    current_pos = tuple(agents_pos[agent])

    # 선호되는 액션 순으로 액션 호출
    for action in action_priorities[agent]:
        # 액션이 적용될 때의 위치
        if action == 'north':
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif action == 'south':
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif action == 'west':
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif action == 'east':
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif action == 'stay':  # 움직이지 않음
            new_pos = current_pos

        # 액션 제약 조건 (위치가 벽 or 위치가 이미 예약된 위치 or 이동이 예약된 이동)
        if new_pos in Reserved or (new_pos, current_pos) in Reserved or (current_pos, new_pos) in Reserved or envmap[new_pos[0], new_pos[1]] == 1:
            continue
        Moves[agent] = action
        Reserved.add(new_pos)
        Reserved.add((new_pos, current_pos))  # 스왑 이동 방지
        Reserved.add((current_pos, new_pos))

        blocking_agent = None
        for idx, pos in enumerate(agents_pos):
            if tuple(pos) == new_pos and idx != agent and idx not in Moves:
                blocking_agent = idx
                break

        # 방해하는 에이전트가 있을 시
        if blocking_agent is not None:
            if PIBT_H(env, blocking_agent, Reserved, Moves, action_priorities, agents_pos, envmap):
                return True
            else:
                Moves.pop(agent)
                Reserved.remove(new_pos)
                Reserved.remove((new_pos, current_pos))
                Reserved.remove((current_pos, new_pos))

        return True  # 이동 성공

    return False  # 이동 실패


def get_super_agents(agent_groups, env):
    super_agents = []
    for set_of_agents in agent_groups:
        if not set_of_agents:
            continue
        agent_super = max(set_of_agents, key=lambda i: np.sum(np.abs(env.agents_pos[i] - env.goals_pos[i])))
        super_agents.append(agent_super)

    if not super_agents:
        return []

    return super_agents


# 프롬프트
class gpt4pathfinding:
    def detection(self, agents_state):
        prompt_text = f"""
                You are given {detection_interval} action logs of agents to detect deadlocks.
                
                Follow these steps in order:

                1. **Classify deadlocks**:
                    - Detect agents that are exhibiting deadlock conditions.
                    - Deadlock conditions:
                        - No movement: The agent's coordinates remain the same for all {detection_interval} logs in the "Not arrived" state.
                        - Wandering: The agent repeatedly visits the same coordinates during {detection_interval} logs in the "Not arrived" state.
                    - Not deadlocks:
                        - Always "Arrived": The agent remains in the "Arrived" state throughout.
                        - Arrived and stationary: Transitioned from "Not arrived" to "Arrived" and has stopped moving.
                        - Consistent movement: Still "Not arrived" but shows regular coordinate changes without revisiting the same location.

                2. **Group deadlocked agents**:
                    - Group deadlocked agents that are within a 2-Manhattan distance of each other. 2-Manhattan distance means that the sum of the absolute differences between the x-coordinates and y-coordinates of two agents is 2 or less.
                    - If a deadlocked agent is within a 2-Manhattan distance of **another agent that has already arrived, include them in the same group**, as these agents can still cause or experience deadlocks. **This includes cases where deadlocked agents are near arrived agents.**

                3. **Provide solutions**:
                    - Use the "leader" method for independently deadlocked agents or if any agent in the group has a goal more than 8 units away in Manhattan distance.
                    - Use the "radiation" method if all agents in the group are close to their goals (less than 8 units), are deadlocked due to nearby agents, and are likely to experience repeated deadlocks.
                    - If a deadlocked agent is near an arrived agent, check their goals and apply the "radiation" method if needed, as this can cause performance issues.

                Rules:
                - Return "[]" if no deadlocks are found.
                - Ensure no duplicate agents.
                - Penalties apply for trivial or non-deadlock cases.
                
                Below are the {detection_interval} action logs of agents.

                {agents_state}

                Do not generate a description or explanation.

                Provide the agent group status in this JSON format:
                {{
                    "agent_id": [Agent IDs in the same group],
                    "solution": "leader" or "radiation"
                }}

                EXAMPLE 1:
                [
                    {{"agent_id": [1, 24], "solution": "leader"}},
                    {{"agent_id": [4, 5], "solution": "radiation"}}
                ]

                EXAMPLE 2:
                []

                EXAMPLE 3:
                [
                    {{"agent_id": [8], "solution": "leader"}}
                ]
                """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are the manager responsible for detecting whether agents are deadlocked in a MAPF problem. You can infer each agent's state based on their behavior."},
                {"role": "user", "content": prompt_text}
            ],
        )
        return response.choices[0].message.content
    
pathfinder = gpt4pathfinding()


torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1)


def test_model(model_range: Union[int, tuple]):
    '''
    test model in 'models' file with model number 
    '''
    network = Network()
    network.eval()
    network.to(device)

    test_set = configs.test_env_settings

    pool = mp.Pool(mp.cpu_count())

    if isinstance(model_range, int):
        state_dict = torch.load('./models/{}.pth'.format(model_range), map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print('----------test model {}----------'.format(model_range))

        for case in test_set:
            print(f"test set: {case[0]} env {case[1]} agents")
            with open('./test_set_300/{}_{}agents.pth'.format(case[0], case[1]), 'rb') as f:
                tests = pickle.load(f)

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success = 0
            avg_step = 0
            for i, j in ret:
                success += i
                avg_step += j

            print("success rate: {:.2f}%".format(success/len(ret)*100))
            print("average step: {}".format(avg_step/len(ret)))
            print()

    elif isinstance(model_range, tuple):
        for model_name in range(model_range[0], model_range[1]+1, configs.save_interval):
            state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()

            print('----------test model {}----------'.format(model_name))

            for case in test_set:
                print("test set: {} length {} agents {} density".format(case[0], case[1], case[2]))
                with open('./test_set_300/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_one_case, tests)

                success = 0
                avg_step = 0
                for i, j in ret:
                    success += i
                    avg_step += j

                print("success rate: {:.2f}%".format(success/len(ret)*100))
                print("average step: {}".format(avg_step/len(ret)))
                print()

            print('\n')


def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(np.array(env_set[0]), np.array(env_set[1]), np.array(env_set[2]))
    envmap = env.map
    obs, _, pos = env.observe()
    
    done = False
    network.reset()

    num_agents = len(env_set[1])
    
    step = 0
    while not done and env.steps < configs.max_episode_length:
        env_copy = copy.deepcopy(env)
        plan = []
        not_arrived = set()
        sim_obs, sim_last_act, sim_pos = env_copy.observe()
        sim_done = False
        for _ in range(resolution_interval):
            if env_copy.steps >= configs.max_episode_length or sim_done:
                break
            actions, _, _, _ = network.step(torch.as_tensor(sim_obs.astype(np.float32)).to(device), torch.as_tensor(sim_pos.astype(np.float32)).to(device))
            plan.append((actions, copy.deepcopy(sim_pos)))
            (sim_obs, sim_last_act, sim_pos), _, sim_done, _ = env_copy.step(actions)
            for i in range(num_agents):
                if not np.array_equal(env_copy.agents_pos[i], env_copy.goals_pos[i]):
                    not_arrived.add(i)
        not_arrived = list(not_arrived)
        # print(not_arrived)

        observations = env.observe_agents()
        FOV_agents = [
            [*(observations[i][observations[i] != 0] - 1).tolist(), i] if np.any(observations[i]) else [i]
            for i in not_arrived
        ]
        agents_to_prompt = list({agent for agent_list in FOV_agents for agent in agent_list})
        agents_to_prompt = sorted(agents_to_prompt)
        # print(agents_to_prompt)

        planned_steps_dict = {i: [] for i in agents_to_prompt}
        goal_logged = {i: False for i in agents_to_prompt}
        for i in plan[:detection_interval]:
            actions, positions = i
            for agent_idx in agents_to_prompt:
                position = positions[agent_idx]
                # 목표 위치와 현재 위치를 비교하여 도달 여부 판단
                arrived_status = "Arrived" if np.array_equal(position, env.goals_pos[agent_idx]) else "Not arrived"
                if not goal_logged[agent_idx]:
                    planned_steps_dict[agent_idx].append(
                        f"(Position: [{position[0]}, {position[1]}], {arrived_status})"
                    )
                    goal_logged[agent_idx] = True
                else:
                    planned_steps_dict[agent_idx].append(
                        f"(Position: [{position[0]}, {position[1]}], {arrived_status})"
                    )
        agents_state = ""
        for agent_idx in planned_steps_dict:
            agent_goal = f" (Goal: [{env.goals_pos[agent_idx][0]}, {env.goals_pos[agent_idx][1]}])"
            agent_log = ", ".join(planned_steps_dict[agent_idx])
            agents_state += f"Agent {agent_idx}{agent_goal}: {agent_log}\n"

        # print(agents_state)

        gpt4_response = pathfinder.detection(agents_state)
        response_text = gpt4_response
        try:
            start_idx = response_text.index('[')
            end_idx = response_text.rindex(']') + 1
            json_part = response_text[start_idx:end_idx]
            json_data = json.loads(json_part)
            # print("Extracted JSON:", json_data)
        except:
            # print("JSON 부분을 찾을 수 없으므로 deadlock이 없다고 가정합니다.")
            json_data = []

        deadlock_exists = len(json_data) > 0

        if not deadlock_exists:
            for actions, _ in plan:
                if env.steps >= configs.max_episode_length or done:
                    break
                (obs, last_act, pos), _, done, _ = env.step(actions)
                step += 1
        else:
            leader_agents = []
            radiation_agents = []

            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict):
                        if item.get('solution') == 'leader':
                            leader_agents.append(item['agent_id'])
                        elif item.get('solution') == 'radiation':
                            radiation_agents.append(item['agent_id'])
                            
            leader_agents = [[agent for agent in group if agent < num_agents] for group in leader_agents]
            radiation_agents = [[agent for agent in group if agent < num_agents] for group in radiation_agents]

            deadlocked_agents = set()
            for group in leader_agents + radiation_agents:
                deadlocked_agents.update(group)

            all_agents = set(range(num_agents))
            no_deadlock_agents = list(all_agents - deadlocked_agents)

            super_agents = get_super_agents(leader_agents, env)
            yield_agents = [agent for group in leader_agents for agent in group if agent not in super_agents]

            random.shuffle(super_agents)
            random.shuffle(yield_agents)
            for group in radiation_agents:
                random.shuffle(group)
            random.shuffle(radiation_agents)
            random.shuffle(no_deadlock_agents)

            for _ in range(resolution_interval):
                if env.steps >= configs.max_episode_length or done:
                    break
                obs_agents = env.observe_agents()
                observation = env.observe()
                agents_pos = env.agents_pos

                manual_actions = [0 for _ in range(num_agents)]
                ml_planned_actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)).to(device), torch.as_tensor(pos.astype(np.float32)).to(device))

                flattened_radiation_agents = [agent for group in radiation_agents for agent in group]
                random.shuffle(flattened_radiation_agents)
                sorted_agents = super_agents + flattened_radiation_agents + no_deadlock_agents + yield_agents
                action_priorities = {}

                for agent in super_agents:
                    all_actions = ['stay', 'north', 'south', 'west', 'east']
                    directions = []
                    if observation[0][agent][2][4, 4] == 1:
                        directions.append('north')
                    if observation[0][agent][3][4, 4] == 1:
                        directions.append('south')
                    if observation[0][agent][4][4, 4] == 1:
                        directions.append('west')
                    if observation[0][agent][5][4, 4] == 1:
                        directions.append('east')
                    if not directions:
                        directions.append('stay')
                    random.shuffle(directions)
                    remaining_actions = [action for action in all_actions if action not in directions]
                    random.shuffle(remaining_actions)
                    action_priorities[agent] = directions + remaining_actions

                for agent in yield_agents:
                    all_actions = ['stay', 'north', 'south', 'west', 'east']
                    directions = ['stay']
                    remaining_actions = [action for action in all_actions if action not in directions]
                    random.shuffle(remaining_actions)
                    action_priorities[agent] = directions + remaining_actions

                for set_of_agents in radiation_agents:
                    x_values = []
                    y_values = []
                    for agent_idx in set_of_agents:
                        x_values.append(observation[2][agent_idx][0])
                        y_values.append(observation[2][agent_idx][1])
                    if len(x_values) == 0 or len(y_values) == 0:
                        continue
                    avg_x = sum(x_values) / len(x_values)
                    avg_y = sum(y_values) / len(y_values)
                    center_coordinates = (avg_x, avg_y)

                    for agent in set_of_agents:
                        all_actions = ['stay', 'north', 'south', 'west', 'east']
                        directions = ['stay', 'north', 'south', 'west', 'east']
                        row_diff = center_coordinates[0] - observation[2][agent][0]
                        col_diff = center_coordinates[1] - observation[2][agent][1]
                        if row_diff < 0:  # 에이전트가 중앙보다 아래에 있으면 북쪽으로 이동 불가
                            if 'north' in directions:
                                directions.remove('north')
                        elif row_diff > 0:  # 에이전트가 중앙보다 위에 있으면 남쪽으로 이동 불가
                            if 'south' in directions:
                                directions.remove('south')
                        if col_diff < 0:  # 에이전트가 중앙보다 오른쪽에 있으면 서쪽으로 이동 불가
                            if 'west' in directions:
                                directions.remove('west')
                        elif col_diff > 0:  # 에이전트가 중앙보다 왼쪽에 있으면 동쪽으로 이동 불가
                            if 'east' in directions:
                                directions.remove('east')
                        random.shuffle(directions)
                        remaining_actions = [action for action in all_actions if action not in directions]
                        random.shuffle(remaining_actions)
                        action_priorities[agent] = directions + remaining_actions

                for agent in no_deadlock_agents:
                    all_actions = ['stay', 'north', 'south', 'west', 'east']
                    directions = [reverse_directiondict[ml_planned_actions[agent]]]
                    remaining_actions = [action for action in all_actions if action not in directions]
                    random.shuffle(remaining_actions)
                    action_priorities[agent] = directions + remaining_actions

                action_dict = PIBT(env, sorted_agents, action_priorities, agents_pos, envmap)
                manual_actions = [directiondict[action_dict.get(agent, 'stay')] for agent in range(num_agents)]
                (obs, last_act, pos), _, done, _ = env.step(manual_actions)
                step += 1

    return np.array_equal(env.agents_pos, env.goals_pos), step


if __name__ == '__main__':

    # create_test(test_env_settings=configs.test_env_settings, num_test_cases=configs.num_test_cases)
    test_model(337500)
