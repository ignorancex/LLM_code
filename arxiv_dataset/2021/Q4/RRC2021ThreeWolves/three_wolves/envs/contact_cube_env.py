import time

import gym
import numpy as np
import pybullet
from trifinger_simulation import TriFingerPlatform, visual_objects
from trifinger_simulation.tasks import move_cube_on_trajectory as task

from three_wolves.envs.base_cube_env import ActionType, BaseCubeTrajectoryEnv
from three_wolves.envs.utilities.env_utils import HistoryWrapper, resetCamera
from three_wolves.deep_whole_body_controller import position_controller, contact_planner
from three_wolves.deep_whole_body_controller.utility import pinocchio_utils, reward_utils, trajectory


class ContactControlEnv(BaseCubeTrajectoryEnv):
    def render(self, mode='human'):
        pass

    def __init__(self, goal_trajectory, visualization, randomization, evaluation=False, history_num=1, robot_type='sim'):
        super(ContactControlEnv, self).__init__(
            goal_trajectory=goal_trajectory,
            action_type=ActionType.POSITION,
            step_size=3)
        self.visualization = visualization
        self.randomization = randomization
        self.evaluation = evaluation
        self.observer = HistoryWrapper(history_num)
        self.kinematics = pinocchio_utils.Kinematics(robot_type)
        self.contact_planner = contact_planner.ContactPlanner()
        self.position_controller = position_controller.PositionController(self.kinematics,
                                                                          self.observer, self.step_size)
        self.max_episode = task.EPISODE_LENGTH
        self.tip_force_offset = []
        # create observation space
        spaces = TriFingerPlatform.spaces
        self.observation_space = gym.spaces.Box(
            low=np.hstack([
                spaces.object_position.gym.low,  # cube position
                [-2 * np.pi] * 3,  # cube rpy
                spaces.object_position.gym.low,  # goal position
                [-0.3] * 3,  # goal-cube difference
                [0]  # goal-cube distance
            ]),
            high=np.hstack([
                spaces.object_position.gym.high,  # cube position
                [2 * np.pi] * 3,  # cube rpy
                spaces.object_position.gym.high,  # goal position
                [0.3] * 3,  # goal-cube difference
                [1]  # goal-cube distance
            ])
        )
        self.action_space = self.contact_planner.action_space

    def reset(self):
        """Reset the environment."""
        # hard-reset simulation
        self.goal_marker = None
        del self.platform

        # initialize simulation
        initial_robot_position = (
            TriFingerPlatform.spaces.robot_position.default
        )
        # initialize cube at the centre
        _random_obj_xy_pos = np.random.uniform(
            low=[-0.04] * 2,
            high=[0.04] * 2,
        )
        _random_obj_yaw_ori = np.random.uniform(-2 * np.pi, 2 * np.pi)
        _random_obj_yaw_ori = pybullet.getQuaternionFromEuler([0, 0, _random_obj_yaw_ori])
        random_object_pose = task.move_cube.Pose(
            position=[_random_obj_xy_pos[0],
                      _random_obj_xy_pos[1],
                      task.INITIAL_CUBE_POSITION[2]],
            orientation=_random_obj_yaw_ori
        )
        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=random_object_pose,
        )

        if self.randomization:
            cube_id = self.platform.cube._object_id
            random_mass = 0.094*np.random.uniform(0.9, 1.1)
            random_lateral_friction = 1*np.random.uniform(0.9, 1)
            random_step_size = np.random.randint(1, 6)
            pybullet.changeDynamics(cube_id, -1, mass=random_mass, lateralFriction=random_lateral_friction)
            self.step_size = random_step_size

        # get goal trajectory
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
            resetCamera()

        self.info = {"time_index": -1, "trajectory": trajectory, "eval_score": 0}
        self.step_count = 0
        self.drop_times = 0
        self.tip_force_offset = []

        # initial step
        robot_action = self._gym_action_to_robot_action(self._initial_action)
        t = self.platform.append_desired_action(robot_action)
        self.info["time_index"] = t
        self.step_count += 1
        obs, _ = self._create_observation(self.info["time_index"])
        return obs

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose
        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )
        cube_pos = object_observation.position
        cube_orn = pybullet.getEulerFromQuaternion(object_observation.orientation)
        finger_pos = self.kinematics.forward_kinematics(robot_observation.position)
        obs_dict = {
            "joint_position": robot_observation.position,  # joint position
            "joint_velocity": robot_observation.velocity,  # joint velocity
            "joint_torque": robot_observation.torque,  # joint torque
            "tip_force": robot_observation.tip_force,  # tip force

            "object_position": cube_pos,  # cube position
            "object_rpy": cube_orn,  # cube orientation
            "goal_position": active_goal,  # goal position
            "object_goal_distance": active_goal - cube_pos,  # cube to goal distance

            "tip_0_position": finger_pos[0],  # tri-finger position 0
            "tip_1_position": finger_pos[1],  # tri-finger position 1
            "tip_2_position": finger_pos[2],  # tri-finger position 2
        }
        self.observer.update(obs_dict)
        rl_obs = np.hstack([
            cube_pos,  # cube position
            cube_orn,  # cube rpy
            active_goal,  # goal position
            active_goal - cube_pos,  # goal-cube difference
            np.linalg.norm(active_goal - cube_pos)  # goal-cube distance
        ])
        return rl_obs, obs_dict

    def _internal_step(self, action):
        self.step_count += 1
        # send action to robot
        robot_action = self._gym_action_to_robot_action(action)
        t = self.platform.append_desired_action(robot_action)

        # update goal visualization
        if self.visualization:
            goal_position = task.get_active_goal(self.info["trajectory"], t)
            self.goal_marker.set_state(goal_position, (0, 0, 0, 1))
            time.sleep(0.001)
        return t

    def apply_action(self, action):
        tg = trajectory.get_interpolation_planner(init_pos=self.observer.dt['joint_position'],
                                                  tar_pos=action,
                                                  start_time=0,
                                                  reach_time=self.step_size)
        for i in range(self.step_size):
            if self.step_count >= self.max_episode:
                break
            _action = tg(i + 1)
            t = self._internal_step(_action)
            self.info["time_index"] = t
            _, obs_dict = self._create_observation(self.info["time_index"])

            if self.evaluation:
                eval_score = self.compute_reward(
                    obs_dict["object_position"],
                    obs_dict["goal_position"],
                    self.info,
                )
                self.info['eval_score'] += eval_score

        # return score

    def update(self, policy_action):
        self._last_goal = self.observer.dt['goal_position']
        contact_face_ids, contact_points = self.contact_planner.compute_contact_points(policy_action)
        self.position_controller.update(contact_points, contact_face_ids)

    def step(self, policy_action):
        self.update(policy_action)
        self.position_controller.tips_reach(self.apply_action, self.tip_force_offset)
        reward = 0
        while not self.Dropped() and not self.step_count >= self.max_episode:
            if (self._last_goal != self.observer.dt['goal_position']).all():
                self.update(policy_action)
            cur_phase_action = self.position_controller.get_action()
            self.apply_action(cur_phase_action)
            reward += self.position_controller.get_reward() * 0.001 * self.step_size

        self.drop_times += 1
        done = self.drop_times >= 3 or self.step_count >= self.max_episode

        if self.evaluation:
            done = self.step_count >= self.max_episode

        return self._create_observation(self.info["time_index"])[0], reward, done, self.info

    def Dropped(self):
        tip_touch = np.subtract(self.observer.dt['tip_force'], self.tip_force_offset[0]) > 0
        cube_pos = np.array(self.observer.dt['object_position'])
        tri_distance = [reward_utils.ComputeDist(self.observer.dt['tip_0_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_1_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_2_position'], cube_pos)]
        is_dropped = np.sum(tip_touch) < 2 or any(np.array(tri_distance) > 0.08)
        return is_dropped


class RealContactControlEnv(ContactControlEnv):
    def __init__(self,
                 goal_trajectory):
        super().__init__(goal_trajectory=goal_trajectory,
                         visualization=False,
                         evaluation=False,
                         randomization=False,
                         robot_type='real')
        self.max_episode = task.EPISODE_LENGTH

    def _internal_step(self, action):
        self.step_count += 1
        # send action to robot
        robot_action = self._gym_action_to_robot_action(action)
        t = self.platform.append_desired_action(robot_action)
        return t

    def step(self, policy_action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        self.update(policy_action)
        self.position_controller.tips_reach(self.apply_action, self.tip_force_offset)
        reward = 0
        while not self.Dropped() and not self.step_count >= self.max_episode:
            if list(self._last_goal) != list(self.observer.dt['goal_position']):
                self.update(policy_action)
            cur_phase_action = self.position_controller.get_action()
            self.apply_action(cur_phase_action)
            # reward += self.position_controller.get_reward() * 0.001 * self.step_size
        # self.drop_times += 1
        done = self.step_count >= self.max_episode

        return self._create_observation(self.info["time_index"])[0], reward, done, self.info

    def reset(self):
        import robot_fingers
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        # get goal trajectory
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}
        self.step_count = 0

        # initial step
        for i in range(int(1./(0.001*self.step_size))):
            robot_action = self._gym_action_to_robot_action(self._initial_action)
            t = self.platform.append_desired_action(robot_action)
            self.info["time_index"] = t
            self.step_count += 1
            obs, _ = self._create_observation(self.info["time_index"])

        return obs


if __name__ == '__main__':
    env = ContactControlEnv(goal_trajectory=None,
                            visualization=True,
                            randomization=False)

    observation = env.reset()
    is_done = False
    t = 0
    while t < env.max_episode:
        observation, score, is_done, info = env.step([0.5 + 0.25 / 2, 0.25 / 2, 0.75 + 0.2 / 2,
                                                      0.5, 0.5, 0.5])
        print("eval_score:", score)
        t += 0.001 * env.step_size
        if is_done:
            env.reset()
