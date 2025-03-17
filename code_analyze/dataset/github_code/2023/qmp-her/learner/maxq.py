import numpy as np
import tensorflow as tf
import os
from envs import make_env
from algorithm.replay_buffer import Trajectory, goal_based_process


class MaxQLearner:
    def __init__(self, args):
        self.iter = 0
        self.args = args
        # get current policy from path (restore tf session + graph)
        self.reach_deviations = [[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1], [0,0,0]] #left&right, in front&behind, and top
        self.reach_primitive_path = args.reach_primitive_path
        self.reach_meta_path = os.path.join(self.reach_primitive_path, "saved_policy-best.meta")
        self.reach_sess = tf.Session()
        self.reach_saver = tf.train.import_meta_graph(self.reach_meta_path)
        self.reach_saver.restore(self.reach_sess, tf.train.latest_checkpoint(self.reach_primitive_path))
        reach_graph = tf.get_default_graph()
        self.reach_raw_obs_ph = reach_graph.get_tensor_by_name("raw_obs_ph:0")
        self.reach_pi = reach_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")

    def compute_reach_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.reach_sess.run(self.reach_pi, {self.reach_raw_obs_ph: obs})
        return action

    def preprocess_reach_observations(self, push_obs):
        obs = goal_based_process(push_obs)
        #remove observation related to object state because reach policy cannot handle
        grip_pos = obs[:3]
        grip_state = obs[9:11]
        grip_vels = obs[20:25]
        object_position = obs[3:6]
        goal_position = obs[-3:]
        observations = []
        reach_goal_obs = np.concatenate([grip_pos, grip_state, grip_vels, goal_position])
        observations.append(reach_goal_obs)
        for deviation in self.reach_deviations:
            target_position = object_position + deviation
            observations.append(np.concatenate([grip_pos, grip_state, grip_vels, target_position]))
        observations_2d = np.vstack(observations)
        return observations_2d


    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        for _ in range(args.episodes):
            obs = env.reset()
            current = Trajectory(obs)
            for timestep in range(args.timesteps):
                if np.random.uniform() > self.args.eps_act:
                    reach_primitive_obs = self.preprocess_reach_observations(obs)
                    primitive_actions = []
                    for reach_primitive_ob in reach_primitive_obs:
                        primitive_actions.append(self.compute_reach_primitive_action(np.reshape(reach_primitive_ob, (1, 13))))
                    primitive_actions = np.vstack(primitive_actions)
                    action = agent.step(obs, explore=False)
                    candidate_actions = np.vstack((np.array(action), primitive_actions))
                    qs = []
                    for candidate_action in candidate_actions:
                        qs.append(agent.compute_q(np.reshape(goal_based_process(obs), (1, 28)), np.reshape(candidate_action, (1,4))))
                    qs = np.vstack(qs)
                    best_action_idx = np.argmax(qs)
                    action = candidate_actions[best_action_idx]
                else:
                    action = np.random.uniform(-1, 1, size=self.args.acts_dims)
                obs, reward, done, _ = env.step(action)
                if timestep == args.timesteps - 1: done = True
                current.store_step(action, obs, reward, done)
                if done: break
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    buffer.dis_balance = buffer.dis_balance * 1.00001
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                agent.target_update()