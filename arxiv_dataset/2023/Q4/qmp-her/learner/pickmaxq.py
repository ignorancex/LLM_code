import numpy as np
import tensorflow as tf
import os
from envs import make_env
from algorithm.replay_buffer import Trajectory, goal_based_process


class PickMaxQLearner:
    def __init__(self, args):
        self.args = args
        self.iter = 0
        # get current policy from path (restore tf session + graph)
        self.pick_goal = [0, 0, 0.6] #left&right, in front&behind, and top
        self.reach_primitive_path = args.reach_primitive_path
        self.reach_meta_path = os.path.join(self.reach_primitive_path, "saved_policy-best.meta")
        self.reach_sess = tf.Session()
        self.reach_saver = tf.train.import_meta_graph(self.reach_meta_path)
        self.reach_saver.restore(self.reach_sess, tf.train.latest_checkpoint(self.reach_primitive_path))
        reach_graph = tf.get_default_graph()
        self.reach_raw_obs_ph = reach_graph.get_tensor_by_name("raw_obs_ph:0")
        self.reach_pi = reach_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")
        if args.push_primitive_path:
            self.push_primitive_path = args.push_primitive_path
            self.push_meta_path = os.path.join(self.push_primitive_path, "saved_policy-best.meta")
            self.push_sess = tf.Session()
            self.push_saver = tf.train.import_meta_graph(self.push_meta_path)
            self.push_saver.restore(self.push_sess, tf.train.latest_checkpoint(self.push_primitive_path))
            push_graph = tf.get_default_graph()
            self.push_raw_obs_ph = reach_graph.get_tensor_by_name("raw_obs_ph:0")
            self.push_pi = reach_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")


    def compute_reach_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.reach_sess.run(self.reach_pi, {self.reach_raw_obs_ph: obs})
        return action
    def compute_push_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.push_sess.run(self.reach_pi, {self.reach_raw_obs_ph: obs})
        return action


    def preprocess_pick_obs(self, pick_obs):
        obs = goal_based_process(pick_obs)
        obs[-3:] = self.pick_goal
        return obs

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        for _ in range(args.episodes):
            obs = env.reset()
            current = Trajectory(obs)
            for timestep in range(args.timesteps):
                if np.random.uniform() > self.args.eps_act:
                    reach_primitive_obs = self.preprocess_pick_obs(obs)
                    primitive_actions = []
                    primitive_action = self.compute_reach_primitive_action(np.reshape(reach_primitive_obs, (1, 28)))
                    primitive_actions.append(primitive_action[0])
                    primitive_action = self.compute_reach_primitive_action(np.reshape(goal_based_process(obs), (1, 28)))
                    primitive_actions.append(primitive_action[0])
                    if self.args.push_primitive_path:
                        primitive_action = self.compute_push_primitive_action(np.reshape(goal_based_process(obs), (1, 28)))
                        primitive_actions.append(primitive_action[0])
                    primitive_actions = np.vstack(primitive_actions)
                    target_action = agent.step(obs, explore=False)
                    candidate_actions = np.vstack((np.array(target_action), primitive_actions))
                    qs = []
                    for candidate_action in candidate_actions:
                        qs.append(agent.compute_q(np.reshape(goal_based_process(obs), (1, 28)), np.reshape(candidate_action, (1,4))))
                    qs = np.vstack(qs)
                    action = candidate_actions[np.argmax(qs)]
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