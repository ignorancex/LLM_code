import numpy as np
import tensorflow as tf
import os
from envs import make_env
from algorithm.replay_buffer import Trajectory, goal_based_process


class HandMaxQLearner:
    def __init__(self, args):
        self.args = args
        self.iter = 0
        # get current policy from path (restore tf session + graph)
        self.pick_goal = [0, 0, 0.6] #left&right, in front&behind, and top
        if args.handX_primitive_path:
            self.handX_primitive_path = args.handX_primitive_path
            self.handX_meta_path = os.path.join(self.handX_primitive_path, "saved_policy-best.meta")
            self.handX_sess = tf.Session()
            self.handX_saver = tf.train.import_meta_graph(self.handX_meta_path)
            self.handX_saver.restore(self.handX_sess, tf.train.latest_checkpoint(self.handX_primitive_path))
            handX_graph = tf.get_default_graph()
            self.handX_raw_obs_ph = handX_graph.get_tensor_by_name("raw_obs_ph:0")
            self.handX_pi = handX_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")
        if args.handY_primitive_path:
            self.handY_primitive_path = args.handY_primitive_path
            self.handY_meta_path = os.path.join(self.handY_primitive_path, "saved_policy-best.meta")
            self.handY_sess = tf.Session()
            self.handY_saver = tf.train.import_meta_graph(self.handY_meta_path)
            self.handY_saver.restore(self.handY_sess, tf.train.latest_checkpoint(self.handY_primitive_path))
            handY_graph = tf.get_default_graph()
            self.handY_raw_obs_ph = handY_graph.get_tensor_by_name("raw_obs_ph:0")
            self.handY_pi = handY_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")
        if args.handZ_primitive_path:
            self.handZ_primitive_path = args.handZ_primitive_path
            self.handZ_meta_path = os.path.join(self.handZ_primitive_path, "saved_policy-best.meta")
            self.handZ_sess = tf.Session()
            self.handZ_saver = tf.train.import_meta_graph(self.handZ_meta_path)
            self.handZ_saver.restore(self.handZ_sess, tf.train.latest_checkpoint(self.handZ_primitive_path))
            handZ_graph = tf.get_default_graph()
            self.handZ_raw_obs_ph = handZ_graph.get_tensor_by_name("raw_obs_ph:0")
            self.handZ_pi = handZ_graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")


    def compute_handX_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.handX_sess.run(self.handX_pi, {self.handX_raw_obs_ph: obs})
        return action
    def compute_handY_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.handY_sess.run(self.handY_pi, {self.handY_raw_obs_ph: obs})
        return action
    def compute_handZ_primitive_action(self, obs):
        # compute action from obs based on current policy by running tf session initialized before
        action = self.handZ_sess.run(self.handZ_pi, {self.handZ_raw_obs_ph: obs})
        return action

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        for _ in range(args.episodes):
            obs = env.reset()
            current = Trajectory(obs)
            for timestep in range(args.timesteps):
                if np.random.uniform() > self.args.eps_act:
                    primitive_actions = []
                    primitive_action = self.compute_handX_primitive_action(np.reshape(goal_based_process(obs), (1, 65)))
                    primitive_actions.append(primitive_action[0])
                    primitive_action = self.compute_handY_primitive_action(np.reshape(goal_based_process(obs), (1, 65)))
                    primitive_actions.append(primitive_action[0])
                    primitive_action = self.compute_handZ_primitive_action(np.reshape(goal_based_process(obs), (1, 65)))
                    primitive_actions.append(primitive_action[0])
                    primitive_actions = np.vstack(primitive_actions)
                    target_action = agent.step(obs, explore=False)
                    candidate_actions = np.vstack((np.array(target_action), primitive_actions))
                    qs = []
                    for candidate_action in candidate_actions:
                        qs.append(agent.compute_q(np.reshape(goal_based_process(obs), (1, 65)), np.reshape(candidate_action, (1,20))))
                    qs = np.vstack(qs)
                    action = candidate_actions[np.argmax(qs)]
                else:
                    action = np.random.uniform(-1, 1, size=self.args.acts_dims)
                obs, reward, done, _ = env.step(action)
                if timestep == args.timesteps - 1: done = True
                current.store_step(action, obs, reward, done)
                if done: break
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.lazier_and_goals_sample_kg())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    buffer.dis_balance = buffer.dis_balance * 1.00001
                    info = agent.train(buffer.lazier_and_goals_sample_kg())
                    args.logger.add_dict(info)
                agent.target_update()
