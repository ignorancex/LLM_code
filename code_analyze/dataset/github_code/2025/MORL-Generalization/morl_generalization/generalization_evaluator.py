from typing import Dict, Optional, Tuple, List, Union
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import _find_spec
import mo_gymnasium as mo_gym
import wandb
import time

from mo_utils.pareto import filter_pareto_dominated
from mo_utils.performance_indicators import (
    cardinality,
    expected_utility,
    hypervolume,
    sparsity,
)
from mo_utils.weights import equally_spaced_weights
from morl_generalization.utils import make_test_envs
from morl_generalization.wrappers import MOAsyncVectorEnv
from experiments.evaluation import get_minmax_values


class MORLGeneralizationEvaluator(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
            self, 
            env: gym.Env,
            algo_name: str,
            seed: int,
            test_envs: List[str],
            algo_suffix: str = "",
            async_envs: bool = True,
            record_video: bool = False,
            record_video_w_freq: Optional[int] = None,
            record_video_ep_freq: Optional[int] = None,
            num_eval_weights: int = 100,
            num_eval_episodes: int = 5,
            fixed_weights: List[List[float]] = None,
            save_weights: bool = False,
            save_metric: str = 'hypervolume',
            normalization: bool = True,
            recover_single_objective: bool = True,
            **kwargs
        ):
        """Wrapper records generalization evaluation metrics for multi-objective reinforcement learning algorithms.

        Args:
            env: The environment that will be wrapped
            algo_name: Name of the algorithm
            seed: Seed for reproducibility
            generalization_algo: Generalization algorithm used for choosing training environment configurations (currently only 'domain_randomization' is implemented)
            test_envs: List of test environments to evaluate the agent on
            algo_suffix: Suffix to add to the algorithm name for logging
            record_video: Whether to record videos of the evaluation
            record_video_w_freq: Number of evaluated weights frequency of recording videos. Consider your `((num_timesteps / eval_mo_freq) * num_eval_weights * num_eval_episodes) % record_video_freq`.
            record_video_ep_freq: Episodic frequency of recording videos (preferably high number, if agent keeps dying, vectorised test environments will reset, resulting in more frequent video recordings)
            num_eval_weights: Number of weights to evaluate the agent on (for LS methods to condition on and for EUM calculation)
            num_eval_episodes: Number of episodes to average over for policy evaluation for each weight (total episodes = num_eval_weights * num_eval_episodes)
            save_weights: Whether to save the best weights for each test environment
            save_metric: Metrics to save the best front (and weights if `save_weights` is set) for
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, 
            algo_name=algo_name, 
            algo_suffix=algo_suffix,
            seed=seed, 
            test_envs=test_envs, 
            record_video=record_video, 
            record_video_w_freq=record_video_w_freq,
            record_video_ep_freq=record_video_ep_freq, 
            save_metrics=save_metric, 
            normalization=normalization,
            recover_single_objective=recover_single_objective,
            **kwargs
        )
        super().__init__(env)
        self.algo_name = algo_name + algo_suffix

        # ============ Evaluation Parameters ============
        self.test_env_names = test_envs
        gym_specs = [_find_spec(env_name) for env_name in test_envs]
        make_fn = [
            lambda env_spec=env_spec: make_test_envs(
                env_spec, 
                self.algo_name, 
                seed,
                record_video=record_video,
                record_video_w_freq=record_video_w_freq,
                record_video_ep_freq=record_video_ep_freq,
                **kwargs
            ) for env_spec in gym_specs
        ]

        if async_envs:
            self.test_envs = MOAsyncVectorEnv(make_fn, copy=False)
        else:
            self.test_envs = mo_gym.wrappers.vector.MOSyncVectorEnv(make_fn)

        if fixed_weights:
            self.eval_weights = [np.array(w) for w in fixed_weights]
            self.num_eval_weights = len(self.eval_weights)
        else:
            self.num_eval_weights = num_eval_weights
            self.eval_weights = equally_spaced_weights(self.unwrapped.reward_dim, self.num_eval_weights)

        self.reward_dim = env.unwrapped.reward_space.shape[0]
        self.num_eval_episodes = num_eval_episodes
        self.normalization = normalization # whether to calculate normalised results
        self.recover_single_objective = recover_single_objective # whether to log single-objective rewards

        if self.normalization:
            self.minmax_ranges = get_minmax_values(env.unwrapped.spec.id)
            for test_env in self.test_env_names:
                assert test_env in self.minmax_ranges, f"Minmax range for {test_env} not found in eval params."
            print("Including normalized metrics in evaluation.")
        if self.recover_single_objective:
            # should only be True if env provides `info['original_scalar_reward']` in `step` function
            print("Plotting single-objective rewards. Please make sure the environment provides `info['original_scalar_reward']` in the `step` function.")
            self.best_single_objective_weights = [
                wandb.Table(
                    columns=["global_step"] + [f"objective_{j}" for j in range(1, self.reward_dim + 1)],
                    data=[],
                ) for _ in test_envs
            ]
            self.best_disc_single_objective_weights = [
                wandb.Table(
                    columns=["global_step"] + [f"objective_{j}" for j in range(1, self.reward_dim + 1)],
                    data=[],
                ) for _ in test_envs
            ]

        # ============ Weights Saving ============
        self.save_weights = save_weights
        self.save_metric = save_metric
        self.best_metrics = -np.inf * np.ones(len(test_envs))
        self.seed = seed 

    def eval_mo(
        self,
        agent,
        w: np.ndarray,
        return_original_scalar=False # when `recover_single_objective` is set to True in the evaluation parameters and the environment's `step` function provides `info['original_scalar_reward']`.
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:
        """Evaluates one episode of the agent in the vectorised test environments.

        Args:
            agent: Agent
            scalarization: scalarization function, taking weights and reward as parameters
            w (np.ndarray): Weight vector

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return. 
            Each is an array where each element corresponds to a sub-environment in the vectorized environment.
        """
        obs, _ = self.test_envs.reset(options={"weights": w, "step":agent.global_step}) # pass in the weight as an option in case video recording is enabled
        done = np.array([False] * self.test_envs.num_envs)
        vec_return = np.zeros((self.test_envs.num_envs, len(w)))
        disc_vec_return = np.zeros_like(vec_return)
        original_return = None
        disc_original_return = None
        if return_original_scalar:
            original_return = np.zeros(self.test_envs.num_envs)
            disc_original_return = np.zeros(self.test_envs.num_envs)
        gamma = np.ones(self.test_envs.num_envs)
        mask = np.ones(self.test_envs.num_envs, dtype=bool)

        if self.algo_name == 'pcn':
            orig_desired_return, orig_desired_horizon = agent.desired_return.copy(), agent.desired_horizon.copy()

        actions = None
        while not all(done):
            actions = agent.eval(
                            obs, 
                            np.tile(w, (self.test_envs.num_envs, 1)), 
                            num_envs = self.test_envs.num_envs,
                            disc_vec_return = disc_vec_return, # used for ESR only
                            prev_actions = actions, # used for recurrent agents only
                        )
            obs, r, terminated, truncated, info = self.test_envs.step(actions)

            if return_original_scalar:
                if 'original_scalar_reward' in info:
                    original_return[mask] += info['original_scalar_reward'][mask]
                    disc_original_return[mask] += gamma[mask] * info['original_scalar_reward'][mask]
                
                if 'final_info' in info: # done step in vectorized env moves info to final_info
                    final_info = info['final_info']
                    for i, finfo in enumerate(final_info):
                        if mask[i] and finfo and 'original_scalar_reward' in finfo:
                            original_return[i] += finfo['original_scalar_reward']
                            disc_original_return[i] += gamma[i] * finfo['original_scalar_reward']
                    
            vec_return[mask] += r[mask]
            disc_vec_return[mask] += gamma[mask, None] * r[mask]
            gamma[mask] *= agent.gamma

            mask &= ~terminated  # Update the mask
            done |= np.logical_or(terminated, truncated)

            if self.algo_name == 'pcn':
                agent.readjust_desired_return_and_horizon(r)

        if self.algo_name == 'pcn': # reset the desired return and horizon to the original values for next repetition
            agent.set_desired_return_and_horizon(orig_desired_return, orig_desired_horizon)

        return (vec_return, disc_vec_return, original_return, disc_original_return)


    def policy_evaluation_mo(
        self, agent, w: Optional[np.ndarray], rep: int = 5, return_original_scalar=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

        Args:
            agent: Agent
            w (np.ndarray): Weight vector
            scalarization: scalarization function, taking reward and weight as parameters
            rep (int, optional): Number of episodes for averaging. Defaults to 5.

        Returns:
            (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return
        """
        evals = [self.eval_mo(agent=agent, w=w, return_original_scalar=return_original_scalar) for _ in range(rep)]
        avg_vec_return = np.mean([eval[0] for eval in evals], axis=0)
        avg_disc_vec_return = np.mean([eval[1] for eval in evals], axis=0)
        avg_original_return = np.mean([eval[2] for eval in evals], axis=0) if evals[0][2] is not None else None
        avg_disc_original_return = np.mean([eval[3] for eval in evals], axis=0) if evals[0][3] is not None else None

        return (
            avg_vec_return,
            avg_disc_vec_return,
            avg_original_return,
            avg_disc_original_return
        )
    
    def log_all_multi_policy_metrics(
        self,
        agent,
        current_fronts: np.ndarray,
        hv_ref_point: np.ndarray,
        reward_dim: int,
        global_step: int,
        idstr: str = "",
        log_metrics: List[str] = ['hypervolume', 'sparsity', 'eum', 'cardinality']
    ):
        """Logs all metrics for multi-policy training (one for each test environment).

        Logged metrics:
        - hypervolume
        - sparsity
        - expected utility metric (EUM)
        - cardinality

        Args:
            current_fronts: List of current Pareto front approximations, has shape of (num_test_envs, num_eval_weights, num_objectives)
            hv_ref_point: reference point for hypervolume computation
            reward_dim: number of objectives
            global_step: global step for logging
            ref_front: reference front, if known
            idstr: for identifying MOO metrics of different types, e.g. "discounted_", "normalised_"
        """
        for i, current_front in enumerate(current_fronts):
            filtered_front = list(filter_pareto_dominated(current_front))
            hv = hypervolume(hv_ref_point, filtered_front)
            sp = sparsity(filtered_front)
            eum = expected_utility(filtered_front, weights_set=self.eval_weights)
            card = cardinality(filtered_front)

            metrics = {
                'hypervolume': hv,
                'sparsity': sp,
                'eum': eum,
                'cardinality': card
            }

            metrics_to_log = {}

            for metric in log_metrics:
                if metric in metrics.keys():
                    metrics_to_log[f"eval/{idstr}{metric}/{self.test_env_names[i]}"] = metrics[metric]
            
            metrics_to_log["global_step"] = global_step
            wandb.log(metrics_to_log, commit=False)

            front = wandb.Table(
                columns=[f"objective_{j}" for j in range(1, reward_dim + 1)],
                data=[p.tolist() for p in filtered_front],
            )
            wandb.log({f"eval/{idstr}front/{self.test_env_names[i]}": front})

            if metrics[self.save_metric] > self.best_metrics[i]:
                self.best_metrics[i] = metrics[self.save_metric]
                best_front = wandb.Table(
                    columns=[f"objective_{j}" for j in range(1, reward_dim + 1)],
                    data=[p.tolist() for p in filtered_front],
                )
                wandb.log({f"eval/best_{self.save_metric}_front/{self.test_env_names[i]}": best_front})
                if self.save_weights:
                    agent.save(
                        save_dir=f"weights/{self.algo_name}/best_{self.save_metric}/seed{self.seed}/{self.test_env_names[i]}",
                        filename=f"{self.test_env_names[i]}", 
                        save_replay_buffer=False
                    )


    def _report(
        self,
        vec_return: np.ndarray,
        disc_vec_return: np.ndarray,
        global_step: int
    ):
        """Logs the evaluation metrics.

        Args:
            scalarized_return: scalarized return
            scalarized_discounted_return: scalarized discounted return
            vec_return: vectorized return
            disc_vec_return: vectorized discounted return
        """
        for i, returns in enumerate(vec_return):
            metrics = {
                "global_step": global_step
            }
            for j in range(returns.shape[0]):
                metrics.update({
                    f"eval/vec_{j}/{self.test_env_names[i]}": vec_return[i][j],
                    f"eval/discounted_vec_{j}/{self.test_env_names[i]}": disc_vec_return[i][j]
                })
            wandb.log(metrics)

    def get_normalized_vec_returns(self, all_vec_returns, minmax_range):
        minmax_array = np.array([minmax_range[str(i)] for i in range(all_vec_returns.shape[-1])])
        min_vals = minmax_array[:, 0].reshape(1, 1, -1) # reshape to (1, 1, n_objectives) for broadcasting
        max_vals = minmax_array[:, 1].reshape(1, 1, -1)

        clipped_vec_returns = np.clip(all_vec_returns, min_vals, max_vals) # broadcasted clipping
        
        # Normalize
        normalized_vec_returns = (clipped_vec_returns - min_vals) / (max_vals - min_vals)
        
        return normalized_vec_returns

    def eval(self, agent, ref_point, global_step, **kwargs):
        print('Evaluating agent on test environments at step: ', global_step)
        start_time = time.time()

        vec_returns = []
        disc_vec_returns = []
        original_scalar_returns = []
        disc_original_scalar_returns = []

        if self.algo_name == 'pcn':
            n = min(len(self.eval_weights), len(agent.experience_replay))
            episodes = agent._nlargest(n)
            desired_returns, desired_horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
            desired_returns = np.float32(desired_returns)
            desired_horizons = np.float32(desired_horizons)

            # for fair comparison, repeat the returns and horizons to match the number of eval_weights
            if n < len(self.eval_weights):
                repeat_factor = int(np.ceil(len(self.eval_weights) / n))
                desired_returns = np.repeat(desired_returns, repeat_factor, axis=0)[:len(self.eval_weights)]
                desired_horizons = np.repeat(desired_horizons, repeat_factor, axis=0)[:len(self.eval_weights)]
            
            desired_horizons = np.expand_dims(desired_horizons, axis=-1)
            desired_horizons = np.tile(desired_horizons, (self.test_envs.num_envs, 1, 1))
            desired_returns = np.tile(desired_returns, (self.test_envs.num_envs, 1, 1))

        for i, ew in enumerate(self.eval_weights):
            if self.algo_name == 'pcn':
                agent.set_desired_return_and_horizon(desired_returns[:, i], desired_horizons[:, i])
            
            (
                vec_return, 
                disc_vec_return,
                original_scalar_return,
                disc_original_scalar_return
            ) = self.policy_evaluation_mo(agent, ew, rep=self.num_eval_episodes, return_original_scalar=self.recover_single_objective)
            vec_returns.append(vec_return)
            disc_vec_returns.append(disc_vec_return)
            if self.recover_single_objective:
                original_scalar_returns.append(original_scalar_return)
                disc_original_scalar_returns.append(disc_original_scalar_return)

        mean_vec_returns = np.mean(vec_returns, axis=0)
        mean_disc_vec_returns = np.mean(disc_vec_returns, axis=0)
        
        # Recover single-objective reward
        # Ideally, we want to evaluate using the exact weight used by the single-objective environment. However, each MORL algorithm 
        # interprets the scale of the weights differently (especially if there's adaptive normalisation) so it's hard to recover the exact weight.
        # So, best compromise would be to use the max original scalar reward across multiple evaluation/weights.
        if self.recover_single_objective:
            original_scalar_returns = np.stack(original_scalar_returns, axis=1)
            disc_original_scalar_returns = np.stack(disc_original_scalar_returns, axis=1)
            # Calculate the index of the maximum single-objective return for each environment
            max_original_indices = np.argmax(original_scalar_returns, axis=1)
            max_disc_original_indices = np.argmax(disc_original_scalar_returns, axis=1)
            
            max_original_scalar_returns = np.max(original_scalar_returns, axis=1)
            max_disc_original_scalar_returns = np.max(disc_original_scalar_returns, axis=1)
            for i in range(len(max_original_scalar_returns)):
                best_weight = self.eval_weights[max_original_indices[i]]
                best_disc_weight = self.eval_weights[max_disc_original_indices[i]]

                metrics = {
                    f"eval/single_objective_return/{self.test_env_names[i]}": max_original_scalar_returns[i],
                    f"eval/single_objective_discounted_return/{self.test_env_names[i]}": max_disc_original_scalar_returns[i],
                    "global_step": global_step
                }
                wandb.log(metrics)

                new_best_weight = [global_step] + best_weight.tolist()
                new_best_disc_weight = [global_step] + best_disc_weight.tolist()
                self.best_single_objective_weights[i].add_data(*new_best_weight)
                self.best_disc_single_objective_weights[i].add_data(*new_best_disc_weight)

                # workaround: have to duplicate the table in order to update it on wandb api
                # see https://github.com/wandb/wandb/issues/2981
                new_table = wandb.Table(
                    columns=self.best_single_objective_weights[i].columns, data=self.best_single_objective_weights[i].data
                )
                new_disc_table = wandb.Table(
                    columns=self.best_disc_single_objective_weights[i].columns, data=self.best_disc_single_objective_weights[i].data
                )
                wandb.log({f"eval/best_single_objective_weights/{self.test_env_names[i]}": new_table})
                wandb.log({f"eval/best_discounted_single_objective_weights/{self.test_env_names[i]}": new_disc_table})

        self._report(
            mean_vec_returns,
            mean_disc_vec_returns,
            global_step=global_step
        )

        vec_returns = np.stack(vec_returns, axis=1)
        disc_vec_returns = np.stack(disc_vec_returns, axis=1)
        
        # Undiscounted front
        for i, current_front in enumerate(vec_returns):
            filtered_front = list(filter_pareto_dominated(current_front))
            front = wandb.Table(
                columns=[f"objective_{j}" for j in range(1, self.reward_dim + 1)],
                data=[p.tolist() for p in filtered_front],
            )
            wandb.log({f"eval/front/{self.test_env_names[i]}": front})

        # Discounted MOO metrics
        self.log_all_multi_policy_metrics(
            agent=agent,
            current_fronts=disc_vec_returns,
            hv_ref_point=ref_point,
            reward_dim=self.reward_dim,
            global_step=global_step,
            idstr="discounted_"
        )

        # Normalized MOO metrics
        if self.normalization:
            # currently only normalizing using discounted vec returns, current minmax ranges cannot be applied to undiscounted returns!
            normalized_returns = np.empty_like(disc_vec_returns)
            for env_idx, env in enumerate(self.test_env_names):
                minmax_range = self.minmax_ranges[env]
                disc_vec_return_for_env = disc_vec_returns[env_idx]
                normalized_returns_for_env = self.get_normalized_vec_returns(disc_vec_return_for_env, minmax_range)
                normalized_returns[env_idx] = normalized_returns_for_env

            self.log_all_multi_policy_metrics(
                agent=agent,
                current_fronts=normalized_returns,
                hv_ref_point=np.zeros(self.reward_dim), # use origin as reference point for normalised metrics
                reward_dim=self.reward_dim,
                global_step=global_step,
                idstr="normalized_",
                log_metrics=["hypervolume", "eum"]
            )

        print(f"Time taken to complete evaluation: {(time.time() - start_time):.2f} seconds")

def make_generalization_evaluator(env, args) -> MORLGeneralizationEvaluator:
    env = MORLGeneralizationEvaluator(
        env,
        algo_name=args.algo,
        seed=args.seed, 
        test_envs=args.test_envs, 
        record_video=args.record_video,
        record_video_ep_freq=args.record_video_ep_freq,
        record_video_w_freq=args.record_video_w_freq,
        **args.generalization_hyperparams
    )
    return env