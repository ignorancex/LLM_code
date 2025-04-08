from typing import Dict, Optional
import numpy as np

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.env.env_context import EnvContext

from utils.custom_keys import Postprocessing_Custom
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from typing import Dict, Optional, Tuple, Union

import numpy as np

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2



from utils.polytope_loader import load_polytope

class EvaluationLoggerCallback(DefaultCallbacks):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.algo_handler = None
        self.A = None
        super().__init__(*args, **kwargs)


    
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        episode.user_data["constraint_violation_distance"] = []
        episode.user_data["bool_violation"] = []
        episode.user_data["bool_violations"] = []
        
        episode.hist_data["constraint_violation_distance"] = []

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            algorithm: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        #create a handler for the algorithm, to access the logger
        #self.algo_handler = algorithm

        self.path_variable = "UPGRADED"

    
    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """Called immediately after a policy's postprocess_fn is called.
        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.
        Args:
            worker: Reference to the current rollout worker.
            episode: Episode object.
            agent_id: Id of the current agent.
            policy_id: Id of the current policy for the agent.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default_policy".
            postprocessed_batch: The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches: Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        pass


    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs,
    ) -> None:
        """Runs when an episode is done.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            kwargs: Forward compatibility placeholder.
        """

        episode.custom_metrics["constraint_violation_distance"] = np.sum(episode.user_data["constraint_violation_distance"])
        episode.custom_metrics["bool_violation"] = np.sum(episode.user_data["bool_violation"])
        episode.custom_metrics["bool_violations"] = np.sum(episode.user_data["bool_violations"])

        
    def on_train_result(
        self,
        *,
        algorithm: Optional["Algorithm"] = None,
        result: dict,
        trainer=None,
        **kwargs,
    ) -> None:
        """Called at the end of Trainable.train().

        Args:
            algorithm: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        
        if "evaluation" in result:
            constraint_violation_distance = result["evaluation"]["custom_metrics"]["constraint_violation_distance"]
        
            result["evaluation"]["custom_metrics"]["constraint_violation_distance_total"] = np.sum(constraint_violation_distance)
            result["evaluation"]["custom_metrics"]["constraint_violation_distance_mean"] = np.mean(constraint_violation_distance)
            
            result["evaluation"]["custom_metrics"]["bool_violation_total"] = np.sum(result["evaluation"]["custom_metrics"]["bool_violation"])
            result["evaluation"]["custom_metrics"]["bool_violation_mean"] = np.mean(result["evaluation"]["custom_metrics"]["bool_violation"])
            
            result["evaluation"]["custom_metrics"]["bool_violations_total"] = np.sum(result["evaluation"]["custom_metrics"]["bool_violations"])
            result["evaluation"]["custom_metrics"]["bool_violations_mean"] = np.mean(result["evaluation"]["custom_metrics"]["bool_violations"])


            
        if "constraint_violation_distance" in result["custom_metrics"]:
            constraint_violation_distance = result["custom_metrics"]["constraint_violation_distance"]
            
            result["custom_metrics"]["constraint_violation_distance_total"] = np.sum(constraint_violation_distance)
            result["custom_metrics"]["constraint_violation_distance_mean"] = np.mean(constraint_violation_distance)
            
            result["custom_metrics"]["bool_violation_total"] = np.sum(result["custom_metrics"]["bool_violation"])
            result["custom_metrics"]["bool_violation_mean"] = np.mean(result["custom_metrics"]["bool_violation"])
            
            result["custom_metrics"]["bool_violations_total"] = np.sum(result["custom_metrics"]["bool_violations"])
            result["custom_metrics"]["bool_violations_mean"] = np.mean(result["custom_metrics"]["bool_violations"])

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Runs on each episode step.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        env_config = worker.get_policy().config.get("env_config")


        if self.A is None:
            self.n_dimensions = base_env.action_space.shape[0]
        
            A,b = load_polytope(n_dim=self.n_dimensions,
                                storage_method=worker.policy_config["model"]["custom_model_config"]["polytope_storage_method"], 
                                generation_method=worker.policy_config["model"]["custom_model_config"]["polytope_generation_method"], 
                                polytope_generation_data=worker.policy_config["model"]["custom_model_config"]["polytope_generation_data"])
        
            self.A = A
            self.b = b


        last_allocation = episode.last_action_for()

        constraint_violation = np.maximum((np.dot(self.A, last_allocation) - self.b),0)
        constraint_violation_sum = constraint_violation.sum()
        
        bool_violation = (constraint_violation > 1e-3).any() #if any constraint is violated 
        bool_violations = (constraint_violation > 1e-3).sum() #total number of violated constraints
        episode.user_data["constraint_violation_distance"].append(constraint_violation_sum)
        episode.user_data["bool_violation"].append(bool_violation)
        episode.user_data["bool_violations"].append(bool_violations)

        
