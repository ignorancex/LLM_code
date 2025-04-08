#adapted from https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/algorithms/on_policy/second_order/cpo.py

from typing import Callable, Dict, List, Type, Union
import numpy as np

import ray
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
)

from torch.utils.data import DataLoader, TensorDataset

from torch.nn.utils.clip_grad import clip_grad_norm_

from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.utils.metrics import NUM_AGENT_STEPS_TRAINED

from ray.rllib.utils.typing import TensorType

from utils.polytope_loader import load_polytope

import torch 


from utils.cost_postprocessing import compute_costs_cpo, compute_gae_for_sample_batch_single


class CostNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def cost(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.cost_value_function()[0].item()
            
            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def cost(*args, **kwargs):
                return 0.0
            
            def value(*args, **kwargs):
                return 0.0

        self._value = value
        self._cost = cost

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Defines extra fetches per action computation.

        Args:
            input_dict (Dict[str, TensorType]): The input dict used for the action
                computing forward pass.
            state_batches (List[TensorType]): List of state tensors (empty for
                non-RNNs).
            model (ModelV2): The Model object of the Policy.
            action_dist: The instantiated distribution
                object, resulting from the model's outputs and the given
                distribution class.

        Returns:
            Dict[str, TensorType]: Dict with extra tf fetches to perform per
                action computation.
        """
        # Return value function outputs. VF estimates will hence be added to
        # the SampleBatches produced by the sampler(s) to generate the train
        # batches going into the loss function.
        return {
            "VF_PREDS_COSTS": model.cost_value_function(),
            SampleBatch.VF_PREDS: model.value_function(),
        }



def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    """This function is used to set the parameters to the model.

    .. note::
        Some algorithms (e.g. TRPO, CPO, etc.) need to set the parameters to the model, instead of
        using the ``optimizer.step()``.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> set_param_values_to_model(model, vals)
        >>> model.weight.data
        tensor([[1., 2.],
                [3., 4.]])

    Args:
        model (torch.nn.Module): The model to be set.
        vals (torch.Tensor): The parameters to be set.

    Raises:
        AssertionError: If the instance of the parameters is not ``torch.Tensor``, or the lengths of
            the parameters and the model parameters do not match.
    """
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'

def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Implementation of Conjugate gradient algorithm.

    Conjugate gradient algorithm is used to solve the linear system of equations :math:`A x = b`.
    The algorithm is described in detail in the paper `Conjugate Gradient Method`_.

    .. _Conjugate Gradient Method: https://en.wikipedia.org/wiki/Conjugate_gradient_method

    .. note::
        Increasing ``num_steps`` will lead to a more accurate approximation to :math:`A^{-1} b`, and
        possibly slightly-improved performance, but at the cost of slowing things down. Also
        probably don't play with this hyperparameter.

    Args:
        fisher_product (Callable[[torch.Tensor], torch.Tensor]): Fisher information matrix vector
            product.
        vector_b (torch.Tensor): The vector :math:`b` in the equation :math:`A x = b`.
        num_steps (int, optional): The number of steps to run the algorithm for. Defaults to 10.
        residual_tol (float, optional): The tolerance for the residual. Defaults to 1e-10.
        eps (float, optional): A small number to avoid dividing by zero. Defaults to 1e-6.

    Returns:
        The vector x in the equation Ax=b.
    """
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x

def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened parameters from the model.

    .. note::
        Some algorithms need to get the flattened parameters from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the parameters are flattened
        and then used to calculate the loss.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> get_flat_params_from(model)
        tensor([1., 2., 3., 4.])

    Args:
        model (torch.nn.Module): model to be flattened.

    Returns:
        Flattened parameters.

    Raises:
        AssertionError: If no gradients were found in model parameters.
    """
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, 'No gradients were found in model parameters.'
    return torch.cat(flat_params)


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    """This function is used to get the flattened gradients from the model.

    .. note::
        Some algorithms need to get the flattened gradients from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the gradients are flattened
        and then used to calculate the loss.

    Args:
        model (torch.nn.Module): The model to be flattened.

    Returns:
        Flattened gradients.

    Raises:
        AssertionError: If no gradients were found in model parameters.
    """
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, 'No gradients were found in model parameters.'
    return torch.cat(grads)





class CPOTorchPolicy(
    CostNetworkMixin,
#    LearningRateSchedule,
    EntropyCoeffSchedule,
    TorchPolicyV2,
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        #validate_config(config)
        n_dimensions = action_space.shape[0]
        self.target_kl = config["target_kl"] #0.01
        self.cg_damping = config["cg_damping"] #0.1
        self.cg_iterations = config["cg_iterations"] #15
        self.cost_limit = config["cost_limit"] #0.0 (0.0 maybe unstable)
        
        self.num_critic_update_iter = config["num_critic_update_iter"] #10
        self.use_critic_norm = config["use_critic_norm"]
        self.critic_norm_coef = config["critic_norm_coef"] #0.001
        
        self.use_max_grad_norm = config["use_max_grad_norm"] #max_grad_norm for the critic
        self.max_grad_norm = config["max_grad_norm"] #max_grad_norm for the critic

        self.total_steps_line_search = config["total_steps_line_search"] #20

        self.norm_grads = config["norm_grads"] #False
        self.bootstrap_costs = config["bootstrap_costs"] #False
        
        self.mini_batch_size_critic = config["mini_batch_size_critic"] #128

        A,b = load_polytope(n_dim=n_dimensions,
                            storage_method=config["model"]["custom_model_config"]["polytope_storage_method"], 
                            generation_method=config["model"]["custom_model_config"]["polytope_generation_method"], 
                            polytope_generation_data=config["model"]["custom_model_config"]["polytope_generation_data"])
        
        self.A = A
        self.b = b

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        
        CostNetworkMixin.__init__(self, config)
        #LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )

        self.optimizer_value = torch.optim.Adam(self.model._critic.parameters(), lr=config["lr_value_critic"])
        self.optimizer_cost = torch.optim.Adam(self.model._critic_costs.parameters(), lr=config["lr_cost_critic"])


        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()


    def _fvp(self, params: torch.Tensor) -> torch.Tensor:
        """Build the Hessian-vector product.

        Build the `Hessian-vector product <https://en.wikipedia.org/wiki/Hessian_matrix>`_ , which
        is the second-order derivative of the KL-divergence.

        The Hessian-vector product is approximated by the Fisher information matrix, which is the
        second-order derivative of the KL-divergence.

        For details see `John Schulman's PhD thesis (pp. 40) <http://joschu.net/docs/thesis.pdf>`_ .

        Args:
            params (torch.Tensor): The parameters of the actor network.

        Returns:
            The Fisher vector product.
        """
        self.model._actor.zero_grad()
        q_dist = self.model._actor(self.model._last_flat_in)
        q_dist = self.dist_class(q_dist, self.model)
        with torch.no_grad():
            p_dist = self.model._actor(self.model._last_flat_in)
            p_dist = self.dist_class(p_dist, self.model)
        kl = torch.distributions.kl.kl_divergence(p_dist.dist, q_dist.dist).mean()

        grads = torch.autograd.grad(
            kl,
            tuple(self.model._actor.parameters()),
            create_graph=True,
        )
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(
            kl_p,
            tuple(self.model._actor.parameters()),
            retain_graph=False,
        )

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        flat_grad_grad_kl = flat_grad_grad_kl.detach()

        # self._logger.store(
        #     {
        #         'Train/KL': kl.item(),
        #     },
        # )
        return flat_grad_grad_kl + params * self.cg_damping

    def _determine_case(
        self,
        b_grads: torch.Tensor,
        ep_costs: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Determine the case of the trust region update.

        Args:
            b_grad (torch.Tensor): Gradient of the cost function.
            ep_costs (torch.Tensor): Cost of the current episode.
            q (torch.Tensor): The quadratic term of the quadratic approximation of the cost function.
            r (torch.Tensor): The linear term of the quadratic approximation of the cost function.
            s (torch.Tensor): The constant term of the quadratic approximation of the cost function.

        Returns:
            optim_case: The case of the trust region update.
            A: The quadratic term of the quadratic approximation of the cost function.
            B: The linear term of the quadratic approximation of the cost function.
        """
        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), 'r is not finite'
            assert torch.isfinite(s).all(), 's is not finite'

            A = q - r**2 / (s + 1e-8)
            B = 2 * self.target_kl - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif ep_costs < 0 <= B:
                # point in trust region is feasible but safety boundary intersects
                # ==> only part of trust region is feasible
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                # point in trust region is infeasible and cost boundary doesn't intersect
                # ==> entire trust region is infeasible
                optim_case = 1
                #print('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety half space is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                #print('Alert! Attempting infeasible recovery!', 'red')

        return optim_case, A, B

    
    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        ep_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if optim_case in (3, 4):
            # under 3 and 4 cases directly use TRPO method
            alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(data: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
                """Project data to [low, high] interval."""
                return torch.clamp(data, low, high)

            #  analytical Solution to LQCLP, employ lambda,nu to compute final solution of OLOLQC
            #  λ=argmax(f_a(λ),f_b(λ)) = λa_star or λb_star
            #  computing formula shown in appendix, lambda_a and lambda_b
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self.target_kl))
            # λa_star = Proj(lambda_a ,0 ~ r/c)  λb_star=Proj(lambda_b,r/c~ +inf)
            # where projection(str,b,c)=max(b,min(str,c))
            # may be regarded as a projection from effective region towards safety region
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(lambda_a, torch.as_tensor(0.0), r_num / eps_cost)
                lambda_b_star = project(lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf))
            else:
                lambda_a_star = project(lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf))
                lambda_b_star = project(lambda_b, torch.as_tensor(0.0), r_num / eps_cost)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )

            # discard all negative values with torch.clamp(x, min=0)
            # Nu_star = (lambda_star * - r)/s
            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)
            # final x_star as final direction played as policy's loss to backward and update
            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:  # case == 0
            # purely decrease costs
            # without further check
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * self.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        return step_direction, lambda_star, nu_star

    
    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        train_batch,
        dist_class,
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        total_steps: int = 15,
        decay: float = 0.8,
        violation_c: int = 0,
        optim_case: int = 0,
    ) -> tuple[torch.Tensor, int]:
        r"""Use line-search to find the step size that satisfies the constraint.

        CPO uses line-search to find the step size that satisfies the constraint. The constraint is
        defined as:

        .. math::

            J^C (\theta + \alpha \delta) - J^C (\theta) \leq \max \{ 0, c \} \\
            D_{KL} (\pi_{\theta} (\cdot|s) || \pi_{\theta + \alpha \delta} (\cdot|s)) \leq \delta_{KL}

        where :math:`\delta_{KL}` is the constraint of KL divergence, :math:`\alpha` is the step size,
        :math:`c` is the violation of constraint.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
            violation_c (int, optional): The violation of constraint. Defaults to 0.
            optim_case (int, optional): The optimization case. Defaults to 0.

        Returns:
            A tuple of final step direction and the size of acceptance steps.
        """
        # get distance each time theta goes towards certain direction
        step_frac = 1.0
        # get and flatten parameters from pi-net
        theta_old = get_flat_params_from(self.model._actor)
        # reward improvement, g-flat as gradient of reward
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        # while not within_trust_region and not finish all steps:
        for step in range(total_steps):
            # get new theta
            new_theta = theta_old + step_frac * step_direction
            # set new theta as new actor parameters
            set_param_values_to_model(self.model._actor, new_theta)
            # the last acceptance steps to next step
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    # loss of policy reward from target/expected reward
                    loss_reward = self._loss_pi(train_batch=train_batch, dist_class=dist_class, adv=train_batch[Postprocessing.ADVANTAGES])
                except ValueError:
                    step_frac *= decay
                    continue
                # loss of cost of policy cost from real/expected reward
                loss_cost = self._loss_pi_cost(train_batch=train_batch, dist_class=dist_class, adv=train_batch["ADVANTAGES_COSTS"])
                # compute KL distance between new and old policy
                #q_dist, _ = self.model(train_batch)
                q_dist = self.model._actor(self.model._last_flat_in)
                q_dist = dist_class(q_dist, self.model)
                kl = torch.distributions.kl.kl_divergence(p_dist.dist, q_dist.dist).mean()
            # compute improvement of reward
            loss_reward_improve = loss_reward_before - loss_reward
            # compute difference of cost
            loss_cost_diff = loss_cost - loss_cost_before

            # average across MPI processes...
            #kl = kl.mean()
            # pi_average of torch_kl above
            loss_reward_improve = loss_reward_improve.mean()
            loss_cost_diff = loss_cost_diff.mean()
            # print(
            #     f'Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}',
            # )
            # check whether there are nan.
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                # print('WARNING: loss_pi not finite')
                pass 
            if not torch.isfinite(kl):
                # print('WARNING: KL not finite')
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                # print('INFO: did not improve improve <0')
                pass
            # change of cost's range
            elif loss_cost_diff > max(-violation_c, 0):
                # print(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
                pass
            # check KL-distance to avoid too far gap
            elif kl > self.target_kl:
                # print(f'INFO: violated KL constraint {kl} at step {step + 1}.')
                pass
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                # print(f'Accept step at i={step + 1}')
                break
            step_frac *= decay
        else:
            # if didn't find a step satisfy those conditions
            # print('INFO: no suitable step found...') 
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0


        set_param_values_to_model(self.model._actor, theta_old)
        return step_frac * step_direction, acceptance_step, {"KL": kl.item()}
    
    def _loss_pi(self, train_batch, dist_class, adv):
        logits, state = self.model(train_batch)
        curr_action_dist = dist_class(logits, self.model)
        
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        loss = -(logp_ratio * adv).mean()
        
        return loss
    
    def _loss_pi_cost(self, train_batch, dist_class, adv):
        logits, state = self.model(train_batch)
        curr_action_dist = dist_class(logits, self.model)
        
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        loss = (logp_ratio * adv).mean()
        
        return loss

    
    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        
        #because we need to set view requirements, make some dummy loss that will never be actually used after init
        
        logits, state = self.model(train_batch)
        curr_action_dist = self.dist_class(logits, self.model)
        policy_loss_reward = self._loss_pi(train_batch, self.dist_class, train_batch[Postprocessing.ADVANTAGES])
        policy_loss_reward = self._loss_pi_cost(train_batch, self.dist_class, train_batch["ADVANTAGES_COSTS"])
        a = train_batch[Postprocessing.VALUE_TARGETS]
        b = train_batch["VALUE_TARGETS_COSTS"]
        c = train_batch["EPSIODE_COSTS"]
        return policy_loss_reward * 0
        
        
    def learn_on_batch(self, postprocessed_batch: SampleBatch) -> Dict[str, TensorType]:

        # Set Model to train mode.
        if self.model:
            self.model.train()
        # Callback handling.
        learn_stats = {}
        self.callbacks.on_learn_on_batch(
            policy=self, train_batch=postprocessed_batch, result=learn_stats
        )
        grad_info = {}

        postprocessed_batch.set_training(True)
        self._lazy_tensor_dict(postprocessed_batch, device=self.devices[0])

        train_batch = postprocessed_batch
        
        theta_old = get_flat_params_from(self.model._actor)
        
        self.model._actor.zero_grad()
        
        logits, state = self.model(train_batch)
        curr_action_dist = self.dist_class(logits, self.model)
        
        p_dist = curr_action_dist

        policy_loss_reward = self._loss_pi(train_batch, self.dist_class, train_batch[Postprocessing.ADVANTAGES])

        policy_loss_reward_before = policy_loss_reward.detach().clone()
        
        policy_loss_reward.backward()
        
        grads = -get_flat_gradients_from(self.model._actor)
        
        if self.norm_grads:
            #norm grads
            grads = grads / (torch.norm(grads) + 1e-8)
        
        x = conjugate_gradients(self._fvp, grads, num_steps=self.cg_iterations)
        
        if self.norm_grads:
            #norm x
            x = x / (torch.norm(x) + 1e-8)
        
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        
        
        self.model._actor.zero_grad()
        
        loss_costs = self._loss_pi_cost(train_batch, self.dist_class, train_batch["ADVANTAGES_COSTS"])
        loss_costs_before = loss_costs.detach().clone()
    
        loss_costs.backward()        
        
        b_grads = get_flat_gradients_from(self.model._actor)
        ep_costs = train_batch["EPSIODE_COSTS"].mean().item() - self.cost_limit
        if self.norm_grads:
            #norm grads
            b_grads = b_grads / (torch.norm(b_grads) + 1e-8)

        p = conjugate_gradients(self._fvp, b_grads, num_steps=self.cg_iterations)
        
        if self.norm_grads:
            #norm p
            p = p / (torch.norm(p) + 1e-8)
        
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)
        
        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )
        
        
        step_direction, accept_step, cpo_search_step_info  = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            train_batch=train_batch,
            dist_class=self.dist_class,
            loss_reward_before=policy_loss_reward_before,
            loss_cost_before=loss_costs_before,
            total_steps=self.total_steps_line_search,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        grad_info.update(cpo_search_step_info)
        
        theta_new = theta_old + step_direction
        set_param_values_to_model(self.model._actor, theta_new)
        
        dataloader = DataLoader(
            dataset=TensorDataset(train_batch[SampleBatch.OBS], train_batch[Postprocessing.VALUE_TARGETS], train_batch["VALUE_TARGETS_COSTS"]),
            batch_size=self.mini_batch_size_critic,
            shuffle=True,
        )

        for _ in range(self.num_critic_update_iter):
            for (
                obs,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self.optimizer_value.zero_grad()
                # Compute a value function loss.
                value_fn_out = self.model._critic(obs).squeeze(-1)
                vf_loss = torch.nn.functional.mse_loss(value_fn_out, target_value_r)
                
                if self.use_critic_norm:
                    for param in self.model._critic.parameters():
                        vf_loss += param.pow(2).sum() * self.critic_norm_coef
                
                vf_loss.backward()
                
                if self.use_max_grad_norm:
                    clip_grad_norm_(self.model._critic.parameters(), self.max_grad_norm)
                
                self.optimizer_value.step()
                
                
                self.optimizer_cost.zero_grad()
                # Compute a cost function loss.
                value_fn_out = self.model._critic_costs(obs).squeeze(-1)
                cost_vf_loss = torch.nn.functional.mse_loss(value_fn_out, target_value_c)
                
                if self.use_critic_norm:
                    for param in self.model._critic_costs.parameters():
                        cost_vf_loss += param.pow(2).sum() * self.critic_norm_coef

                cost_vf_loss.backward()
                
                if self.use_max_grad_norm:
                    clip_grad_norm_(self.model._critic_costs.parameters(), self.max_grad_norm)
                
                self.optimizer_cost.step()
        
        
        with torch.no_grad():
            loss_costs = self._loss_pi_cost(train_batch, self.dist_class, train_batch["ADVANTAGES_COSTS"])
            loss_policy = self._loss_pi(train_batch, self.dist_class, train_batch[Postprocessing.ADVANTAGES])
            
            loss = loss_policy + loss_costs
            
            grad_info["LossPi"] = loss_policy.item()
            grad_info["LossCost"] = loss_costs.item()
            grad_info["Total"] = loss.item()
        
        grad_info["entropy"] = curr_action_dist.entropy().mean().item()
        
        grad_info["LossPi"] = policy_loss_reward.item()
        grad_info["LossCost"] = loss_costs.item()
        grad_info["CostFunctionLoss"] = cost_vf_loss.item()
        grad_info["ValueFunctionLoss"] = vf_loss.item()
        grad_info["EpisodeCosts"] = ep_costs
        
        grad_info["AcceptanceStep"] = accept_step
        grad_info["Alpha"] = alpha.item()
        grad_info["FinalStepNorm"] = step_direction.norm().mean().item()
        grad_info["xHx"]  = xHx.mean().item()
        grad_info["H_inv_g"] = x.norm().item()
        grad_info["gradient_norm"] = torch.norm(grads).mean().item()
        grad_info["cost_gradient_norm"] = torch.norm(b_grads).mean().item()
        grad_info["Lambda_star"] = lambda_star.item()
        grad_info["Nu_star"] = nu_star.item()
        grad_info["OptimCase"] = int(optim_case)
        grad_info["A"] = A.item()
        grad_info["B"] = B.item()
        grad_info["q"] = q.item()
        grad_info["r"] = r.item()
        grad_info["s"] = s.item()
        
        
        grad_info.update(self.stats_fn(postprocessed_batch))

        fetches = self.extra_compute_grad_fetches()

        fetches = dict(fetches, **{LEARNER_STATS_KEY: grad_info})

        
        if self.model:
            fetches["model"] = self.model.metrics()
        
        
        fetches.update(
            {
                "custom_metrics": learn_stats,
                NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count,
            }
        )

        return fetches

        
    def optimizer(
        self,
    ) -> Union[List["torch.optim.Optimizer"], "torch.optim.Optimizer"]:
        """Custom the local PyTorch optimizer(s) to use.

        Returns:
            The local PyTorch optimizer(s) to use for this Policy.
        """
        
        #do not actually use this optimizer!
        if hasattr(self, "config"):
            optimizers = [
                torch.optim.Adam(self.model.parameters(), lr=0.0)
            ]
        else:
            optimizers = [torch.optim.Adam(self.model.parameters(), lr=0.0)]
        if getattr(self, "exploration", None):
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        return optimizers



    

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
            }
        )



    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?

        with torch.no_grad():
            sample_batch = compute_costs_cpo(
                self, sample_batch, other_agent_batches, episode, self.bootstrap_costs
            )

            sample_batch = compute_gae_for_sample_batch_single(
                self, sample_batch, other_agent_batches, episode
            )

            sample_batch = compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )
            return sample_batch
