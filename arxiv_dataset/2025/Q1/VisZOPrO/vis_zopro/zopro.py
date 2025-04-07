from collections import defaultdict
from contextlib import contextmanager, nullcontext
import math
import gc
from os import environ, path, makedirs
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD
from trl import RLOOConfig
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding, GenerationConfig
from transformers.optimization import get_scheduler
from datasets import Dataset
from peft import PeftModel
from accelerate import Accelerator
from trl.trainer.utils import disable_dropout_in_model, batch_generation, forward, get_reward, first_true_indices, truncate_response, print_rich_table, log_table_to_comet_experiment
from accelerate.utils import gather_object
import wandb
from tqdm import tqdm
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict
)


INVALID_LOGPROB = 1.0


class PrevDecayScheduler:
    def __init__(self, num_total_steps, initial_value, scheduling="exp"):
        self.num_total_steps = num_total_steps
        self.initial_value = initial_value
        self.current_value = initial_value
        self.scheduling = scheduling
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.scheduling == "exp":
            self.current_value = self.initial_value * torch.exp(torch.tensor(-self.steps))
        elif self.scheduling == "linear":
            self.current_value = self.current_value - 1 / self.num_total_steps
        else:
            raise NotImplementedError("Only 'exp' and 'linear' scheduling are supported.")

    def get_decay(self):
        return self.current_value

    def state_dict(self):
        return {
            "num_total_steps": self.num_total_steps,
            "current_value": self.current_value,
            "steps": self.steps,
            "scheduling": self.scheduling,
            "initial_value": self.initial_value,
        }

    def load_state_dict(self, state_dict):
        self.num_total_steps = state_dict["num_total_steps"]
        self.current_value = state_dict["current_value"]
        self.steps = state_dict["steps"]
        self.scheduling = state_dict["scheduling"]
        self.initial_value = state_dict["initial_value"]


class ZORLOOTrainer():
    def __init__(
        self,
        policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        config: RLOOConfig,
        previous_update,  # update from previous iteration
        reward_delta,  # reward delta from previous and current iteration
        zo_eps: float = 1e-4,
        previous_decay: str = "exp",
        data_collator: DataCollatorWithPadding = None,
        ref_policy: nn.Module = None,
        callbacks: list = None,
        wandb_enabled: bool = True,
    ):
        self.accelerator = Accelerator()
        self.wandb_enabled = wandb_enabled
        self.precompute_orth(previous_update, reward_delta)
        prev_decay = 1 - 1e-3  # do not start with 1 as it would not add any noise
        self.zo_eps = zo_eps
        self.model = policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.config = config
        self.ref_policy = ref_policy
        self.callbacks = callbacks
        if data_collator is None:
            self.data_collator = DataCollatorWithPadding(self.processing_class)

        self.model_adapter_name = config.model_adapter_name
        self.ref_adapter_name = config.ref_adapter_name
        self.is_peft_model = isinstance(self.model, PeftModel)

        # FIXME
        policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        # setup dataloaders
        config.local_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps * config.num_mini_batches
        config.local_mini_batch_size = config.local_batch_size // config.num_mini_batches
        self.local_dataloader_batch_size = config.local_batch_size // config.rloo_k
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(config.seed)
        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.local_seed = config.seed + self.accelerator.process_index * 100003  # Prime

        config.num_total_batches = math.ceil(len(self.train_dataset) * config.num_train_epochs / (config.local_batch_size * self.accelerator.num_processes))
        if config.num_sample_generations > 0:
            self.sample_generations_freq = max(1, config.num_total_batches // config.num_sample_generations)
        torch.manual_seed(self.local_seed)
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        # setup models
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)

        if ref_policy:
            self.ref_policy = ref_policy
        elif self.is_peft_model:
            self.ref_policy = None
        else:
            raise NotImplementedError("Reference policy is required for non-PEFT models. Copying policy as reference policy is not supported atm.")
        self.reward_model = self.reward_model.to(self.accelerator.device)
        self.model = self.model.to(self.accelerator.device)
        if self.ref_policy:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.ref_policy.eval()
        self.model.eval()
        self.reward_model.eval()

        # setup optimizer (not needed) and scheduler
        mock_optim = SGD(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_scheduler(
            config.lr_scheduler_type,
            mock_optim,
            num_warmup_steps=config.get_warmup_steps(config.num_total_batches),
            num_training_steps=config.num_total_batches,
            scheduler_specific_kwargs=config.lr_scheduler_kwargs,
        )
        self.prev_decay_scheduler = PrevDecayScheduler(config.num_total_batches, prev_decay, scheduling=previous_decay)

    def precompute_orth(self, previous_update, reward_delta):
        # precomputes orthogonal projection matrices for each parameter as
        # it is shared across all iterations
        mus = {}  # projections of previous update
        v_norms = {}  # projection matrices used to get orth
        assert len(reward_delta) == len(previous_update)
        for key in previous_update.keys():
            mu = previous_update[key]
            v = reward_delta[key]
            v = v.to(self.accelerator.device)
            mu = mu.to(self.accelerator.device)
            v_flat = v.flatten()
            norm_v = v_flat.norm(p=2)
            v_norm = (v_flat / norm_v).to(mu)

            mus[key] = mu.cpu()
            v_norms[key] = v_norm.cpu()
        self.prevs = mus
        self.v_norms = v_norms

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)  # model should not be wrapped
        with unwrapped_model.disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                unwrapped_model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                unwrapped_model.set_adapter(self.model_adapter_name or "default")

    def zo_perturb_params(self, seed, scaling_factor=1):
        # perturb the parameters of model2train
        torch.manual_seed(seed)
        if self.is_peft_model:
            model_name_params = get_peft_model_state_dict(self.model, adapter_name=self.model_adapter_name).items()
        else:
            model_name_params = [x for x in self.model.named_parameters() if x[1].requires_grad]
        for name, param in model_name_params:
            prev = self.prevs[name].to(param.device)
            v_norm = self.v_norms[name].to(param.device)
            prev_decay = self.prev_decay_scheduler.get_decay()
            noise = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            ).flatten()
            proj_noise = torch.dot(v_norm, noise)
            proj_noise = (noise - proj_noise * v_norm).reshape(param.data.shape)
            # match the norm of prev to that of proj_noise
            proj_noise = proj_noise * (torch.norm(prev) / (torch.norm(proj_noise) + 1e-8))
            noise_scale = torch.sqrt(torch.tensor(1.0 - prev_decay**2, dtype=v_norm.dtype, device=v_norm.device))
            z = prev_decay * prev + noise_scale * proj_noise
            z *= self.zo_eps
            param.data = param.data + scaling_factor * z
        if self.is_peft_model:
            set_peft_model_state_dict(self.model, dict(model_name_params), adapter_name=self.model_adapter_name)

    def save_checkpoint(self, update, zo_random_seed):
        if not self.accelerator.is_main_process:
            return
        model_state = self.model.state_dict() if not self.is_peft_model else get_peft_model_state_dict(self.model, adapter_name=self.model_adapter_name)
        checkpoint = {
            "model": model_state,
            "scheduler": self.scheduler.state_dict(),
            "update": update,
            "prev_decay": self.prev_decay_scheduler.state_dict(),
            "config": self.config,
            "zo_random_seed": zo_random_seed,
            "zo_eps": self.zo_eps,
        }
        if self.wandb_enabled:
            wandb_run_id = wandb.run.id
            checkpoint["wandb_run_id"] = wandb_run_id
        makedirs(self.config.output_dir, exist_ok=True)
        torch.save(checkpoint, path.join(self.config.log_dir, "checkpoint.pt"))

    def load_checkpoint(self, zo_random_seed, total_iterations):
        if not path.exists(path.join(self.config.log_dir, "checkpoint.pt")) or not path.isfile(path.join(self.config.log_dir, "checkpoint.pt")):
            return 0, zo_random_seed, None
        checkpoint = torch.load(path.join(self.config.log_dir, "checkpoint.pt"))
        if checkpoint["update"] >= total_iterations:
            return 0, zo_random_seed, None
        if self.is_peft_model:
            set_peft_model_state_dict(self.model, checkpoint["model"], adapter_name=self.model_adapter_name)
        else:
            self.model.load_state_dict(checkpoint["model"])

        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.prev_decay_scheduler.load_state_dict(checkpoint["prev_decay"])
        self.config = checkpoint["config"]
        self.zo_eps = checkpoint["zo_eps"]
        wandb_run_id = checkpoint.get("wandb_run_id", None)
        return checkpoint["update"], checkpoint["zo_random_seed"], wandb_run_id

    @torch.inference_mode()
    def train(self, random_seed_0=None, resume_if_possible=True):
        config = self.config
        start_time = time.time()
        device = self.accelerator.device
        stats_shape = (config.num_ppo_epochs, config.num_mini_batches, config.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        zo_loss_stats = torch.zeros(stats_shape, device=device)
        zo_random_seed = (
            random_seed_0
            if random_seed_0 is not None
            else np.random.randint(1000000000)
        )

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            temperature=(config.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
        iterator = range(1, config.num_total_batches + 1)
        if self.accelerator.is_main_process:
            iterator = tqdm(iterator)
        # FIXME: bug: if the training is completed then subsequent runs will still load the same checkpoint
        # and NOT train the model. # may be fixed now but I would be careful.
        total_iterations = config.num_total_batches - 1 if resume_if_possible else float("inf")
        start_update, zo_random_seed, wandb_run_id = self.load_checkpoint(zo_random_seed, total_iterations)

        if self.wandb_enabled and self.accelerator.is_main_process:
            if wandb_run_id is not None:
                wandb.init(
                    project="zorloo",
                    name=environ.get("RUN_NAME", config.exp_name),
                    id=wandb_run_id,
                    resume="must"
                )
            else:
                wandb.init(
                    project="zorloo",
                    name=environ.get("RUN_NAME", config.exp_name),
                )
        for update in iterator:
            data = next(iter_dataloader)
            if update <= start_update:
                continue
            queries = data["input_ids"].to(self.accelerator.device)
            queries = queries.repeat(config.rloo_k, 1)
            context_length = queries.shape[1]
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []

            # Generate responses and compute logprobs
            query_responses, logitss = batch_generation(
                self.model,
                queries,
                config.local_rollout_forward_batch_size,
                self.processing_class.pad_token_id,
                generation_config,
            )

            # Process responses in batches
            for i in range(0, queries.shape[0], config.local_rollout_forward_batch_size):
                query = queries[i: i + config.local_rollout_forward_batch_size]
                query_response = query_responses[i: i + config.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]
                logits = logitss[i: i + config.local_rollout_forward_batch_size]
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del logits, all_logprob
                torch.cuda.empty_cache()

                if self.ref_policy is None:
                    with self.null_ref_context():
                        ref_output = forward(self.model, query_response, self.processing_class.pad_token_id)
                else:
                    ref_output = forward(self.ref_policy, query_response, self.processing_class.pad_token_id)
                ref_logits = ref_output.logits[:, context_length-1: -1]
                ref_logits /= config.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if config.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    raise NotImplementedError("There probably is logic in visualisation/trainers that breaks if this is implemented.")
                    postprocessed_response = truncate_response(
                        config.stop_token_id, self.processing_class.pad_token_id, response
                    )

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == self.processing_class.pad_token_id) - 1

                if isinstance(self.reward_model, nn.Module):
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, self.processing_class.pad_token_id, context_length
                    )
                else:
                    score = torch.tensor(
                        self.reward_model(
                            self.processing_class.batch_decode(postprocessed_query_response, skip_special_tokens=True)
                        ),
                        dtype=torch.float,
                    ).to(self.accelerator.device)

                # Store batch results
                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)

            # Concatenate all batched results
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            del (logprob, ref_logprob, score)
            torch.cuda.empty_cache()
            gc.collect()

            # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
            if config.missing_eos_penalty is not None:
                scores[~contain_eos_token] -= self.config.missing_eos_penalty
            # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

            # 4. compute rewards
            # Compute KL divergence
            kl = logprobs - ref_logprobs

            # Normalize rewards
            # if config.normalize_reward:
            #     scores = (scores - scores.mean()) / (scores.std() + 1e-8)
            #     scores = torch.clamp(scores, -config.reward_clip_range, config.reward_clip_range)

            # Compute total reward with KL penalty
            if False:  # config.token_level_kl:
                # Token-level KL penalty: apply KL penalty per token
                kl_reward = -config.kl_coef * kl

                # Get the index of the last non-padded token for each sequence
                eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                last_reward = torch.zeros_like(kl)
                # Ensure scores has correct shape and type
                scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)

                # Combine KL reward and last reward
                non_score_reward = kl_reward.sum(1)  # Keep this for logging
                reward = last_reward + kl_reward
                rlhf_reward = reward.sum(1)  # Sum across sequence length
            else:
                # Sequence-level KL penalty: sum KL across tokens first
                sequence_kl = kl.sum(1)
                non_score_reward = -config.kl_coef * sequence_kl
                rlhf_reward = non_score_reward + scores

            # vectorized RLOO advantages implementation
            rlhf_reward = rlhf_reward.reshape(config.rloo_k, -1)
            baseline = (rlhf_reward.sum(0) - rlhf_reward) / (config.rloo_k - 1)
            advantages = rlhf_reward - baseline
            advantages = advantages.flatten()

            # Normalize advantages
            # if config.normalize_advantage:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(config.num_ppo_epochs):
                b_inds = np.random.permutation(config.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, config.local_batch_size, config.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + config.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    accumulated_grads = 0
                    for micro_batch_start in range(0, config.local_mini_batch_size, config.per_device_train_batch_size):
                        micro_batch_end = micro_batch_start + config.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                        # Get batch data
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        self.zo_perturb_params(zo_random_seed, scaling_factor=1)
                        loss1, stats4metrics = self.get_ppo_loss(
                            mb_advantage,
                            mb_responses,
                            mb_query_responses,
                            mb_logprobs,
                            padding_mask[micro_batch_inds],
                            context_length,
                        )
                        self.zo_perturb_params(zo_random_seed, scaling_factor=-2)
                        loss2, _ = self.get_ppo_loss(
                            mb_advantage,
                            mb_responses,
                            mb_query_responses,
                            mb_logprobs,
                            padding_mask[micro_batch_inds],
                            context_length,
                        )
                        self.zo_perturb_params(zo_random_seed, scaling_factor=1)
                        grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
                        loss = (loss1.item() + loss2.item()) / 2
                        accumulated_grads += grad / config.gradient_accumulation_steps
                        if (gradient_accumulation_idx+1) % config.gradient_accumulation_steps == 0:
                            # if in distributed environment, gather the various grads
                            # and average them before stepping
                            if self.accelerator.num_processes > 1:
                                grad_sync = torch.tensor(accumulated_grads, device=self.accelerator.device)  # Create a copy to avoid in-place modifications during all_reduce
                                dist.all_reduce(grad_sync, op=dist.ReduceOp.AVG)  # AVG for average gradient
                                accumulated_grads = grad_sync.item()
                            self.zo_step(zo_random_seed, accumulated_grads)
                            accumulated_grads = 0

                        pg_clipfrac = (stats4metrics["pg_losses2"] > stats4metrics["pg_losses"]).float().mean()
                        prob_dist = torch.nn.functional.softmax(stats4metrics["logits"], dim=-1)
                        entropy = torch.logsumexp(stats4metrics["logits"], dim=-1) - torch.sum(prob_dist * stats4metrics["logits"], dim=-1)
                        approxkl = 0.5 * (stats4metrics["logprobs_diff"]**2).mean()
                        approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                        pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                            pg_clipfrac
                        )
                        pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = stats4metrics["pg_loss"]
                        entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                        ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = stats4metrics["new_ratio"].mean()
                        zo_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = loss
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    del (
                        loss1, loss2, grad, loss, stats4metrics,
                        pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    torch.cuda.empty_cache()
            # compute metrics
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.mean()
            eps = int(update / (time.time() - start_time))
            metrics = {}
            metrics["eps"] = eps
            metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
            metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
            metrics["objective/non_score_reward"] = (
                self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
            )
            metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
            metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
            metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
            metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
            metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
            metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
            metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
            metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
            metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
            metrics["val/num_eos_tokens"] = (responses == self.processing_class.eos_token_id).sum().item()
            metrics["lr"] = self.scheduler.get_last_lr()[0]
            metrics["prev_decay"] = self.prev_decay_scheduler.get_decay()
            metrics["zo_loss"] = self.accelerator.gather_for_metrics(zo_loss_stats).mean().item()
            metrics["zo_random_seed"] = zo_random_seed
            if self.wandb_enabled and self.accelerator.is_main_process:
                wandb.log(metrics)
            else:
                self.accelerator.print(metrics)
            # self.state.epoch = self.state.episode / (config.rloo_k * self.train_dataset_len)  # used by self.log

            del kl, mean_kl, mean_entropy, scores
            if config.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

            # todo check scheduling is correct (considering gradient accumulation)
            self.scheduler.step()
            self.prev_decay_scheduler.step()
            zo_random_seed += 1
            self.save_checkpoint(update, zo_random_seed)

    def zo_step(self, zo_random_seed, projected_grad):
        """
        Update the parameters with the estimated gradients.
        """
        if torch.isnan(torch.tensor(projected_grad)) or torch.isinf(
            torch.tensor(projected_grad)
        ):
            print("Projected grad is NaN or Inf")
            return

        # Reset the random seed for sampling zs
        torch.manual_seed(zo_random_seed)
        if self.is_peft_model:
            model_name_params = get_peft_model_state_dict(self.model, adapter_name=self.model_adapter_name).items()
        else:
            model_name_params = [x for x in self.model.named_parameters() if x[1].requires_grad]
        for name, param in model_name_params:
            prev = self.prevs[name].to(param.device)
            v_norm = self.v_norms[name].to(param.device)
            prev_decay = self.prev_decay_scheduler.get_decay()
            noise = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            ).flatten()
            proj_noise = torch.dot(v_norm, noise)
            proj_noise = (noise - proj_noise * v_norm).reshape(param.data.shape)
            # match the norm of prev_orth to that of proj_noise
            proj_noise = proj_noise * (torch.norm(prev) / (torch.norm(proj_noise) + 1e-8))
            noise_scale = torch.sqrt(torch.tensor(1.0 - prev_decay**2, dtype=v_norm.dtype, device=v_norm.device))

            z = prev_decay * prev + noise_scale * proj_noise
            # z *= self.zo_eps

            if (
                "bias" not in name
                and "layer_norm" not in name
                and "layernorm" not in name
            ):
                param.data = param.data - self.scheduler.get_last_lr()[0] * (
                    projected_grad * z
                    + self.config.weight_decay * param.data
                )
            else:
                param.data = param.data - self.scheduler.get_last_lr()[0] * (
                    projected_grad * z
                )
        if self.is_peft_model:
            set_peft_model_state_dict(self.model, dict(model_name_params), adapter_name=self.model_adapter_name)

    def get_ppo_loss(self, mb_advantage, mb_responses, mb_query_responses, mb_logprobs, padding_mask, context_length):
        config = self.config

        # Forward pass
        output = forward(self.model, mb_query_responses, self.processing_class.pad_token_id)
        logits = output.logits[:, context_length-1: -1]
        logits /= config.temperature + 1e-7

        # Compute new logprobs
        new_all_logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
        new_logprobs = torch.masked_fill(
            new_logprobs, padding_mask, INVALID_LOGPROB
        )

        # Compute probability ratios
        new_ratio = (new_logprobs - mb_logprobs).exp()
        new_logprobs = new_logprobs.sum(1)
        mb_logprobs = mb_logprobs.sum(1)
        logprobs_diff = new_logprobs - mb_logprobs
        ratio = torch.exp(logprobs_diff)

        # PPO clipped loss
        pg_losses = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - config.cliprange, 1.0 + config.cliprange)
        pg_loss_max = torch.max(pg_losses, pg_losses2)
        pg_loss = pg_loss_max.mean()

        # Final loss
        loss = pg_loss

        return loss, {"pg_losses": pg_losses, "pg_losses2": pg_losses2, "new_ratio": new_ratio, "logits": logits, "logprobs_diff": logprobs_diff, "pg_loss": pg_loss}

    def generate_completions(self, sampling: bool = False):
        args = self.config
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.config.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                query_response, _ = batch_generation(
                    self.model,
                    query,
                    query.shape[0],
                    processing_class.pad_token_id,
                    generation_config,
                )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        args.stop_token_id, processing_class.pad_token_id, response
                    )
                table["query"].extend(
                    gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                )
                table["model response"].extend(
                    gather_object(processing_class.batch_decode(postprocessed_response))
                )

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                if isinstance(self.reward_model, nn.Module):
                    _, score, _ = get_reward(
                        self.reward_model,
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                else:
                    score = torch.tensor(
                        self.reward_model(
                            processing_class.batch_decode(postprocessed_query_response, skip_special_tokens=True)
                        ),
                        dtype=torch.float,
                    ).to(postprocessed_query_response.device)
                table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0: 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )
