import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, loss_type, weight_regularization="none", base_model=None, reg_lambda=0.01, ignore_index = -100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert loss_type in ["sequence", "token", "none"]
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.weight_regularization = weight_regularization
        self.base_model = base_model
        self.reg_lambda = reg_lambda

    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        nll_loss.masked_fill_(padding_mask, 0.0)

        if num_items_in_batch is None:
            num_items_in_batch = (~padding_mask).sum()

        if self.loss_type == "sequence":
            sequence_weights = inputs.pop("sequence_weights")

            weighted_nll_loss = nll_loss.sum(dim=(-2,-1)) * sequence_weights
            loss = weighted_nll_loss.sum() / num_items_in_batch

        elif self.loss_type == "token":
            token_weights = inputs.pop("token_weights")

            token_weights[padding_mask.bool().squeeze(-1)] = 0.0 # This is need to avoid over normalizing the loss

            weighted_nll_loss = nll_loss * token_weights[:, :, None]
            loss = weighted_nll_loss.sum() / num_items_in_batch
        elif self.loss_type == "none": # TODO: Check if this is correct
            loss = nll_loss.sum() / num_items_in_batch
        else:
            raise RuntimeError(f"Unknown loss type {self.loss_type}, please use [sequence/token]")
        
        # If we are in a distributed setting, we need to normalize the loss by the number of processes
        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        
        if self.weight_regularization == "l1":
            l1_loss = torch.tensor(0.0, device=logits.device)
            num_params = 0
            for param, base_param in zip(model.parameters(), self.base_model.parameters()):
                if param.requires_grad:
                    l1_loss += torch.abs(param - base_param).sum()
                    num_params += param.numel()
            loss += (self.reg_lambda * l1_loss) / num_params
        if self.weight_regularization == "l2":
            l2_loss = torch.tensor(0.0, device=logits.device)
            num_params = 0
            for param, base_param in zip(model.parameters(), self.base_model.parameters()):
                if param.requires_grad:
                    l2_loss += ((param - base_param) ** 2).sum()
                    num_params += param.numel()
            loss += (self.reg_lambda * l2_loss) / num_params

        return loss

    # def get_batch_samples(self, epoch_iterator, num_batches): # Overwrite data prefetching to get new num_items_in_batch
    #     batch_samples = []
    #     num_items_in_batch = None
    #     for _ in range(num_batches):
    #         try:
    #             batch_samples += [next(epoch_iterator)]
    #         except StopIteration:
    #             break

    #     if len(batch_samples) > 0 and "labels" in batch_samples[0]:
    #         # For now we don't support object detection
    #         try:
    #             if self.loss_type == "sequence":
    #                 num_items_in_batch = sum([(batch["sequence_weights"]).sum() for batch in batch_samples])
    #             elif self.loss_type == "token":
    #                 # TODO: Zero out token weights for padding tokens in preprocessor (Doubles the training time)
    #                 num_items_in_batch = sum([(batch["token_weights"][batch["labels"][..., 1:].contiguous().ne(-100).squeeze(-1)]).sum() for batch in batch_samples])

    #         except (TypeError, AttributeError):
    #             pass

    #     if self.args.average_tokens_across_devices:
    #         num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

    #     if torch.is_tensor(num_items_in_batch):
    #         num_items_in_batch = num_items_in_batch.item()

    #     return batch_samples, num_items_in_batch

class SFATrainer(Trainer):
    def __init__(self, model_state=None, averaging_freq=0.25, beta=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_state = model_state
        self.averaging_freq = averaging_freq
        self.beta = beta

        # Calculate total steps and averaging interval
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if len(train_dataloader) % self.args.gradient_accumulation_steps != 0:
            num_update_steps_per_epoch += 1
        num_train_epochs = self.args.num_train_epochs
        self.total_steps = int(num_update_steps_per_epoch * num_train_epochs)
        self.averaging_steps = int(self.averaging_freq * self.total_steps)
        self.last_average_step = 0
        print(num_update_steps_per_epoch)
        print(num_train_epochs)
        print(f"Averging every {self.averaging_steps}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Custom training step with synchronized model averaging"""
        step = self.state.global_step

        if (step > 0 and step % self.averaging_steps == 0) and self.base_model_state is not None and step != self.last_average_step:
            current_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            
            # Perform weighted averaging of parameters
            for key in current_state:
                if key in self.base_model_state:
                    current_state[key] = (
                        self.beta * self.base_model_state[key].to(current_state[key].device) + 
                        (1 - self.beta) * current_state[key]
                    )
            
            if isinstance(model, DDP):
                model.module.load_state_dict(current_state)
            else:
                model.load_state_dict(current_state)
            
            if dist.is_initialized():
                dist.barrier()
            
            self.last_average_step = step
            
        # Call parent's training step first
        loss = super().training_step(model, inputs, num_items_in_batch)

        return loss
        