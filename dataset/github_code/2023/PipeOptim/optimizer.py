# Copyright (c) Microsoft Corporation.
#           (c) I-Ching Tseng
# Licensed under the MIT license.

import torch.optim
import time
import math
from collections import deque  # Efficient ring buffer implementation.
#from spectrain import SpectrainOptimizer
import spectrain
# import spectrain_chc


class Version:

    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version + 1)


class OptimizerWithWeightPrediction(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self,
                 optim_name,
                 modules,
                 master_parameters,
                 model_parameters,
                 loss_scale,
                 num_versions,
                 verbose_freq=0,
                 partitions=1,
                 **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters
        self.model_parameters = model_parameters  # model_parameters is None if not fp16.
        self.loss_scale = loss_scale
        self.optim_name = optim_name

        self.num_versions = num_versions

        self.base_optimizer = getattr(torch.optim,
                                      optim_name)(master_parameters,
                                                  **optimizer_args)

        self.latest_version = Version()
        self.current_version = Version()
        self.initialize_queue()
        self.verbose_freq = verbose_freq
        self.batch_counter = 0

        self.update_interval = partitions
        # self.update_interval = 1
        print("update interval", self.update_interval)


    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        self.queue = deque(maxlen=self.num_versions)
        for i in range(self.num_versions):
            self.queue.append(self.get_params(clone=True))
        self.buffered_state_dicts = self.queue[0][0]

    def get_params1(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params1(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # Don't update running_mean and running_var; these should
                # accumulate normally.
                # mask might have a different shape, so don't copy it to
                # the module this way.
                if "running_" in key or "mask" in key:
                    state_dict[key] = cur_state_dict[key]
            #module.load_state_dict(state_dict)
            for key in state_dict:
                module.state_dict()[key].data.copy_(state_dict[key].data)

            # Load the mask.
            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    # NOTE: Do we need to clone here?
                    #attribute = state_dict[key]
                    attribute.data.copy_(state_dict[key].data)
        self.current_version = version

    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[
                            key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(
                            state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # Don't update running_mean and running_var; these should
                # accumulate normally.
                # mask might have a different shape, so don't copy it to
                # the module this way.
                if "running_" in key or "mask" in key:
                    state_dict[key] = cur_state_dict[key]
            module.load_state_dict(state_dict)

            # Load the mask.
            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    # NOTE: Do we need to clone here?
                    attribute = state_dict[key]
        self.current_version = version

    '''for sgd'''
    def sgd_load_predicted_weights(self, forward=True):
        # pass
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # forward_s = rank // 2 + world_size - rank - 1
        # backward_s = rank // 2
        forward_s = world_size - rank
        backward_s = 0
        version_diff = forward_s if forward else backward_s

        for group in self.param_groups:
            # print(group)
            momentum = group['momentum']
            dampening = group['dampening']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                ''''''
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state:
                        buf = param_state['momentum_buffer']
                        d_p = buf.mul(momentum).add(1-dampening, d_p)

                p.data.add_(-group['lr'], torch.mul(version_diff, d_p))

    '''for rmsprop'''
    def rmsprop_load_predicted_weights(self, forward=True):
        # pass
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # forward_s = rank // 2 + world_size - rank - 1
        # backward_s = rank // 2
        forward_s = world_size - rank
        backward_s = 0
        version_diff = forward_s if forward else backward_s

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                # d_p = p.grad.data
                param_state = self.state[p]
                square_avg = param_state['square_avg']
                denom = square_avg.sqrt().add_(group['eps'])
                step_size = group['lr']
                p.data.addcdiv_(p.grad, denom, value=-step_size * version_diff)

    '''for adam and adamw'''
    def load_predicted_weights(self, forward=True):
        # pass
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        # forward_s = rank // 2 + world_size - rank - 1
        # backward_s = rank // 2
        forward_s = world_size - rank
        backward_s = 0
        version_diff = forward_s if forward else backward_s

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            # print("beta1", beta1, beta2)
            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad.data
                amsgrad = group['amsgrad']
                param_state = self.state[p]
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)

                exp_avg_buf, exp_avg_sq_buf = param_state['exp_avg'], param_state['exp_avg_sq']
                # '''
                if param_state['step'] != 0:
                    bias_correction1 = 1 - beta1 ** param_state['step']
                    bias_correction2 = 1 - beta2 ** param_state['step']
                else:
                    bias_correction1 = 1 - beta1
                    bias_correction2 = 1 - beta2

                eps = 1e-8
                denom = (exp_avg_sq_buf.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg_buf, denom, value=-step_size * version_diff)


    def load_old_params(self):
        # if self.num_versions > 1:
        self.set_params(*self.queue[0])

    # def load_new_params(self):
    #     if self.num_versions > 1:
    #         self.set_params(*self.queue[-1])

    def zero_grad(self):
        if self.batch_counter % self.update_interval == 0:
            self.base_optimizer.zero_grad()

    def step(self, closure=None, s=None, find_median=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # print(f'[rank: {torch.distributed.get_rank()}] optimizer.py line 339')
        # Update the gradient every `update_interval` steps.
        if self.batch_counter % self.update_interval != self.update_interval - 1:
            self.batch_counter += 1
            return None

        log_timing = self.verbose_freq > 0 and self.batch_counter % self.verbose_freq == 0
        if log_timing:
            start_time = time.time()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.model_grads_to_master_grads(self.model_parameters,
                                                   self.master_parameters)
            # TODO: This division might not be in the right place, given that
            # scaling happens right after. Look into this if problems arise.
            if self.loss_scale != 1.0:
                for parameter in self.master_parameters:
                    parameter.grad.data = parameter.grad.data / self.loss_scale

        for p in self.param_groups[0]['params']:
            if p.grad is not None:
                p.grad.div_(self.update_interval)

        # assert
        # if self.optim_name == 'SpectrainCHC':
        # print('Correct optim name!')
        loss = self.base_optimizer.step()
        # else:
        #     print('Error')
        # raise Exception('Wrong optim name!')

        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.master_params_to_model_params(self.model_parameters,
                                                     self.master_parameters)
        self.latest_version = self.latest_version.incr()
        # if self.num_versions > 1:
        self.buffered_state_dicts = self.queue[0][0]
        self.queue.append(self.get_params(clone=False))

        if log_timing:
            print("Optimizer step took: %.3f" % (time.time() - start_time))
        self.batch_counter += 1
        return loss