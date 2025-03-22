#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-10 22:58
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:19
Description        : Callback for trainer.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import logging
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.trainer_callback import ProgressCallback, TrainerCallback

logger = logging.getLogger('DeepSpeed')


class LoggerCallback(ProgressCallback):
    """
    Callback that logs the training information.
    """

    def __init__(self):
        super().__init__()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is None:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)

            elapsed = self.training_bar.format_dict['elapsed']
            elapsed_str = self.training_bar.format_interval(elapsed)
            
            rate = self.training_bar.format_dict["rate"]
            remaining = (self.training_bar.total - self.training_bar.n
                         ) / rate if rate and self.training_bar.total else 0
            remaining = self.training_bar.format_interval(remaining)
            speed = 1 / rate if rate > 0 else 0
            speed = round(speed, 2)
            
            timing = f"{self.training_bar.n}/{self.training_bar.total},  " + elapsed_str + " < " + remaining + f",  {speed}s/it"
            logs["iter"] = timing
            
            log_str = ""
            for key, value in logs.items():
                if len(log_str) > 0:
                    log_str += f", {key}: {value}"
                else:
                    log_str += f"{key}: {value}"
            
            logger.info(log_str)


class SaveCallback(TrainerCallback):
    """
    Callback that saves the model at a specified time interval.
    """

    def __init__(self):
        super().__init__()
        self.should_save = torch.tensor([0], dtype=torch.int).cuda()

    def on_step_end(self, args, state, control, **kwargs):
        if args.save_strategy == "no":
            return control
        if hasattr(control, "latest"):
            control.latest = False
        else:
            setattr(control, "latest", False)

        if not hasattr(control, "last_time"):
            setattr(control, "last_time", time.time())

        if state.is_world_process_zero:
            current_time = time.time()
            if current_time - control.last_time > args.save_last_interval:
                self.should_save = torch.tensor([1], dtype=torch.int).cuda()
                control.last_time = current_time

        dist.all_reduce(self.should_save, op=dist.ReduceOp.SUM)
        dist.barrier()

        if self.should_save == 1:
            logger.info(
                f"Saving latest checkpoint at {state.global_step} step")
            control.should_save = True
            control.latest = True
            self.should_save = torch.tensor([0], dtype=torch.int).cuda()

        return control


class ProfilerCallback(TrainerCallback):
    """
    Callback that profiles the training speed, etc.
    """

    def __init__(self, prof):
        super().__init__()
        self.prof = prof

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.prof.start()

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.prof.stop()
