#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-15 15:13
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-15 17:35
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

from typing import List

import torch
import torch.nn as nn

@torch.no_grad()
def add_pre_module_hook(module: nn.Module, name: str, cache: List[torch.Tensor], pos: int = None, read: bool = False):
    """Add a pre-hook to a module.

    Parameters
    ----------
    module : nn.Module
        Module to add the hook to.
    name : str
        Name of the cache entry.
    cache : List[torch.Tensor]
        Cache to save the input to.
    pos : int, optional
        Position in the sequence, by default None
    read : bool, optional
        Whether to add a read hook or a write hook, by default False
    """
    
    def read_hook(module, args, kwargs):
        if pos is None:
            cache[name] = kwargs
        else:
            assert 'hidden_states' in kwargs, "Hooked module must have hidden_states"
            cache[name] = kwargs['hidden_states'][:, pos]

    def write_hook(module, args, kwargs):
        if pos is None:
            return cache[name]
        
        assert 'hidden_states' in kwargs, "Hooked module must have hidden_states"
        kwargs['hidden_states'][:, pos] = cache[name]
        return args, kwargs

    if read:
        return module.register_forward_pre_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_pre_hook(write_hook, with_kwargs=True)
    

@torch.no_grad()
def add_pre_module_hook_single_head(module: nn.Module, name: str, cache: List[torch.Tensor], pos: int, head_dim: int, read: bool = False, head_idx: int = None):
    """Add a pre-hook to a module to modify a single head.

    Parameters
    ----------
    module : nn.Module
        Module to add the hook to.
    name : str
        Name of the cache entry.
    cache : List[torch.Tensor]
        Cache to save the input to.
    pos : int
        Position in the sequence.
    head_dim : int
        Head dimension.
    read : bool, optional
        Wether to add a read hook or a write hook, by default False
    head_idx : int, optional
        The index of the head to be modified, by default None
    """

    def read_hook(module, args, kwargs):
        # [batch, seq_len, dim]
        cache[name] = args[0][:, pos]

    def write_hook(module, args, kwargs):
        args[0][:, pos, head_idx * head_dim : (head_idx + 1) * head_dim] = cache[name][:, head_idx * head_dim : (head_idx + 1) * head_dim]
        return args, kwargs

    if read:
        return module.register_forward_pre_hook(read_hook, with_kwargs=True)
    else:
        return module.register_forward_pre_hook(write_hook, with_kwargs=True)
