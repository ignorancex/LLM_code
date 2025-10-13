import torch
import math
from .metapruner import MetaPruner
from .scheduler import linear_scheduler
from .. import function
from ..._helpers import _FlattenIndexMapping


class GroupNormPruner(MetaPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-4,
        alpha=4,
        iterative_steps=1,
        iterative_sparsity_scheduler=linear_scheduler,
        ch_sparsity=0.5,
        global_pruning=False,
        channel_groups=dict(),
        max_ch_sparsity=1.0,
        soft_keeping_ratio=0.0,
        ch_sparsity_dict=None,
        round_to=None,
        ignored_layers=None,
        customized_pruners=None,
        unwrapped_parameters=None,
        output_transform=None,
    ):
        super(GroupNormPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            iterative_sparsity_scheduler=iterative_sparsity_scheduler,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            global_pruning=global_pruning,
            channel_groups=channel_groups,
            max_ch_sparsity=max_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            output_transform=output_transform,
        )
        #self.model = model
        #self.example_inputs = example_inputs
        self.reg = reg
        self.alpha = alpha
        self.groups = list(self.DG.get_all_groups())
        self.soft_keeping_ratio = soft_keeping_ratio
        self.cnt = 0
    @torch.no_grad()
    def regularize(self, model, base=16):
        #DG = tp.DependencyGraph().build_dependency(model, example_inputs= self.example_inputs)
        # Group should be updated once we process the pruned one. Torch_Pruning does not cover it.
        self.groups = list(self.DG.get_all_groups())
        print(f"len(self.groups): {len(self.groups)}")
        for i, group in enumerate(self.groups):
            print(f"i: {i}")
            ch_groups = self.get_channel_groups(group)
            group_norm = 0
            # Get group norm
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler

                #if idxs[0] != 0 or idxs[-1] + 1 >= 4:
                #    continue

                # Conv out_channels
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    print(f"[group_norm_pruner.py] idxs: {idxs}")
                    print(f"[group_norm_pruner.py] len(layer.weight.data): {len(layer.weight.data)}")
                    print(f"[group_norm_pruner.py] type(layer.weight.data): {type(layer.weight.data)}")
                    w = layer.weight.data[idxs].flatten(1)
                    print(f"[group_norm_pruner.py] w.shape: {w.shape}")
                    local_norm = w.pow(2).sum(1)
                    print(f"[group_norm_pruner.py - 1] local_norm.shape: {local_norm.shape}")
                    #print(local_norm.shape, layer, idxs, ch_groups)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    print(f"[group_norm_pruner.py - 2] local_norm.shape: {local_norm.shape}")
                    group_norm+=local_norm
                    if layer.bias is not None:
                        group_norm += layer.bias.data[idxs].pow(2)
                # Conv in_channels
                elif prune_fn in [function.prune_conv_in_channels,function.prune_linear_in_channels]:
                    '''
                    # Chicago Added This Part
                    #print(f"[group_norm_pruner.py - First 2-1] (layer.weight).shape: {(layer.weight).shape}")
                    #w = (layer.weight).transpose(0, 1).flatten(1)
                    #print(f"[group_norm_pruner.py - 2-1] w.shape: {w.shape}")
                    print(f"\n\nidxs[-1] + 1: {idxs[-1] + 1} &&& layer.weight.shape[1]: {layer.weight.shape[1]}\n\n\n") # 8, 4
                    print(f"\n\nw.shape: {w.shape} &&& layer.weight.shape: {layer.weight.shape}\n\n\n") # w.shape: torch.Size([8, 72]) &&& layer.weight.shape: torch.Size([8, 4, 2, 2])
                    if (w.shape[0] == layer.weight.shape[1] or idxs[-1] + 1 <= layer.weight.shape[1]):
                    #if (idxs[-1] + 1 <= layer.weight.shape[1]):
                        print(f"[group_norm_pruner.py - First Branch-1] w.shape: {w.shape}")
                        print(f"[group_norm_pruner.py - First Branch-1] (layer.weight).shape: {(layer.weight).shape}")
                        w = (layer.weight).transpose(0, 1).flatten(1)
                        print(f"[group_norm_pruner.py - First Branch-2] w.shape: {w.shape}")
                    #elif (w.shape[0] != layer.weight.shape[1] and w.shape[0] == layer.weight.shape[0] and len(idxs) <= layer.weight.shape[1]):
                    else:
                        print(f"[group_norm_pruner.py - Second Branch-1] w.shape: {w.shape}")
                        print(f"[group_norm_pruner.py - Second Branch-1] (layer.weight).shape: {(layer.weight).shape}")
                        w = (layer.weight).flatten(1)
                        print(f"[group_norm_pruner.py - Second Branch-2] w.shape: {w.shape}")
                    '''
                    w = (layer.weight).transpose(0, 1).flatten(1)

                    if (w.shape[0] != group_norm.shape[0]):
                        print(f"[group_norm_pruner.py - 222-2] w.shape: {w.shape}")
                        print(f"[group_norm_pruner.py - 222-2] group_norm.shape: {group_norm.shape}")
                        print(f"[group_norm_pruner.py - 222-2] idxs: {idxs}")
                        if hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping):
                            # conv - latten
                            w = w.view(
                                group_norm.shape[0],
                                w.shape[0] // group_norm.shape[0],
                                w.shape[1],
                            ).flatten(1)
                            print(f"[group_norm_pruner.py - 2-3] w.shape: {w.shape}")
                        elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                            # group conv
                            w = w.view(w.shape[0] // group_norm.shape[0],
                                    group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                            print(f"[group_norm_pruner.py - 2-4] w.shape: {w.shape}")
                        else:
                            # Chicago Created
                            w = w.view(w.shape, -1)
                            idxs = [i for i in range(w.shape[0])]

                        '''
                        a = w.view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        b = w.view(w.shape[0] // group_norm.shape[0],
                                    group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                        #print(f"[group_norm_pruner.py - 2-5] a.shape: {a.shape}")
                        print(f"[group_norm_pruner.py - 2-5] b.shape: {b.shape}")
                        w = b
                        '''

                    local_norm = w.pow(2).sum(1)
                    print(f"[group_norm_pruner.py - 3] w.shape: {w.shape}")
                    print(f"[group_norm_pruner.py - 3] local_norm.shape: {local_norm.shape}")
                    if ch_groups>1:
                        if len(local_norm)==len(group_norm):
                            local_norm = local_norm.view(ch_groups, -1).sum(0)
                            print(f"[group_norm_pruner.py - 4] local_norm.shape: {local_norm.shape}")
                        local_norm = local_norm.repeat(ch_groups)
                        print(f"[group_norm_pruner.py - 5] local_norm.shape: {local_norm.shape}")
                    print(f"[group_norm_pruner.py] group_norm.shape: {group_norm.shape}")
                    print(f"[group_norm_pruner.py - 6] local_norm.shape: {local_norm.shape}")
                    print(f"[group_norm_pruner.py] local_norm[0].shape: {local_norm[0].shape}")
                    print(f"[group_norm_pruner.py - 6] idxs: {idxs}")
                    group_norm += local_norm[idxs]
                # BN
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine:
                        w = layer.weight.data[idxs]
                        local_norm = w.pow(2)
                        if ch_groups>1:
                            local_norm = local_norm.view(ch_groups, -1).sum(0)
                            local_norm = local_norm.repeat(ch_groups)
                        group_norm += local_norm

                        #b = layer.bias.data[idxs]
                        #local_norm = b.pow(2)
                        #if ch_groups>1:
                        #    local_norm = local_norm.view(ch_groups, -1).sum(0)
                        #    local_norm = local_norm.repeat(ch_groups)
                        #group_norm += local_norm

            current_channels = len(group_norm)
            if ch_groups>1:
                group_norm = group_norm.view(ch_groups, -1).sum(0)
                group_stride = current_channels//ch_groups
                group_norm = torch.cat([group_norm+group_stride*i for i in range(ch_groups)], 0)
            group_norm = group_norm.sqrt()
            if group_norm == None:
                print(f"group_norm is NONE")
            #print(f"group_norm: {group_norm}")
            print(f"type(group_norm): {type(group_norm)}")
            print(f"group_norm.shape: {group_norm.shape}")
            print(f"group_norm.device: {group_norm.device}")
            print(f"group_norm.max(): {group_norm.max()}")
            print(f"group_norm.min(): {group_norm.min()}")
            base = 16
            scale = base**((group_norm.max() - group_norm) / (group_norm.max() - group_norm.min()))
            #if self.cnt%1000==0:
            #    print("="*15)
            #    print(group)
            #    print("Group {}".format(i))
            #    print(group_norm)
            #    print(scale)
            
            # Update Gradient
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    w = layer.weight.data[idxs]
                    g = w * scale.view( -1, *([1]*(len(w.shape)-1)) ) #/ group_norm.view( -1, *([1]*(len(w.shape)-1)) ) * group_size #group_size #* scale.view( -1, *([1]*(len(w.shape)-1)) )
                    layer.weight.grad.data[idxs]+=self.reg * g 
                    #if layer.bias is not None:
                    #    b = layer.bias.data[idxs]
                    #    g = b * scale
                    #    layer.bias.grad.data[idxs]+=self.reg * g 
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    gn = group_norm
                    if hasattr(dep.target, 'index_transform') and isinstance(dep.target.index_transform, _FlattenIndexTransform):
                        gn = group_norm.repeat_interleave(w.shape[1]//group_norm.shape[0])
                    # regularize input channels
                    if prune_fn==function.prune_conv_in_channels and layer.groups>1:
                        scale = scale[:len(idxs)//ch_groups]
                        idxs = idxs[:len(idxs)//ch_groups]
                    print(f"[212] layer.weight.data.shape: {layer.weight.data.shape}")
                    print(f"[213] idxs: {idxs}")
                    w = layer.weight.data[:, idxs]
                    g = w * scale.view( 1, -1, *([1]*(len(w.shape)-2))  ) #/ gn.view( 1, -1, *([1]*(len(w.shape)-2)) ) * group_size #* scale.view( 1, -1, *([1]*(len(w.shape)-2))  )
                    layer.weight.grad.data[:, idxs]+=self.reg * g
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        w = layer.weight.data[idxs]
                        g = w * scale #/ group_norm * group_size
                        layer.weight.grad.data[idxs]+=self.reg * g 

                        #b = layer.bias.data[idxs]
                        #g = b * scale #/ group_norm * group_size
                        #layer.bias.grad.data[idxs]+=self.reg * g 
        self.cnt+=1