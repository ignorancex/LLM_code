import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features, convnext_tiny_7_features
import torch
from torch import Tensor
from util.node import Node
import numpy as np
from collections import defaultdict, OrderedDict
import pdb


class HComPNet(nn.Module):
    def __init__(self,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layers: dict, #nn.Module,
                 num_parent_nodes: int,
                 root: Node
                 ):
        super().__init__()
        self._net = feature_net
        for node_name in add_on_layers:
            setattr(self, '_'+node_name+'_add_on', add_on_layers[node_name])
            setattr(self, '_'+node_name+'_num_protos', add_on_layers[node_name].weight.shape[0])
        self._pool = pool_layer
        self._avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
        for node_name in classification_layers:
            setattr(self, '_'+node_name+'_classification', classification_layers[node_name])
        self._multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        self._softmax = nn.Softmax(dim=1)
        self.root = root

        for node_name in add_on_layers:
            num_prototypes = getattr(self, '_'+node_name+'_num_protos')
            proto_presence = torch.zeros(num_prototypes, 2)
            proto_presence = nn.Parameter(proto_presence, requires_grad=True)
            nn.init.xavier_normal_(proto_presence, gain=1.0)
            setattr(self, '_'+node_name+'_proto_presence', proto_presence)

        self.args = args
        

    def forward(self, xs,  inference=False, apply_overspecificity_mask=False, return_inner_product=False):
        features = self._net(xs) 
        proto_features = {}
        proto_features_softmaxed = {}
        pooled = {}
        out = {}

        proto_layer_input_features = features

        for node in self.root.nodes_with_children():
            proto_features[node.name] = getattr(self, '_'+node.name+'_add_on')(proto_layer_input_features)
            proto_features[node.name] = proto_features[node.name]
            proto_features_softmaxed[node.name] = self._softmax(proto_features[node.name])
            pooled[node.name] = self._pool(proto_features_softmaxed[node.name])

            if inference:
                pooled[node.name] = torch.where(pooled[node.name] < 0.1, 0., pooled[node.name])  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out[node.name] = getattr(self, '_'+node.name+'_classification')(pooled[node.name]) #shape (bs*2, num_classes) # these are logits

        if return_inner_product:
            return features, proto_features, proto_features_softmaxed, pooled, out
        else:
            return features, proto_features_softmaxed, pooled, out
    

    def get_joint_distribution(self, out, leave_out_classes=None, apply_overspecificity_mask=False, device='cuda', softmax_tau=1):
        batch_size = out['root'].size(0)
        top_level = out['root']
        bottom_level = self.root.distribution_over_furthest_descendents(net=self, batch_size=batch_size, out=out, leave_out_classes=leave_out_classes,\
                                                                        apply_overspecificity_mask=apply_overspecificity_mask, device='cuda', softmax_tau=softmax_tau)    
        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)
        bottom_level = bottom_level[:,idx]        

        return top_level, bottom_level
    
    def get_classification_layers(self):
        return [getattr(self, attr) for attr in dir(self) if attr.endswith('_classification')] 


base_architecture_to_features = {'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features,
                                 'convnext_tiny_7': convnext_tiny_7_features}

class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        torch.nn.init.normal_(self.weight, mean=1.0,std=0.1)

        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            torch.nn.init.constant_(self.bias, val=0.)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)
    

class DinoV2(torch.nn.Module):

    def __init__(self, model_name, latent_shape):
        super(DinoV2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).cuda()
        self.latent_shape = latent_shape

    def forward(self, x):
        out = self.model(x, is_training=True)['x_norm_patchtokens']
        out = out.view(out.shape[0], self.latent_shape[0], self.latent_shape[1], out.shape[2])
        out = out.permute(0, 3, 1, 2)
        return out


def get_network(args: argparse.Namespace, root=None): 

    if 'dinov2_vits14' in args.net:
        features = DinoV2(model_name=args.net, latent_shape=(int(args.image_size/14), int(args.image_size/14)))
        first_add_on_layer_in_channels = 384
    else:
        features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
        features_name = str(features).upper()
        if 'next' in args.net:
            features_name = str(args.net).upper()
        if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        else:
            raise Exception('other base architecture NOT implemented')

        
    parent_nodes = root.nodes_with_children()
    add_on_layers = {}
    for node in parent_nodes:
        add_on_layers[node.name] = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=node.num_protos, \
                                            kernel_size=1, stride = 1, padding=0, bias=False) # is bias required ??, prev True now set to False for unit length conv2d
        print(f'Assigned {node.num_protos} protos to node {node.name}')

    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 

    if args.bias:
        print('--------- Using bias in Classification layer ---------')

    classification_layers = {}
    for node in parent_nodes:
        classification_layers[node.name] = NonNegLinear(node.num_protos, node.num_children(), bias=True if args.bias else False)
        
        classification_layers[node.name].weight.requires_grad = False
        start_idx = 0
        # element-wise disabling gradient is not possible
        # so setting the values to negative so during training they will become zero because of relu
        # and once relu becomes zero its gradients also become zero and they wont get trained
        for child_node in node.children:
            child_label = node.children_to_labels[child_node.name]
            end_idx = start_idx + node.num_protos_per_child[child_node.name]
            classification_layers[node.name].weight[child_label, :start_idx] = -0.5
            classification_layers[node.name].weight[child_label, end_idx:] = -0.5
            start_idx = end_idx

        classification_layers[node.name].weight.requires_grad = True        
        
    return features, add_on_layers, pool_layer, classification_layers


    