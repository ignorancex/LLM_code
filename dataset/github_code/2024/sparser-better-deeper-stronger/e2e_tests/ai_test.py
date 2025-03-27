from common import MockArgs, MockLogger, verify_bias_zero, verify_orthogonal
import torch.nn as nn
import torch
from sparselearning.core import Masking
from orthogonal.tests.test import are_almost_same
from models.initializers import Orthogonal, orthogonal_matrix, conv_orthogonal
from sparselearning.weight_preprocesser import ApproximateIsometryPreprocesser
from models.vgg_plain import VGG16
import models.cifar_resnet as cifar_resnet
from setup_utils import get_mask, get_optimizer
from main import get_data


def lower_than_random(loss, tensor, mask):
    if torch.count_nonzero(mask) / torch.numel(mask) > 0.9:
        return True  # skip
    if len(tensor.shape) == 2:
        if tensor.size(1) > tensor.size(0):
            orthogonal_q = orthogonal_matrix(tensor.transpose(0,1)).transpose(0,1)
        else:
            orthogonal_q = orthogonal_matrix(tensor)
        cw = mask * orthogonal_q
    if len(tensor.shape) == 4:
        orthogonal_c = conv_orthogonal(tensor)
        cw = orthogonal_c.view((orthogonal_c.shape[0], -1)) * mask.view((mask.shape[0], -1))
    if cw.shape[0] <= cw.shape[1]:
        mul =  torch.matmul(cw, cw.T)
        rand_loss = torch.norm(mul - torch.eye(mul.shape[0]))
    else:
        mul = torch.matmul(cw.T, cw)
        rand_loss = torch.norm(mul - torch.eye(mul.shape[1]))
    return loss < rand_loss


def verify_ortho_loss(module: nn.Module, masking: Masking):
    for name, param in module.named_parameters():
        if name in masking.masks:
            mask = masking.masks[name]
            if len(param.data.shape) == 2:
                cw = param.data * mask
            elif len(param.data.shape) == 4:
                cw = param.data.view((param.data.shape[0], -1)) * mask.view((mask.shape[0], -1))
            if cw.shape[0] <= cw.shape[1]:
                mul =  torch.matmul(cw, cw.T)
                loss = torch.norm(mul - torch.eye(mul.shape[0]))
            else:
                mul = torch.matmul(cw.T, cw)
                loss = torch.norm(mul - torch.eye(mul.shape[1]))
            assert lower_than_random(loss, param.data, mask)
            

def verify_init(model: nn.Module):
    model.apply(verify_orthogonal)
    model.apply(verify_bias_zero)


def verify_preprocess(model: nn.Module, masking: Masking):
    verify_ortho_loss(model, masking)


def test(model_name: str, conv_init: str):
    assert model_name in ['cifar_resnet_32', 'cifar_resnet_56', 'cifar_resnet_110', 'vgg-C']
    assert conv_init in ['conv_orthogonal', 'delta_orthogonal']

    print(f'Testing AI {model_name} {conv_init}')

    if 'cifar_resnet' in model_name:
        model = cifar_resnet.Model.get_model_from_name(model_name, Orthogonal(conv_init_type=conv_init))
    elif model_name == 'vgg-C':
        model = VGG16('C', num_classes=10, init_weights=Orthogonal(conv_init_type=conv_init))

    print(model)

    print('Testing initialization')
    verify_init(model)
    print('Ok')
    
    args = MockArgs(None)
    optimizer = get_optimizer(args, model)
    _, _, train_loader, _ = get_data(args)
    masking = get_mask(args, model, optimizer, train_loader, 'cpu')

    preprocesser = ApproximateIsometryPreprocesser(model, masking, args, MockLogger(), 'cpu')
    preprocesser()

    print('Testing preprocess')
    verify_preprocess(model, masking)
    print('Ok')


if __name__ == '__main__':
    for model_name in ['cifar_resnet_32', 'cifar_resnet_56', 'cifar_resnet_110', 'vgg-C']:
        for conv_init in ['conv_orthogonal', 'delta_orthogonal']:
            test(model_name, conv_init)
    print('All OK')