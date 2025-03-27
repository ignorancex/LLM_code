from common import MockArgs, verify_bias_zero, verify_orthogonal
import torch.nn as nn
import torch
from sparselearning.core import Masking
from orthogonal.tests.test import are_almost_same
from models.initializers import Orthogonal
from sparselearning.weight_preprocesser import SparseOrthogonalPreprocesser
from models.vgg_plain import VGG16
import models.cifar_resnet as cifar_resnet
from setup_utils import get_mask, get_optimizer
from main import get_data


def verify_nonzeros(module: nn.Module, masking: Masking, conv_nonzeros_type: str):
    for name, param in module.named_parameters():
        if name in masking.masks:
            mask = masking.masks[name]
            unpruned = torch.count_nonzero(mask).item()
            nonzeros = torch.count_nonzero(param.data).item()
            if len(param.data.shape) == 2:
                assert unpruned == nonzeros
            elif len(param.data.shape) == 4:
                c_out, c_in, k, _ = param.data.shape
                central_nonzeros = torch.count_nonzero(param.data[:, :, k//2, k//2])
                assert central_nonzeros == nonzeros  # all nonzeros are in the center
                if conv_nonzeros_type == 'less':
                    target_density = unpruned / (c_out * c_in * k * k)
                elif conv_nonzeros_type == 'more':
                    target_density = min(1.0, unpruned / (c_out * c_in * k))
                else:
                    raise NotImplementedError()
                central_density = nonzeros / (c_out * c_in)
                assert are_almost_same(torch.tensor(central_density), torch.tensor(target_density), 0.1)


def verify_init(model: nn.Module):
    model.apply(verify_orthogonal)
    model.apply(verify_bias_zero)


def verify_preprocess(model: nn.Module, masking: Masking, conv_nonzeros_type: str):
    model.apply(verify_orthogonal)
    model.apply(verify_bias_zero)
    verify_nonzeros(model, masking, conv_nonzeros_type)


def test(model_name: str, conv_init: str, nonzeros: str):
    assert model_name in ['cifar_resnet_32', 'cifar_resnet_56', 'cifar_resnet_110', 'vgg-C']
    assert conv_init in ['conv_orthogonal', 'delta_orthogonal']
    assert nonzeros in ['less', 'more']

    print(f'Testing EI {model_name} {conv_init} {nonzeros}')

    if 'cifar_resnet' in model_name:
        model = cifar_resnet.Model.get_model_from_name(model_name, Orthogonal(conv_init_type=conv_init))
    elif model_name == 'vgg-C':
        model = VGG16('C', num_classes=10, init_weights=Orthogonal(conv_init_type=conv_init))

    print(model)

    print('Testing initialization')
    verify_init(model)
    print('Ok')
    
    args = MockArgs(nonzeros)
    optimizer = get_optimizer(args, model)
    _, _, train_loader, _ = get_data(args)
    masking = get_mask(args, model, optimizer, train_loader, 'cpu')

    preprocesser = SparseOrthogonalPreprocesser(model, masking, args, 'cpu')
    preprocesser()

    print('Testing preprocess')
    verify_preprocess(model, masking, nonzeros)
    print('Ok')


if __name__ == '__main__':
    for model_name in ['vgg-C', 'cifar_resnet_32', 'cifar_resnet_56', 'cifar_resnet_110']:
        for conv_init in ['conv_orthogonal', 'delta_orthogonal']:
            for nonzeros in ['less', 'more']:
                test(model_name, conv_init, nonzeros)
    print('All OK')