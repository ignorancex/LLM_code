
import re
from pathlib import Path
from typing import Dict, Literal, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import torch
from torch import nn
from torch import optim
import typer
app = typer.Typer()

class LinearModel(nn.Module):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor):
        return (self.bias + (self.weight * x))

    def __repr__(self):
        res = f'{self.bias.item():.2f}+{self.weight.item():.2f}*x'
        return res

class SigmoidModel(nn.Module):

    def __init__(self, center_init: float=0.0, learnable_params: List[str]=['bias', 'range', 'scale', 'center', 'rate']):
        super(SigmoidModel, self).__init__()
        param_init_dict = {'bias': 0.0, 'range': 1.0, 'scale': 1.0, 'center': center_init, 'rate': 1.0}
        self.params = nn.ParameterDict({learnable_param: nn.Parameter(torch.tensor([param_init_dict[learnable_param]]), requires_grad=(True if (learnable_param in learnable_params) else False)) for learnable_param in param_init_dict})

    def forward(self, x: torch.Tensor):
        (bias, range, scale, center, rate) = [self.params[param] for param in ['bias', 'range', 'scale', 'center', 'rate']]
        return (bias + (range / (1 + (rate * torch.exp(((- (x - center)) * (scale ** 2)))))))

    def __repr__(self):
        (bias, range, scale, center, rate) = [self.params[param] for param in ['bias', 'range', 'scale', 'center', 'rate']]
        res = f'{float(bias):.2f}+{float(range):.2f} / (1+{float(rate):.2f}*exp(-(x-{float(center):.2f})* {(float(scale) ** 2):.2f}))'
        return res

class PolySigmoidModel(nn.Module):

    def __init__(self):
        super(PolySigmoidModel, self).__init__()
        self.bias = nn.Parameter(torch.tensor([0.5]))
        self.power_coef = nn.Parameter(torch.tensor([1.0]))
        self.rate = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor):
        return ((self.bias ** 2) + (((1 - (self.bias ** 2)) * (x ** (self.power_coef ** 2))) / ((x ** (self.power_coef ** 2)) + self.rate)))

    def __repr__(self):
        return f'{(self.bias.item() ** 2):.2f}+{(1 - (self.bias.item() ** 2)):.2f}*x^{(self.power_coef.item() ** 2):.2f}/(x^{(self.power_coef.item() ** 2):.2f}+{self.rate.item():.2f})'

class TheoryExponentModel(nn.Module):

    def __init__(self):
        super(TheoryExponentModel, self).__init__()
        self.M = nn.Parameter(torch.tensor([1.0]))
        self.N = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return ((self.M * torch.exp((- x))) + (self.N ** 2))
        return ((self.invsigma * torch.exp((- x))) + (0.5 * (1 - self.invsigma)))

    def __repr__(self):
        return f'{self.M.item():.2f}*exp(-x) + {(self.N.item() ** 2):.2f}'
        return f'{self.invsigma.item():.2f}*exp(-x) + 0.5*(1-{self.invsigma.item():.2f})'

def val_set_data_process(data_path: Path=Path('data/0529_intelli_val_set.csv'), test_set_size: int=1, filter_pattern: str='') -> Dict[(str, pd.DataFrame)]:
    data = pd.read_csv(data_path, sep='\t')
    bmk_l = ['ARC', 'HellaSwag', 'MMLU', 'Winogrande']
    bmk_cols = [col for col in data.columns if any([(bmk in col) for bmk in bmk_l])]
    for bmk in bmk_cols:
        data[bmk] = (data[bmk] / 100)
        data[f'{bmk}_err'] = (1 - data[bmk])
    data['average_bmk'] = data[bmk_cols].mean(axis=1)
    data['normalized_error_rate'] = (1 - data['average_bmk'])

    def get_model_family(model: str) -> str:
        if ('step' in model):
            res = 'step'
        else:
            res = '-'.join(model.split('-')[:(- 1)])
        return res
    data['model_family'] = data['model'].apply(get_model_family)
    data['compression_rate'] = (1 / data['bpc'])
    data['normalized_compression_rate'] = ((1 / data['bpc']) - 1)
    data['normalized_acc'] = (1 - data['normalized_error_rate'])
    data['log_normalized_acc'] = np.log(data['normalized_acc'])
    data['log_normalized_error_rate'] = np.log(data['normalized_error_rate'])
    res = {'train_data': None, 'test_data': None, 'filter_data': None}
    if (len(filter_pattern) > 0):
        pattern = re.compile(filter_pattern)
        filter_mask = data['model'].apply((lambda x: bool(pattern.search(x))))
        res['filter_data'] = data[filter_mask]
        data = data[(~ filter_mask)]
    data = data.sort_values('bpc', ascending=False)
    res['test_data'] = data.iloc[(- test_set_size):]
    res['train_data'] = data.iloc[:(- test_set_size)]
    return res

def fit_model(data: pd.DataFrame, x_key: str, y_key: str, model_type: Literal[('linear', 'sigmoid')], fitting_args: Dict[(str, Any)]={'lr': 0.1, 'optim': 'adam'}) -> nn.Module:
    x = torch.tensor(data[x_key].values).float()
    y = torch.tensor(data[y_key].values).float()
    if (model_type == 'linear'):
        model = LinearModel()
    elif (model_type == 'sigmoid'):
        learnable_params = fitting_args['learnable_params']
        model = SigmoidModel(center_init=x.mean().item(), learnable_params=learnable_params)
    elif (model_type == 'polysigmoid'):
        model = PolySigmoidModel()
    elif (model_type == 'theory_exponent'):
        model = TheoryExponentModel()
    else:
        raise ValueError(f'model_type {model_type} not supported')
    criterion = nn.MSELoss()
    optim_2_cls = {'adam': optim.Adam, 'sgd': optim.SGD}
    optimizer = optim_2_cls[fitting_args['optim']](model.parameters(), lr=fitting_args['lr'])
    print(f'optimizer: {optimizer}')
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if ((epoch % 100) == 0):
            print(f'epoch {epoch}: loss {loss.item()}')
    return model

def plot_fitted_curve(ax: Axes, model: nn.Module, x_fit: np.ndarray, text_yloc_offset: float=0.1, **plot_kwargs):
    with torch.no_grad():
        y_fit = model(torch.tensor(x_fit).float())
    fontsize = plot_kwargs.pop('fontsize', 10)
    ax.plot(x_fit, y_fit, **plot_kwargs)
    ax.text(x_fit.min(), (y_fit.max() + text_yloc_offset), f'y = {model}', fontsize=fontsize, color=plot_kwargs['color'])
    return ax
if (__name__ == '__main__'):
    app()
