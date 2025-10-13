import argparse
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from utils import *

from tsai.models.FCN import FCN
from tsai.models.RNN import RNN
from tsai.models.MLP import MLP
from tsai.models.InceptionTime import InceptionTime

def get_real_datasets_attr(model_type, dataset, xai_name, attr_class=None):
    model_path = f'./models/{model_type}/{dataset}'
    train_x, test_x, train_y, test_y, enc1 = read_UCR_UEA(dataset, None)

    test_y = np.argmax(test_y, axis=1)
    train_y = np.argmax(train_y, axis=1)
    num_class = len(np.unique(train_y))
    if model_type == 'RNN':
        model = RNN(c_in=1, c_out=num_class)
    elif model_type == 'FCN':
        model = FCN(c_in=1, c_out=num_class)
    elif model_type == 'InceptionTime':
        model = InceptionTime(c_in=1, c_out=num_class)
    elif model_type == 'MLP':
        model = MLP(c_in=1, c_out=num_class, seq_len=train_x.shape[-1])
    else:
        raise ValueError('Wrong model pick')
    state_dict = torch.load(f'{model_path}/weight.pt', map_location='cuda:1')
    model.load_state_dict(state_dict)
    model.eval()
    # test_y_pred = np.load(f'{model_path}/test_preds.npy')
    # train_y_pred = np.load(f'{model_path}/train_preds.npy')

    attr_save_dir = f'attributions/{model_type}/{dataset}/{xai_name}/' if (attr_class is None) \
        else f'attributions/{dataset}_class_{str(attr_class)}/{xai_name}/'

    attr_gp, _ = get_attr(model, test_x, None, None,
                          save_dir=os.path.join(attr_save_dir, 'test_exp.pkl'),
                          xai_name=xai_name, target_class=attr_class)

    attr_gp, _ = get_attr(model, train_x, None, None,
                          save_dir=os.path.join(attr_save_dir, 'train_exp.pkl'),
                          xai_name=xai_name, target_class=attr_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify datasets')

    parser.add_argument('--dataset_name', type=str, default=None, help='dataset_name')
    parser.add_argument('--model_type', type=str, default='FCN', help='model type')
    parser.add_argument('--xai_name', type=str, default='DeepLift', help='type of explainer')

    args = parser.parse_args()

    get_real_datasets_attr(args.model_type, args.dataset_name, args.xai_name, attr_class=None)