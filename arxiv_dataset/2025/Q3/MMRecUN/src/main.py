# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
from utils.test_model import test
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--unlearn', '-u', type=str, default=None, 
                        help='''
                        None: Learn on all data from scratch,
                        gold: Retrain from scratch using retain data, all data except forget user data, 
                        specified as forget_percentage in config file,
                        reverse: Unlearn using reverse learning for user data,
                        gold_item: Retrain from scratch using retain data, all data except forget item data,
                        specified as forget_percentage in config file,
                        reverse_item: Unlearn using reverse learning for item data,
                        gold_user_item: Retrain from scratch using retain data, all data except forget user and item data.''')
    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    best_model = quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, unlearn=args.unlearn, save_model=True)
    
    # These functions test a trained model. The model can either be loaded from a file or exist already in memory.
    # The first function call tests the best model obtained from training, while the second function call tests a model loaded from a specific file.
    # test(model=args.model, dataset=args.dataset, config_dict=config_dict, unlearn=args.unlearn, name=best_model)
    # test(model=args.model, dataset=args.dataset, config_dict=config_dict, unlearn=args.unlearn, name='Original_Frozen_MGCN_clothing.pth')

