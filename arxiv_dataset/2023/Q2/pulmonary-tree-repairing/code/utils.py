import argparse
import sys
import ntpath
import time
import json
from pathlib import Path
import glob
import numpy as np
import struct
import os, shutil
import colorama
from colorama import Fore, Style
from torchvision.utils import save_image
from colorama import init
from random import shuffle,seed
init(autoreset=True)


def main_parser(args=sys.argv[1:]):

    # Parser definition
    parser = argparse.ArgumentParser(description="Parses command.")

    # Parser Options
    parser.add_argument("-tr", "--train", action='store_true', help="Start training with default parameters")
    parser.add_argument("-te", "--test", action='store_true', help="Test with pretrained weights")
    parser.add_argument("-resume", "--resume", action='store_true', help="Resume with pretrained weights")
    parser.add_argument("-ckp", "--checkpoint", default=None, help="Path of the checkpoint for inference/test")
    parser.add_argument("-m", "--model", help="Select a model")
    parser.add_argument("-ms", "--modelSum", action='store_true', help="Show the summary of models and configurations")
    
    options = parser.parse_args(args)
    if not options.model:
        print("--please specify model (-m)!")

    return options


def custom_print(str_print, text_width=64, style='*', top_border=True, bottom_border=True):
    if top_border:
        print(style * text_width)

    print('{:^{width}}'.format(str_print, width=text_width))

    if bottom_border:
        print(style * text_width)


def config_reader(model):

    try:
        path = 'config_' + model + '.json'
        # print(path)
        with open(path, 'r') as fp:
            config = json.load(fp)
    except:
        user_input = input("Unable to read config.json file! ") or "N"
        exit()

    return config


def format_dir_path(path):

    if not path.endswith("/"):
        path = path.rstrip() + "/"

    return path
            

def get_npz_list(root, data_type='.npz'):
        
        data_type = '*'+ data_type
        img_list = glob.glob(format_dir_path(root) + data_type)
        
        return img_list

def config_shower(model, text_width=64):

    config = config_reader(model)
    custom_print(Fore.YELLOW + "Hyper-parameters and Configurations", text_width=text_width)
    for c in config:
        custom_print("{}:".format(c).upper() + Fore.YELLOW + "{}".format(config[c]), text_width=text_width, style='-' )

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)