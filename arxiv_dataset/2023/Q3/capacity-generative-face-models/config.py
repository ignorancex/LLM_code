# ///////////// Copyright 2023 Vishnu Boddeti. All rights reserved. /////////////
# //
# //   Project     : Capacity of Generative Face Models
# //   File        : config.py
# //   Description : Configuration file for capacity of generative face models
# //
# //   Created On: 08/20/2023
# //   Created By: Vishnu Boddeti <mailto:vishnu@msu.edu>
# ////////////////////////////////////////////////////////////////////////////

import argparse
import re
from ast import literal_eval as make_tuple


def parse_args():    
    parser = argparse.ArgumentParser(description='Capacity of Generative Face Models')

    # the following two parameters can only be provided at the command line.
    parser.add_argument('--image-path', type=str, default='./images/', metavar='', help='path where images are present')
    parser.add_argument('--feature-path', type=str, default='./features/', metavar='', help='path where extracted features are present')
    parser.add_argument('--output-path', type=str, default='./results/', metavar='', help='path where capacity results are saved')
    args, remaining_argv = parser.parse_known_args()

    # ======================= Settings =====================================
    parser.add_argument('--gen-type', type=str, choices=['conditional', 'unconditional'], help='type of generative model')
    parser.add_argument('--dataset', type=str, metavar='', help='dataset of images')
    parser.add_argument('--face-model', type=str, metavar='', help='generative face model')
    parser.add_argument('--quantile', type=float, default=0.05, metavar='', help='generative face model')
    parser.add_argument('--ref-dataset', type=str, choices=["lfw", "cfp_fp", "cplfw", "agedb_30", "calfw"], help='generative face model')
    parser.add_argument('--metadata-file', type=str, default=None, help='generative face model')
    parser.add_argument('--max-samples', type=int, default=10000, help='maximum number of samples to use')
    parser.add_argument('--save-results', type=bool, default=True, help='write results to file')
    parser.add_argument("-f", required=False)
    args = parser.parse_args(remaining_argv)

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args