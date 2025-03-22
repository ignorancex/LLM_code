import os
import logging
import warnings

import torch

warnings.filterwarnings('ignore') # sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from transformers import logging
logging.set_verbosity_error()

import sys

temp_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
if '--TEMP_DIR' in sys.argv:
    ind = sys.argv.index('--TEMP_DIR')
    temp_dir = sys.argv[ind + 1]
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import platform
import numpy as np

from RST.src.ELib import ELib
from RST.src.EPretrainProj import EPretrainProj

def main():
    parser = argparse.ArgumentParser()
    # general params
    parser.add_argument("--cmd", default=None, type=str, required=True, help='')
    parser.add_argument("--TEMP_DIR", default=temp_dir, type=str, required=False, help='')
    # api params
    parser.add_argument("--itr", default=1, type=int, required=False, help='')
    parser.add_argument("--per_query", default=False, type=bool, required=False, help='')
    parser.add_argument("--model_path", default=None, type=str, required=True, help='')
    parser.add_argument("--train_path", default=None, type=str, required=True, help='')
    parser.add_argument("--valid_path", default=None, type=str, required=False, help='')
    parser.add_argument("--test_path", default=None, type=str, required=False, help='')
    parser.add_argument("--unlabeled_path", default=None, type=str, required=False, help='')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help='')
    parser.add_argument("--device", default=None, type=int, required=True, help='')
    parser.add_argument("--seed", default=None, type=int, required=True, help='')
    parser.add_argument("--train_sample", default=None, type=int, required=True, help='')
    parser.add_argument("--unlabeled_sample", default=None, type=int, required=True, help='')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    device = 'cpu'
    device_name = device
    if args.device >= 0:
        device = 'cuda:' + str(args.device)
        device_name = torch.cuda.get_device_name(args.device)
    device_2 = 'cpu'
    if 'device_2' in args and (args.device_2 is not None and args.device_2 >= 0):
        device_2 = 'cuda:' + str(args.device_2)
        device_name = device_name + ', ' + torch.cuda.get_device_name(args.device)
    print('setup:',
          '| python>', platform.python_version(),
          '| numpy>', np.__version__,
          '| pytorch>', torch.__version__,
          '| device>', device_name)

    if args.cmd.startswith('bert'): # pretrain
        seed = args.seed
        result = list()
        for cur_itr in range(args.itr):
            print('------------------------------------')
            print('iteration ' + str(cur_itr + 1) + ' began with seed=\'' + str(seed) + '\'   at ' + ELib.get_time())
            if cur_itr >= 0:
                output_dir = args.output_dir + '_' + str(cur_itr)
                cur_perf = EPretrainProj.run(args.cmd, args.per_query, args.train_path, args.valid_path,
                                             args.test_path, args.unlabeled_path, args.model_path,
                                             None, None, None, None, output_dir, device, None, seed,
                                             args.train_sample, args.unlabeled_sample)
                if cur_perf is not None:
                    result.append(cur_perf)
            seed += 1230
        if len(result) > 0:
            ELib.print_iteration_results(result)
        ELib.PASS()
    ELib.PASS()


if __name__ == "__main__":
    print("Started at", ELib.get_time())

    if len(sys.argv) <= 1:
        print('no argument was passed...')
    else:
        main()

    print("\nDone at", ELib.get_time())
    pass


