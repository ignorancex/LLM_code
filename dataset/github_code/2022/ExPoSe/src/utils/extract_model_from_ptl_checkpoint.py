import argparse

import torch as t

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--save', default='model.pt')
args = parser.parse_args()

ckpt = t.load(args.file, map_location='cpu')

weights = {k.lstrip('model.'): v for k, v in ckpt['state_dict'].items()}

t.save(weights, args.save)
