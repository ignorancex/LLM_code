import subprocess
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--elicit", default=0.4, type=float)
argParser.add_argument("--softmax", default=10.0, type=float)
argParser.add_argument("--length", type=int, default=9)
argParser.add_argument("--device", default=0, type=int)
argParser.add_argument("--category", type=str)
args = argParser.parse_args()


if args.category == 'bomb':
    selection = [3, 7, 36, 55, 71, 78, 101, 151, 154, 236, 274, 313]
elif args.category == 'misinformation':
    selection = [25, 26, 31, 39, 62, 72, 121, 156, 255, 283]
elif args.category == 'hacking':
    selection = [1, 10, 14, 16, 17, 38, 40, 41, 54, 56, 408]
elif args.category == 'theft':
    selection = [12, 18, 30, 42, 46, 63, 69, 87, 99, 124, 519]
elif args.category == 'suicide':
    selection = [34, 138, 188, 175, 225, 349, 374, 378, 382, 388, 469]

for i in selection:
    subprocess.run('CUDA_VISIBLE_DEVICES={} python main.py --path ./results_category/{}/res{} --q_index {} --elicit {} --softmax {} --length {}'.format(args.device, args.category, i, i, args.elicit, args.softmax, args.length), shell=True)
