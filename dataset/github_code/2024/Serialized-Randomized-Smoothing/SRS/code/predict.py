""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
import os
from datasets import get_dataset, get_attacked_dataset, DATASETS, get_num_classes
from core import Smooth, smooth_fit_function, deq_
from time import time
import torch
import torch.backends.cudnn as cudnn
from architectures import get_architecture
from implicit_model.model import mdeq
from implicit_model.config import config
from implicit_model.config import update_config
import datetime

import torchattacks

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

# attack randomized smoothing classifier
parser.add_argument("--attack", type=str, default=None, help="the leveraged attack")
parser.add_argument("--budget", type=float, default=0., help="the l2 norm of the perturbation")
parser.add_argument("--step_length", type=float, default=0.1, help="the step size of PGD")
parser.add_argument("--step_num", type=int, default=20, help="the number of steps of PGD")
parser.add_argument("--t", type=int, default=1, help="the temperature hyperparameter of the fitting function")
parser.add_argument("--sn", type=int, default=100, help="number of samplings for fitting function")

# implicit model hyperparameters
parser.add_argument("--f_solver", type=str, default='broyden', help="the solver for deq")
parser.add_argument("--f_thresh", type=int, default=8, help='the threshold of the iterations for the solver')
parser.add_argument("--srs", dest='srs', action='store_true', default=False, help='if use SRS')
parser.add_argument("--warmup_solver", type=str, default=None, help='the solver of warm-up')
parser.add_argument("--warmup_thresh", type=int, default=None, help='the number of iterations of warm-up')
parser.add_argument("--warmup_interval", type=int, default=1000, help='the number of intervals of warm-up')
parser.add_argument("--conf_drop", type=int, default=None, help='the number of samples in two-stage certification')
parser.add_argument("--detail", dest='detail', action='store_true', default=False, help='if print the details')


args = parser.parse_args()

cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    # load the base classifier
    if "resnet" in args.base_classifier:
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        checkpoint = checkpoint['state_dict']
    elif "implicit" in args.base_classifier:
        checkpoint = torch.load(args.base_classifier)
        model_name = args.base_classifier.split('/')[-2]
        args.cfg = f'./code/implicit_model/config/{model_name}.yaml'
        update_config(config, args)
        base_classifier = mdeq.get_cls_net(config)
    else:
        raise ValueError("No such model! ")
    base_classifier.load_state_dict(checkpoint)

    if "implicit" in args.base_classifier:
        base_classifier = torch.nn.DataParallel(base_classifier, device_ids=[0]).cuda()

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, args)
    if args.attack:
        fit_function = smooth_fit_function(smoothed_classifier, t=args.t, num=args.sn).cuda()

    # prepare output file
    len_file_name = len(args.outfile.split('/')[-1])
    outdir = args.outfile[:-len_file_name - 1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if args.attack is None:
        dataset = get_dataset(args.dataset, args.split)
    elif args.attack == 'PGD_base':
        atk = torchattacks.PGDL2(deq_(base_classifier), eps=args.budget, alpha=0.1, steps=20)
        dataset = get_attacked_dataset(args.dataset, args.split, atk)
    elif args.attack == 'PGD_smooth':
        atk = torchattacks.PGDL2(fit_function, eps=args.budget, alpha=0.1, steps=20)
        dataset = get_attacked_dataset(args.dataset, args.split, atk)
    else:
        raise ValueError("Choose attack from ['PGD_smooth', 'PGD_base']!")

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)

    f.close()
