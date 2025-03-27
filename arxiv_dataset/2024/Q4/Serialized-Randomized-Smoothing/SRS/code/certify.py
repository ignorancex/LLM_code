# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import torch.backends.cudnn as cudnn
import datetime
from architectures import get_architecture
from implicit_model.model import mdeq
from implicit_model.config import config
from implicit_model.config import update_config

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

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
    elif "mdeq" in args.base_classifier:
        checkpoint = torch.load(args.base_classifier)
        model_name = args.base_classifier.split('/')[-2]
        args.cfg = f'./code/implicit_model/config/{model_name}.yaml'
        update_config(config, args)
        base_classifier = mdeq.get_cls_net(config)
    else:
        raise ValueError("No such model! ")
    base_classifier.load_state_dict(checkpoint)

    if "mdeq" in args.base_classifier:
        base_classifier = torch.nn.DataParallel(base_classifier, device_ids=[0]).cuda()

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, args)

    # prepare output file
    len_file_name = len(args.outfile.split('/')[-1])
    outdir = args.outfile[:-len_file_name-1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
