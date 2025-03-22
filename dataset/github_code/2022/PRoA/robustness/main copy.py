from argparse import ArgumentParser
from re import A
import re
import traceback
import os
import git
import torch as ch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import cox
import cox.utils
import cox.store
import scipy.optimize
from tqdm import tqdm

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults
    from .defaults import check_and_fill_args
except:
    print(traceback.format_exc())
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    loader = (train_loader, val_loader)

    # New loss
    def chance_loss(X, label, model, T=0, eps=0.01, sample_size=10000):
        kwargs = {
        'spatial_constraint': 45,
        'tries':1,
        'use_best': True, 
        'attack_type': 'random',
        'do_tqdm': False
        }
        # model.eval()
        bs=1000
        im = X.repeat(bs, 1, 1, 1)
        all_label = label.repeat(bs)
        all_output = ch.ones(sample_size, 10)
        all_output = all_output.cuda()
        with ch.no_grad():
            for i in range(int(sample_size/bs)):                
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                output, _ = model(im_spat)
                all_output[bs*i:bs*(i+1), :] = F.softmax(output, dim=1)
            output_ori, _ = model(X)
        output_ori = F.softmax(output_ori, dim=1)
        output_sorted, ind_sorted = output_ori.sort(dim=1)
        d = (output_sorted[:, -1]-output_sorted[:, -2])/2   
        output_ori = output_ori.repeat(sample_size, 1)
        inf_n = ch.norm(output_ori-all_output, p=float('inf'), dim=1)

        g = inf_n
        T = d

        alpha = ch.tensor([1.], requires_grad = True)
        def min_obj(alpha):
            alpha = ch.tensor(alpha).cuda()
            alpha.requires_grad_()  
            g_c = ch.exp(alpha*(g - T)).mean()-eps
            min = g_c 
            min.backward(retain_graph=True)
            return min.data.cpu().numpy(), alpha.grad.data.cpu()
        result = scipy.optimize.basinhopping(min_obj,
                                    alpha.detach().numpy(), niter=10,
                                    minimizer_kwargs={"method":"L-BFGS-B", "jac":True, "bounds":[(0, None)]}
                                    ) # NB: we will compute the jacobian
        return ch.tensor(max(0, result.fun))
        
    def custom_train_loss(model, inp, targ, lam=0.01, T=0, eps=0.01, sample_size=500):
        logits, _ = model(inp)
        ce_loss = F.cross_entropy(logits, targ)
        if ce_loss >= 100:
            print("error1")
            print(logits)
        cur_cons_loss = 0
        # for i in tqdm(range(inp.size(0))):
        for i in range(inp.size(0)):
            cur_cons_loss = cur_cons_loss + chance_loss(inp[i,:,:,:].unsqueeze(0), targ[i].unsqueeze(0), model=model, T=T, eps=eps, sample_size=sample_size)
        return ce_loss+lam*(cur_cons_loss/inp.size(0))


    adv_crit = ch.nn.CrossEntropyLoss(reduction='none').cuda()
    def custom_adv_loss(model, inp, targ, lam=0.5, T=0, eps=0.01, sample_size=100):
        logits = model(inp)
        ce_loss = adv_crit(logits, targ)
        adv_loss = ch.ones_like(ce_loss)
        for i in range(inp.size(0)):
            adv_loss[i] = ce_loss[i] + lam*chance_loss(inp[i,:,:,:].unsqueeze(0), targ[i].unsqueeze(0), model=model, T=T, eps=eps, sample_size=sample_size)
        return adv_loss, logits

    if not args.eval_only:
        # if args.adv_train:
        #     args.custom_train_loss = custom_train_loss
        #     args.custom_adv_loss = custom_adv_loss
        args.custom_train_loss = custom_train_loss
        args.custom_adv_loss = custom_adv_loss
        model = train_model(args, model, loader, store=store)
    else:
        eval_model(args, val_loader, model, store)

    return model

def setup_args(args):
    '''
    Set a number of path related values in an arguments object. Also run the
    sanity check.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)
    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)
    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    # repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
    #                     search_parent_directories=True)
    # git_commit = repo.head.object.hexsha
    # args.git_commit = git_commit

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.as_dict()
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)
