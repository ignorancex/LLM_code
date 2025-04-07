import argparse
import os
import sys

from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

try:
    from text_complete import text_results
except ModuleNotFoundError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_transe",
                                     description="Trains the TransE algorithm for kg embeddings")
    parser.add_argument('-d', '--dimension', required=True, type=int, default=200,
                        help='Embedding dimension for h, r and t',
                        dest='dim')
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    if args.data == 'riedel':
        fb_path = os.path.join(wd, "data", "RESIDE_KG/")
        out_path = os.path.join(wd, "kg-embeddings", "checkpoints", "transe_{d}_riedel.ckpt".format(d=args.dim))
        hyperparams = {'batch_size': 128,
                       'neg_ent': 25,
                       'neg_rel': 0,
                       'margin': 5.0,
                       'niter': 100}
    elif args.data == 'gids':
        fb_path = os.path.join(wd, "data", "GIDS_KG/")
        out_path = os.path.join(wd, "kg-embeddings", "checkpoints", "transe_{d}_gids.ckpt".format(d=args.dim))
        hyperparams = {'batch_size': 128,
                       'neg_ent': 25,
                       'neg_rel': 0,
                       'margin': 5.0,
                       'niter': 100}
    else:
        print('Unknown dataset, please run pre-processing steps.')
        sys.exit()

    train_dataloader = TrainDataLoader(in_path=fb_path,
                                       batch_size=hyperparams['batch_size'],
                                       threads=8,
                                       sampling_mode="normal",
                                       bern_flag=1,
                                       filter_flag=1,
                                       neg_ent=hyperparams['neg_ent'],
                                       neg_rel=hyperparams['neg_rel'])


    test_dataloader = TestDataLoader(fb_path, "link")

    transe = TransE(ent_tot=train_dataloader.get_ent_tot(),
                    rel_tot=train_dataloader.get_rel_tot(),
                    dim=args.dim,
                    p_norm=1,
                    norm_flag=True)

    model = NegativeSampling(model=transe,
                             loss=MarginLoss(margin=hyperparams['margin']),
                             batch_size=train_dataloader.get_batch_size())

    # train the model
    trainer = Trainer(model=model,
                      data_loader=train_dataloader,
                      train_times=hyperparams['niter'],
                      alpha=1.0,
                      use_gpu=True)
    trainer.run()
    transe.save_checkpoint(out_path)
    transe.load_checkpoint(out_path)
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=True)
    try:
        text_results("Finished training TransE with {n} dimensional vectors on {d} data.".format(n=args.dim,
                                                                                                 d=args.data))
    except:
        print('Done')
