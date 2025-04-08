from __future__ import division
from __future__ import print_function

# import debugpy
# debugpy.listen(('0.0.0.0', 5679))
# debugpy.wait_for_client()

import numpy as np
import time
import argparse
import os
import datetime
import logging
import re
import random
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from utils import *
from modules import *

t_begin = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, cnn, or gin).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
# parser.add_argument('--lr-decay', type=int, default=200,
#                     help='After how epochs to decay LR by a factor of gamma.')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')
parser.add_argument('--plateau-factor', type=float, default=0.5, help='The factor for ReduceLROnPlateau.')
parser.add_argument('--plateau-patience', type=int, default=20, help='The patience for ReduceLROnPlateau.')

parser.add_argument('--smoothness-weight', type=float, default=0,
                    help='The weight for Dirichlet energy, default = 0 (off).')
parser.add_argument('--degree-weight', type=float, default=0, help='The weight for degree loss, default = 0 (off).')
parser.add_argument('--sparsity-weight', type=float, default=0,
                    help='The weight for sparsity loss, default = 0 (off).')

parser.add_argument('--save-probs', action='store_true', default=False,
                    help='Save the probs during test.')
parser.add_argument('--not-save-checkpoint', action='store_true', default=False)
parser.add_argument('--file-name', type=str, default='')
parser.add_argument('--exp-name', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor

args.data_path = args.file_name
if args.data_path != '' and args.suffix == 'scRNAseq':
    data = np.load(args.data_path)
    args.num_atoms = data.shape[2]
    args.timesteps = data.shape[1]
    args.dims = 1

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Save model and meta-data.
if args.save_folder:
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if os.name == 'nt':
        timestamp = timestamp.replace(":", "-")
    if os.path.exists(args.save_folder) == False:
        os.makedirs(args.save_folder, exist_ok=True)
    if args.save_folder[-1] == '/':
        args.save_folder = args.save_folder[:-1]
    if args.prior:
        save_folder = '{}/{}-it-prior-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                                args.decoder, timestamp)
    else:
        save_folder = '{}/{}-it-E{}-D{}-exp{}/'.format(args.save_folder, args.suffix, args.encoder,
                                                        args.decoder, timestamp)
    if args.exp_name is not None:
        save_folder = save_folder[:-1] + "-" + args.exp_name + "/"
    os.mkdir(save_folder)

    if args.save_probs:
        probs_folder = save_folder + 'probs/'
        os.mkdir(probs_folder)

    if args.not_save_checkpoint == False:
        ckpt_file = save_folder + 'checkpoint/'
        os.mkdir(ckpt_file)

    log_file = os.path.join(save_folder, 'log.txt')
    logging.basicConfig(filename = log_file, 
                            filemode="a", 
                            format = '%(asctime)s %(name)s %(levelname)s:%(message)s',
                            level = logging.INFO,
                            )
    log = logging.getLogger()
    tf_board_writer = SummaryWriter(log_dir = save_folder)
    
else:
    class Log:
        def info(self, *args):
            print(*args)
    log = Log()

    log.info("WARNING: No save_folder provided!")


if args.prediction_steps > args.timesteps:
    args.prediction_steps = args.timesteps

log.info(args)
if args.dynamic_graph:
    log.info("Testing with dynamically re-computed graph.")

fn_train = np.load(args.file_name)
fn_test = np.load(args.file_name.replace("train", "test"))

train_loader = load_data_scRNAseq(fn_train, args.batch_size, args.suffix, time_steps=args.timesteps)#, norm_flag=True)
test_loader = load_data_scRNAseq(fn_test, args.batch_size, args.suffix, time_steps=args.timesteps)#, norm_flag=True)

# Generate off-diagonal interaction graph: discarded
# off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
log.info(f"num_atoms: {args.num_atoms}")
off_diag = np.ones([args.num_atoms, args.num_atoms])

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'gin':
    encoder = GINEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)

current_epoch = 0
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    rel_rec = checkpoint['rel_rec']
    rel_send = checkpoint['rel_send']
    scheduler = checkpoint['scheduler']
    optimizer = checkpoint['optimizer']
    current_epoch = int(re.findall("ckpt_(\d+).pt", args.checkpoint)[0]) + 1
    


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
#                                 gamma=args.gamma)
scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True
        )

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = [0.01 for i in range(args.edge_types - 1)]
    prior = np.array([1-sum(prior)] + prior)  # TODO: hard coded for now
    log.info("Using prior")
    log.info(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, rel_rec, rel_send):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    de_train = []
    dl_train = []
    sl_train = []
    probs_train = []
    encoder.train()
    decoder.train()
    
    for batch_idx, (data,) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 100,
                             burn_in=True,
                             burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output = decoder(data, edges, rel_rec, rel_send,
                             args.prediction_steps)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)

        loss_de = dirichlet_energy(adj=prob, data=data, num_nodes=args.num_atoms, cuda=args.cuda)
        loss_dl = degree_loss(adj=prob, num_nodes=args.num_atoms, cuda=args.cuda)
        loss_sl = sparsity_loss(adj=prob, num_nodes=args.num_atoms)

        loss = loss_nll + loss_kl + \
                args.smoothness_weight * loss_de - \
                args.degree_weight * loss_dl + \
                args.sparsity_weight * loss_sl
        loss = loss / data.size(0)  # normalize by batch size

        probs_train.append(prob)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        de_train.append(loss_de.item())
        dl_train.append(loss_dl.item())
        sl_train.append(loss_sl.item())
    
    scheduler.step(np.mean(nll_train)+np.mean(kl_train))
    
    if args.not_save_checkpoint == False:
        torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                "rel_rec": rel_rec,
                "rel_send": rel_send,
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(ckpt_file, f"ckpt_{epoch}.pt"))
    if epoch == args.epochs-1:
        torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                "rel_rec": rel_rec,
                "rel_send": rel_send,
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_folder, f"ckpt_{epoch}.pt"))
    
    if args.save_probs:
        np_probs = np.concatenate([element.detach().cpu().numpy() for element in probs_train])
        probs_save_file = probs_folder + 'probs_' + str(epoch) + '.npy'
        np.save(probs_save_file, np_probs)
    
    if tf_board_writer is not None:
        if args.save_probs:
            update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train, prob=np_probs)
        else:
            update_tensorboard(tf_board_writer=tf_board_writer, epoch=epoch, 
                                nll_train=nll_train, kl_train=kl_train, mse_train=mse_train,
                                de_train=de_train, dl_train=dl_train, sl_train=sl_train)
    
    ############## test #################
    nll_test = []
    kl_test = []
    mse_test = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data,)in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)

        logits = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)
        prob = my_softmax(logits, -1)

        # validation output uses teacher forcing
        output = decoder(data, edges, rel_rec, rel_send, 1)

        target = data[:, :, 1:, :]
        loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        mse_test.append(F.mse_loss(output, target).item())
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())
    ############# End test ###############
    
    log.info('Epoch: {:04d}, nll_train: {:.10f}, kl_train: {:.10f}, mse_train: {:.10f}, de_train: {:.10f}, dl_train: {:.10f}, sl_train: {:.10f}, nll_test: {:.10f}, kl_test: {:.10f}, mse_test: {:.10f}, time: {:.4f}s'.format
                (epoch, 
                 np.mean(nll_train), 
                 np.mean(kl_train), 
                 np.mean(mse_train), 
                 np.mean(de_train),
                 np.mean(dl_train),
                 np.mean(sl_train),
                 np.mean(nll_test), 
                 np.mean(kl_test), 
                 np.mean(mse_test), 
                 time.time() - t)
                )
    return np.mean(nll_train), rel_rec, rel_send

# Train model
for epoch in range(current_epoch, args.epochs):
    train_loss, rel_rec, rel_send = train(epoch, rel_rec, rel_send)
    
log.info("Optimization Finished!")
