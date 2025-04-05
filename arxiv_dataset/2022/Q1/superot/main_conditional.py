import torch
from torch import nn, optim
from torch.autograd import Variable, grad

import cGAN
import utils
import visdom
from scipy.sparse import csc_matrix
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression

import numpy as np
import sys
import pandas as pd
import os
import pickle

torch.manual_seed(1)

# ============ EVALUATION =================
CLONE_ANNOTATION = 'clone_annotation_in_vitro.npz'
CELL_METADATA = 'cell_metadata_in_vitro.txt'

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
cellMetadataInVitroDay = np.loadtxt(
    CELL_METADATA, skiprows=1, usecols=(0,))
metadata = pd.read_csv(CELL_METADATA, sep='\\t', header=0)
cellMetadataInVitroType = np.genfromtxt(
    CELL_METADATA, dtype='str',  skip_header=1, usecols=(2,))
inpDataset = np.load('dat1_test_unsupervised', allow_pickle=True)
inpDataset = torch.from_numpy(inpDataset).float().to(device)
clusters_test = torch.from_numpy(np.load('clusters_test', allow_pickle=True)).float().to(device)
targetDataset = np.load('unsupervised_dat2_train', allow_pickle=True)
targetIndices = np.load('day4_6_unsupervised', allow_pickle=True)
day2ind = np.load('day2Ind_test', allow_pickle=True)
day4_6ind = np.load('day4_6Ind_test', allow_pickle=True)
print(day2ind.shape, day4_6ind.shape, targetDataset.shape, targetIndices.shape, clusters_test.shape)
clone_data = np.load(CLONE_ANNOTATION)
clone_data = csc_matrix(
    (clone_data['data'], clone_data['indices'], clone_data['indptr']), shape=(130887, 5864)).toarray()


y = []
for i in range(len(targetDataset)):
    if cellMetadataInVitroType[targetIndices[i]] == 'Monocyte':
        y.append(0)

    if cellMetadataInVitroType[targetIndices[i]] == 'Neutrophil':
        y.append(1)

lr = LogisticRegression(multi_class='ovr').fit(targetDataset, y)
acc = 0.0
# ============ PARSE ARGUMENTS =============

args = utils.setup_args()
args.save_name = args.save_file + args.env
print(args)

# ============ GRADIENT PENALTY (for discriminator) ================


def calc_gradient_penalty(netD, real_data, fake_data, clusters):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, clusters)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambG
    return gradient_penalty


def calc_gradient_penalty_rho(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    _, disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambG2
    return gradient_penalty


# ============= TRAINING INITIALIZATION ==============
# initialize discriminator
netD = cGAN.Discriminator(5964, 500)
print("Discriminator loaded")

# initialize generator
netG = cGAN.Generator(5964, 500)
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    print("Using GPU")

# load data

loader = utils.setup_data_loaders_unsupervised(args.batch_size)
print('Data loaded')
sys.stdout.flush()

# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr=args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr=args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
mse = nn.MSELoss(reduce=False)
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()

# =========== LOGGING INITIALIZATION ================

vis = utils.init_visdom(args.env)
tracker = utils.Tracker()
tracker_plot = None
scale_plot = None

# ============================================================
# ============ MAIN TRAINING LOOP ============================
# ============================================================

for epoch in range(args.max_iter):
        for it, (s_inputs, clusters, t_inputs) in enumerate(loader):
            s_inputs, clusters, t_inputs = Variable(s_inputs), Variable(clusters), Variable(t_inputs)
            if torch.cuda.is_available():
                s_inputs, clusters, t_inputs = s_inputs.cuda(), clusters.cuda(), t_inputs.cuda()
                netD.cuda()
                netG.cuda()

        # ================== Train generator =========================
            if it % args.critic_iter == args.critic_iter-1:
                netG.train()
                netD.eval()

                netG.zero_grad()

                # pass source inputs through generator network
                s_generated, s_scale = netG(s_inputs, clusters)

                # pass generated source data and target inputs through discriminator network
                s_outputs = netD(s_generated, clusters)
                 
                G_loss = args.lamb0*torch.mean(torch.sum(mse(s_generated, s_inputs), dim=1))
                if args.psi2 == "EQ":
                    G_loss += -args.lamb2*torch.mean(s_outputs)
                else:
                    G_loss += args.lamb2 * \
                        torch.mean(LOG2.expand_as(s_outputs) +
                                   logsigmoid(s_outputs) - s_outputs)
                # update params
                G_loss.backward()
                G_opt.step()

        # ================== Train discriminator =========================

            else:
                netD.train()
                netG.eval()

                netD.zero_grad()

                # pass source inputs through generator network
                s_generated, s_scale = netG(s_inputs, clusters)

                # pass generated source data and target inputs through discriminator network
                s_outputs, t_outputs = netD(s_generated, clusters), netD(t_inputs, clusters)

                # compute loss
                # D_loss = 0
                D_loss = calc_gradient_penalty(
                    netD, s_generated.data, t_inputs.data, clusters)
                if args.psi2 == "EQ":
                    D_loss += torch.mean(s_outputs) - torch.mean(t_outputs)
                else:
                    D_loss += -torch.mean(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs) - torch.mean(
                        LOG2.expand_as(t_outputs)+logsigmoid(t_outputs))
                # update params
                D_loss.backward()
                D_opt.step()

# ================= Log results ===========================================

        netD.eval()
        netG.eval()
        for s_inputs, clusters, t_inputs in loader:
            num = s_inputs.size(0)
            s_inputs, clusters, t_inputs = Variable(s_inputs), Variable(clusters), Variable(t_inputs)
            if torch.cuda.is_available():
                s_inputs, clusters, t_inputs = s_inputs.cuda(), clusters.cuda(), t_inputs.cuda()

            s_generated, s_scale = netG(s_inputs, clusters)
            s_outputs, t_outputs = netD(s_generated, clusters), netD(t_inputs, clusters)

            W_loss = args.lamb0 * \
                torch.mean(torch.sum(mse(s_generated, s_inputs), dim=1))
            W_loss += torch.mean((LOG2.expand_as(s_outputs) +
                                 logsigmoid(s_outputs)-s_outputs))
            W_loss += torch.mean(LOG2.expand_as(t_outputs) +
                                 logsigmoid(t_outputs))
            tracker.add(W_loss.cpu().data, num)
            # save transported points
            if epoch % 10 == 0 and epoch > 200:
                with open(args.save_name+str(epoch)+"_trans.txt", 'ab') as f:
                    np.savetxt(f, s_generated.cpu().data.numpy(), fmt='%f')
        tracker.tick()

    # save models
        torch.save(netD.cpu().state_dict(), args.save_name+"_netD.pth")
        torch.save(netG.cpu().state_dict(), args.save_name+"_netG.pth")

        if torch.cuda.is_available():
            netD.cuda()
            netG.cuda()

        # Evaluate current set of transported points, and save best acc in 'acc' 
        inpDataset2, s_scale = netG(Variable(inpDataset), Variable(clusters_test))
        totalCount = 0.0
        predictedY = lr.predict(inpDataset2.cpu().detach().numpy())
        for i in range(len(day2ind)):
            correct = metadata.loc[day4_6ind[i]][2]
            actual_calc= 1 if correct == 'Neutrophil' else 0
            if actual_calc == predictedY[i]:
                totalCount += 1
        new_acc = totalCount/len(day2ind)
        if new_acc > acc:
            print("NEW ACC: ", new_acc)
            acc = new_acc
            np.array([acc]).dump("acc")
            torch.save(netG.cpu().state_dict(), "modelunsupervised")
            if args.lamb0 == 0.6:
                torch.save(netG.cpu().state_dict(), "modeltransportunsupervised")

    # save tracker
        with open(args.save_name+"_tracker.pkl", 'wb') as f:
            pickle.dump(tracker, f)
        utils.plot(tracker, epoch, t_inputs.cpu().data.numpy(), args.env, vis)
