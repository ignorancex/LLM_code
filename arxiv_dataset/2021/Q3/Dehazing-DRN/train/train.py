import random
import argparse
from skimage.measure import compare_ssim

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

from utils.utils import *
from utils.vgg16 import Vgg16
import models.train_net as net

torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, default='./facades/train512/', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False, default='./facades/val512/', help='path to val dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--Net', default='', help="path to net (to continue training)")
parser.add_argument('--exp', default='./checkpoints_new', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
opt = parser.parse_args()
print(opt)

# get logger
create('./Log/')
trainLogger = open('./Log/train.log', 'a+')

create(opt.exp)

# set seed
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
dataloader = getLoader(opt.dataroot)
valDataloader = getLoader(opt.valDataroot)



ngf = opt.ngf
inputChannelSize = opt.inputChannelSize
outputChannelSize = opt.outputChannelSize

# get models
Net = net.dehaze(inputChannelSize, outputChannelSize, ngf)

Net.apply(weights_init)
if opt.Net != '':
    Net.load_state_dict(torch.load(opt.Net))

criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss()
criterionMSE = nn.MSELoss()

lambdaIMG = opt.lambdaIMG

Net.cuda()
criterionBCE.cuda(), criterionCAE.cuda()


vgg = Vgg16()
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.pth")))
vgg.cuda()

# get optimizer
optimizerG = optim.Adam(Net.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.00005)

# NOTE training loop
ganIterations = 0
best_ssim, best_psnr = 0, 0
Net.train()
for epoch in range(opt.epochs):
    if epoch > opt.annealStart:
        adjust_learning_rate(optimizerG, opt.lr, opt.annealEvery)

    for batch_id, data in enumerate(dataloader, 0):
        input, target, trans, ato = data
        batch_size = target.size(0)
        target, input, trans, ato = Variable(target), Variable(input), Variable(trans), Variable(ato)
        target, input, trans, ato = target.float().cuda(), input.float().cuda(), trans.float().cuda(), ato.float().cuda()

        # get paired data
        fine_dehaze, tran_hat, atp_hat, dehaze = Net(input)

        for p in Net.parameters():
            p.requires_grad = True
        Net.zero_grad()

        L_img_ = criterionCAE(fine_dehaze, target)
        L_img = lambdaIMG * L_img_
        L_img.backward(retain_graph=True)

        L_tran_ = criterionCAE(tran_hat, trans)

        # gradie_h_est, gradie_v_est = gradient(tran_hat)
        # gradie_h_gt, gradie_v_gt = gradient(trans)
        # L_tran_h = criterionCAE(gradie_h_est, gradie_h_gt)
        # L_tran_v = criterionCAE(gradie_v_est, gradie_v_gt)

        # L_tran = lambdaIMG * (L_tran_ + 2 * L_tran_h + 2 * L_tran_v)
        L_tran = lambdaIMG * L_tran_
        L_tran.backward(retain_graph=True)

        features_trans = vgg(trans)
        f_1 = Variable(features_trans[1].data, requires_grad=False)
        features_tran_hat = vgg(tran_hat)
        content_loss1 = 0.8 * lambdaIMG * criterionCAE(features_tran_hat[1], f_1)
        content_loss1.backward(retain_graph=True)

        f_0 = Variable(features_trans[0].data, requires_grad=False)
        content_loss0 = 0.8 * lambdaIMG * criterionCAE(features_tran_hat[0], f_0)
        content_loss0.backward(retain_graph=True)

        L_ato_ = criterionCAE(atp_hat, ato)
        L_ato = lambdaIMG * L_ato_
        L_ato.backward(retain_graph=True)

        re_input = fine_dehaze * tran_hat + atp_hat - atp_hat * tran_hat
        re_loss_ = criterionCAE(input, re_input)
        re_loss = lambdaIMG * re_loss_
        re_loss.backward(retain_graph=True)

        optimizerG.step()

        if ganIterations % opt.display == 0:
            trainLogger.write('[%d/%d][%d/%d] dehaze L1 loss: %f trans L1 loss: %f trans L1+gradient loss: %f ato L1 loss: %f\n'
                   % (epoch, opt.epochs, batch_id, len(dataloader), L_img.item(), L_tran_.item(), L_tran.item(), L_ato.item()))
            trainLogger.flush()

    avg_mse = 0
    total_ssim = 0
    total_psnr = 0
    i = 0
    for val_batch_id, val_data in enumerate(valDataloader, 0):
        i += 1
        varIn, varTar, _, _ = val_data

        varIn, varTar = Variable(varIn), Variable(varTar)
        varIn, varTar = varIn.float().cuda(), varTar.float().cuda()

        prediction, val_tran_hat, val_atp_hat, val_dehaze = Net(varIn)

        prediction = prediction.data.cpu().numpy().squeeze()
        prediction = prediction.transpose((1, 2, 0))
        varTar = varTar.data.cpu().numpy().squeeze()
        varTar = varTar.transpose((1, 2, 0))

        per_ssim = compare_ssim(prediction, varTar, multichannel=True)
        total_ssim = total_ssim + per_ssim

        per_psnr = compare_psnr(prediction, varTar)
        total_psnr = total_psnr + per_psnr

    print('>>>>>>>>%s>>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
    avg_ssim = total_ssim / len(valDataloader)
    print('===>avg_ssim: %.6f' % avg_ssim)
    avg_psnr = total_psnr / len(valDataloader)
    print('===>avg_psnr: %.6f' % avg_psnr)


    trainLogger.write('===>avg_ssim: %.6f\n' % avg_ssim)
    trainLogger.write('===>avg_psnr: %.6f\n' % avg_psnr)
    trainLogger.flush()

    torch.save(Net, '%s/net_epoch_%d.pth' % (opt.exp, epoch))
    Net.train()

trainLogger.close()

