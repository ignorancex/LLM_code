
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable
from utils import random_masking, perturb, KL_div
import model


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32, help = 'hidden channel sieze')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=1., help='beta for beta-vae')

    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--perturbed', action='store_true', help='Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio for perturbation of data, see Ren et al.')

    opt = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    dataset = dset.CIFAR10(root=opt.dataroot, download=True,train = True,
                           transform=transforms.Compose([
                               transforms.Resize((opt.imageSize)),
                               transforms.ToTensor(),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                           shuffle=True, num_workers=int(opt.workers))

    

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)

    image_size = [32, 32, 3]
    netG = model.Decoder(image_size, nz, 256)
    netG.apply(weights_init)

    netE = model.Encoder(image_size, nz)
    netE.apply(weights_init)

    
    netE.to(device)
    netG.to(device)
    # setup optimizer
    
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay = 3e-5)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay = 3e-5)

    netE.train()
    netG.train()

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    rec_l = []
    kl = []
    tloss = []
    for epoch in range(opt.niter):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)

            if opt.perturbed:
                x = perturb(x, opt.ratio, device)


            x_d = F.interpolate(x, scale_factor=0.25)
            x_d = F.interpolate(x_d, scale_factor=4)
            # x_d = random_masking(x, 0.5, 4)


            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()

            [z,mu,logvar] = netE(x_d.float())
            recon = netG(z)
            
            recon = recon.contiguous()
            recon = recon.view(-1,256)

            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            
            loss =  recl + opt.beta*kld.mean()
            
                
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            for group in optimizer1.param_groups: # clip gradients
                utils.clip_grad_norm_(group['params'], 100, norm_type=2)
            for group in optimizer2.param_groups: # clip gradients
                utils.clip_grad_norm_(group['params'], 100, norm_type=2)
            

            optimizer1.step()
            optimizer2.step()
            rec_l.append(recl.detach().cpu().item())
            kl.append(kld.mean().detach().cpu().item())
           
            if not i % 100:
                print('epoch:{} recon:{} kl:{}'.format(epoch,np.mean(rec_l),np.mean(kl)
                    ))

        torch.save(netG.state_dict(), './saved_models/netG.pth')
        torch.save(netE.state_dict(), './saved_models/netE.pth')
