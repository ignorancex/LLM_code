import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from nets.net import L_net,fusionnet
import datetime


coef=(1,0.5,0.1,0.2,0.1)
num_epochs =40

lr = 1e-4
step_size=10
gamma=0.1
weight_decay = 0
batch_size=8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net_fusion=fusionnet().to(device)
net_L=L_net().to(device)

optimizer1 = torch.optim.Adam(net_fusion.parameters(), lr=lr, weight_decay=weight_decay)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
optimizer2 = torch.optim.Adam(net_L.parameters(), lr=lr, weight_decay=weight_decay)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

trainloader = DataLoader(SICE_training(),batch_size=batch_size, shuffle=True, num_workers=0)

exppath=os.path.join("exp",time.strftime("%m_%d_%H_%M", time.localtime()))
os.makedirs(exppath,exist_ok=True)
os.makedirs(os.path.join(exppath,'model'),exist_ok=True)
prev_time = time.time()

for epoch in range(num_epochs):

    net_fusion.train()
    net_L.train()

    losslist_total=[]
    losslist_recon=[]
    losslist_smooth=[]
    losslist_initialize=[]
    losslist_suppress=[]
    losslist_consist=[]

    for i, (img1,img2,img3,index) in enumerate(trainloader):

        img1= img1.cuda()
        img2= img2.cuda()
        img3= img3.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        L1,L2,L3=net_L(img1),net_L(img2),net_L(img3)

        y3,R3,Rhat=net_fusion(img1,img2,L3)

        loss_recon=F.l1_loss(y3,img3)
        loss_smooth=illu_smooth(L3,img3)
        loss_initialize=F.l1_loss(L3,torch.max(img3,1,keepdim=True)[0])
        loss_suppress=torch.mean(F.relu(Rhat*(L3.detach())-img3)+F.relu(Rhat*(L2.detach())-img2)+F.relu(Rhat*(L1.detach())-img1))
        loss_consist=F.l1_loss(Rhat,R3)
        
        loss_total=coef[0]*loss_recon+coef[1]*loss_smooth+coef[2]*loss_initialize+coef[3]*loss_suppress+coef[4]*loss_consist

        loss_total.backward()

        optimizer1.step()
        optimizer2.step()

        losslist_total.append(loss_total.item())
        losslist_recon.append(loss_recon.item())
        losslist_smooth.append(loss_smooth.item())
        losslist_initialize.append(loss_initialize.item())
        losslist_suppress.append(loss_suppress.item())
        losslist_consist.append(loss_consist.item())


        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [loss_total: %f] [loss_smooth: %f] [loss_recon: %f] [loss_initialize: %f] [loss_suppress: %f] [loss_consist: %f] ETA: %.10s"
            % (
                epoch+1,
                num_epochs,
                i,
                len(trainloader),
                np.mean(losslist_total),
                np.mean(losslist_smooth),
                np.mean(losslist_recon),
                np.mean(losslist_initialize),
                np.mean(losslist_suppress),
                np.mean(losslist_consist),
                time_left,
            )
        )


    # adjust the learning rate and save model                
    scheduler1.step()  
    scheduler2.step() 

    if epoch % 1 == 0:
        checkpoint = {
                    'net_fusion': net_fusion.state_dict(),
                    'net_L': net_L.state_dict(),
                    }
        torch.save(checkpoint, os.path.join(exppath,'model', 'ckpt_%s.pth' % (str(epoch+1))))







            

