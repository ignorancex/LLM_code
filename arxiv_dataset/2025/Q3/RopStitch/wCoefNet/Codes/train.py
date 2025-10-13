import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, build_output_model, Network, CoefNetwork
from dataset import TrainDataset, TestDataset
import glob
from loss import cal_ddmperception_loss, cal_ddmperception_loss_test
import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torchvision.models as models



last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model_coef')
# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def maskSSIM(image1, image2, mask):
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)

    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim

def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    coef_net = CoefNetwork()
    net = Network()
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
        coef_net = coef_net.cuda()
        vgg_model = vgg_model.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(coef_net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 1: load homo net
    checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(checkpoint['model'])
    print('load homo model from {}!'.format(args.woCoefNet_path))
    
    # 2: load coef net
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        coef_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load coef model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training coef mdoel from stratch!')


    print('net:',net)
    print('coef_net:',coef_net)


    print("##################start training#######################")
    score_print_fre = 1000
    best_midplane_loss = 1e10

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        midplane_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))


        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()

            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, coef_net, inpu1_tesnor, inpu2_tesnor)
        
            # result: DDM
            DDM_ref_list = batch_out['DDM_ref_list']
            DDM_tgt_list = batch_out['DDM_tgt_list']


            # calculate loss for mid-plane (constrained from homo vector): vector_ref + vector_tgt = 0
            midplane_loss = cal_ddmperception_loss(vgg_model, inpu1_tesnor, inpu2_tesnor, DDM_ref_list, DDM_tgt_list) * 10

            # calculate loss for overlapping regions
            overlap_loss = midplane_loss * 0

            # calculate loss for non-overlapping regions
            nonoverlap_loss = midplane_loss * 0

            total_loss = overlap_loss + midplane_loss + nonoverlap_loss
            total_loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(coef_net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            midplane_loss_sigma += midplane_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_midplane_loss = midplane_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                midplane_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f} Mid-plane Loss: {:.4f} Non-overlap Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader), average_loss, average_overlap_loss, average_midplane_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                # writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                # writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                # writer.add_image("output_2H", ((output_H_ref[0,0:3,:,:] + output_H_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)
                # writer.add_image("output_2Mesh", ((output_tps_ref[0,0:3,:,:] + output_tps_tgt[0,0:3,:,:])/2 + 1.)/2., glob_iter)

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('mdiplane loss', average_midplane_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

            glob_iter += 1


        scheduler.step()
        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_coefmodel.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': coef_net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)

        if best_midplane_loss > midplane_loss_sigma:
            best_midplane_loss = midplane_loss_sigma
            filename ='best_coefmodel.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': coef_net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter, 'best_midplane_loss': best_midplane_loss}
            torch.save(state, model_save_path)

        # testing
        if True:
            #ssim1_list = []
            # ssim2_list = []
            ssim3_list = []
            planeloss_test_list = []
            print("----------- starting testing ----------")
            net.eval()
            for i, batch_value in enumerate(test_loader):
                #if i%100 == 0:
                if True:

                    input1_tensor = batch_value[0].float()
                    input2_tensor = batch_value[1].float()

                    if torch.cuda.is_available():
                        input1_tensor = input1_tensor.cuda()
                        input2_tensor = input2_tensor.cuda()

                    with torch.no_grad():
                        batch_out,flag_check = build_output_model(net, coef_net, input1_tensor, input2_tensor)
                        if not flag_check:
                            print("image idx:{}, warp size is too huge {}! use the original image pairs.".format(i+1, batch_out))
                            # batch_out = {}
                            # mask = torch.ones_like(input1_tesnor).to(input1_tesnor.device)
                            # batch_out['output_ref'] = torch.cat((input1_tesnor+1, mask), 1)
                            # batch_out['output_tgt'] = torch.cat((input2_tesnor+1, mask), 1)
                            continue

                        output_tps_ref = batch_out['output_ref']
                        output_tps_tgt = batch_out['output_tgt']
                        #################
                        DDM_ref = batch_out['DDM_ref']
                        DDM_tgt = batch_out['DDM_tgt']
                        midplane_loss_test = cal_ddmperception_loss_test(vgg_model, input1_tensor, input2_tensor, DDM_ref, DDM_tgt)
                        planeloss_test_list.append(midplane_loss_test.item())
                        #######################

                        # # SSIM 1
                        # input1_np = ((input1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                        # output = ((output_H[0,0:3,...]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                        # overlap_mask = output_H[0,3:6,...].cpu().detach().numpy().transpose(1,2,0)
                        # ssim1 = skimage.measure.compare_ssim(input1_np*overlap_mask, output*overlap_mask, data_range=255, multichannel=True)
                        # SSIM 2
                        # output_ref = (output_H_ref[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
                        # output_tgt = (output_H_tgt[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
                        # overlap_mask = output_H_ref[0,3:6,...] * output_H_tgt[0,3:6,...]
                        # overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
                        #ssim2 = skimage.measure.compare_ssim(output_ref*overlap_mask, output_tgt*overlap_mask, data_range=255, multichannel=True)
                        # ssim2 = maskSSIM(output_ref, output_tgt, overlap_mask)

                        # SSIM 3
                        output_ref = (output_tps_ref[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
                        output_tgt = (output_tps_tgt[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
                        overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
                        overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
                        ssim3 = maskSSIM(output_ref, output_tgt, overlap_mask)
                        ssim3_list.append(ssim3)
                        #torch.cuda.empty_cache()

            #writer.add_scalar('SSIM1', np.mean(ssim1_list), epoch+1)
            # writer.add_scalar('SSIM2', np.mean(ssim2_list), epoch+1)
            writer.add_scalar('SSIM3', np.mean(ssim3_list), epoch+1)
            writer.add_scalar('planeloss_test', np.mean(planeloss_test_list), epoch+1)

    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--woCoefNet_path', type=str, default='/home/my123/pyProjects/RopStitch_FinalCode/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--train_path', type=str, default='/media/my123/0d52819b-6878-4445-b5be-37548de0a05d/pyProjects/StitchUDIS/UDIS-D/training/')
    parser.add_argument('--test_path', type=str, default='/media/my123/0d52819b-6878-4445-b5be-37548de0a05d/pyProjects/StitchUDIS/UDIS-D/testing/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)


