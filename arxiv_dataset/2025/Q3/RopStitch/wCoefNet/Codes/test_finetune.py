import argparse
import torch
from collections import OrderedDict
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2
#from torch_homography_model import build_model
from network import get_stitched_result, Network, CoefNetwork, build_new_ft_model
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import glob
from loss import cal_lp_loss2
import torchvision.transforms as T

#import PIL
resize_512 = T.Resize((512,512))

# path of project
#nl: os.path.dirname("__file__") ----- the current absolute path
#nl: os.path.pardir ---- the last path
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


#nl: path to save the model files
# MODEL_DIR = os.path.join(last_path, 'model_homo')
MODEL_COEF_DIR = os.path.join(last_path, 'model_coef')


#nl: create folders if it dose not exist
# if not os.path.exists(MODEL_COEF_DIR):
#     os.makedirs(MODEL_COEF_DIR)

def maskSSIM(image1, image2, mask):
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)
    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim

def maskPSNR(image1, image2, mask):
    image1 = image1 * mask / 255
    image2 = image2 * mask / 255
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / mask.sum())
    psnr = 20 * np.log10(1 / rmse)
    return psnr


def loadSingleData( img1_name, img2_name):

    # load image1
    input1 = cv2.imread(img1_name)
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0

    # load image2
    input2 = cv2.imread(img2_name)
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0

    max_range = 2000
    if max(input1.shape[0],input1.shape[1]) > max_range:
        scale_width = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[0])
        scale_hight = int((max_range / max(input1.shape[1],input1.shape[0])) * input1.shape[1])
        input1 = cv2.resize(input1,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)
        input2 = cv2.resize(input2,(scale_hight,scale_width), interpolation=cv2.INTER_AREA)

    if input1.shape != input2.shape:
        input2 = cv2.resize(input2,(input1.shape[1],input1.shape[0]), interpolation=cv2.INTER_AREA)

    input1 = np.transpose(input1, [2, 0, 1])
    input2 = np.transpose(input2, [2, 0, 1])
    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)


def test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt):
    with torch.no_grad():
        output,check_flag = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)

    if not check_flag:
        print("image idx:{}, warp size is too huge! pass this image.")
        output = {}
        mask = torch.ones_like(input1_tensor).to(input1_tensor.device)
        output['output_ref'] =  torch.cat((input1_tensor + 1, mask), 1) 
        output['output_tgt'] =  torch.cat((input2_tensor + 1, mask), 1) 
        img1 = output['output_ref'][0,0:3,...]
        img2 = output['output_tgt'][0,0:3,...]
        output['stitched']  =  img1*(img1/(img1+img2+1e-6)) + img2*(img2/(img1+img2+1e-6))


    output_tps_ref = output['output_ref']
    output_tps_tgt = output['output_tgt']
    stitch_result = output['stitched'].cpu().detach().numpy().transpose(1,2,0) *127.5
    warp1 = output_tps_ref[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5
    warp2 = output_tps_tgt[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5

    overlap_mask = output_tps_ref[0,3:6,...]  * output_tps_tgt[0,3:6,...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
    ssim3 = maskSSIM(warp1, warp2, overlap_mask)
    psnr3 = maskPSNR(warp1, warp2, overlap_mask)
    return ssim3,psnr3,stitch_result

def test_once_original(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt):
    print("image idx:{}, warp size is too huge! pass this image.")
    output = {}
    mask = torch.ones_like(input1_tensor).to(input1_tensor.device)
    output['output_ref'] =  torch.cat((input1_tensor + 1, mask), 1) 
    output['output_tgt'] =  torch.cat((input2_tensor + 1, mask), 1) 
    img1 = output['output_ref'][0,0:3,...]
    img2 = output['output_tgt'][0,0:3,...]
    output['stitched']  =  img1*(img1/(img1+img2+1e-6)) + img2*(img2/(img1+img2+1e-6))


    output_tps_ref = output['output_ref']
    output_tps_tgt = output['output_tgt']
    stitch_result = output['stitched'].cpu().detach().numpy().transpose(1,2,0) *127.5
    warp1 = output_tps_ref[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5
    warp2 = output_tps_tgt[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5

    overlap_mask = output_tps_ref[0,3:6,...]  * output_tps_tgt[0,3:6,...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
    ssim3 = maskSSIM(warp1, warp2, overlap_mask)
    psnr3 = maskPSNR(warp1, warp2, overlap_mask)
    return ssim3,psnr3,stitch_result


def ternary_search(net, coef_net, input1, input2, low=0.0, high=1.0, max_iter=20,  max_out_height=4000, tol=1e-6):

    best_alpha = None
    best_ssim = -float('inf')
    best_psnr = -float('inf')
    best_img = None
    
    for _ in range(max_iter):
        if high - low < tol:
            break
        
        # 计算两个三分点
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        
        # 计算 mid1 和 mid2 的 SSIM
        input1_tensor_512 = resize_512(input1)
        input2_tensor_512 = resize_512(input2)
        with torch.no_grad():
            batch_out = build_new_ft_model(net, coef_net, input1_tensor_512, input2_tensor_512, alpha=mid1)
            rigid_mesh = batch_out['rigid_mesh']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']
            ssim1, psnr1, img1 = test_once(input1, input2, rigid_mesh, mesh_ref, mesh_tgt)

        with torch.no_grad():
            batch_out = build_new_ft_model(net, coef_net, input1_tensor_512, input2_tensor_512, alpha= mid2)
            rigid_mesh = batch_out['rigid_mesh']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']
            ssim2, psnr2, img2 = test_once(input1, input2, rigid_mesh, mesh_ref, mesh_tgt)
        
        # print('mid1:{}, ssim1:{}, mid2:{}, ssim2:{}'.format(mid1,ssim1,mid2,ssim2))
        # 更新最优值
        if ssim1 > best_ssim:
            best_ssim = ssim1
            best_psnr = psnr1
            best_alpha = mid1
            best_img = img1
        if ssim2 > best_ssim:
            best_ssim = ssim2
            best_psnr = psnr2
            best_alpha = mid2
            best_img = img2
        
        # 调整搜索区间
        if ssim1 < ssim2:
            low = mid1  # 峰值在 [mid1, high]
        else:
            high = mid2  # 峰值在 [low, mid2]

    return best_alpha, best_ssim, best_psnr, best_img

def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define the network
    net = Network()
    coef_net = CoefNetwork()
    if torch.cuda.is_available():
        net = net.cuda()
        coef_net = coef_net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if they are exist
    # homo model
    homo_checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(homo_checkpoint['model'])
    optimizer.load_state_dict(homo_checkpoint['optimizer'])
    start_epoch = homo_checkpoint['epoch']
    scheduler.last_epoch = start_epoch
    print('load homo net model from {}!'.format(args.woCoefNet_path))    

    # coef model
    coef_ckpt_list = glob.glob(MODEL_COEF_DIR + "/*.pth")
    coef_ckpt_list.sort()
    if len(coef_ckpt_list) != 0:
        coef_model_path = coef_ckpt_list[-1]
        coef_checkpoint = torch.load(coef_model_path)

        coef_net.load_state_dict(coef_checkpoint['model'])
        print('load coef model from {}!'.format(coef_model_path))
    else:
        start_epoch = 0
        print('training coef from stratch!')

    fintune_psnr_list = []
    fintune_ssim_list = []

    path_ave_other_fusion = '../ave_other_fusion_finetune/'
    if not os.path.exists(path_ave_other_fusion):
        os.makedirs(path_ave_other_fusion)

    # collection all data piars
    datas = OrderedDict()
    extensions = ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']
    
    datas_list = glob.glob(os.path.join(args.path, '*'))
    for data in sorted(datas_list):
        data_name = data.split('/')[-1]
        if data_name == 'input1' or data_name == 'input2' :
            datas[data_name] = {}
            datas[data_name]['path'] = data
            full_img_list = []
            for ex in extensions:
                full_img_list.extend(glob.glob(os.path.join(data, ex)))

            datas[data_name]['image'] = full_img_list
            datas[data_name]['image'].sort()

    for idx in range(len(datas['input1']['image'])):
        
        torch.cuda.empty_cache()

        net.load_state_dict(homo_checkpoint['model'])
        optimizer.load_state_dict(homo_checkpoint['optimizer'])
        start_epoch = homo_checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print('finetune for a new image, init homo model from {}!'.format(args.woCoefNet_path))

        coef_net.load_state_dict(coef_checkpoint['model'])
        print('finetune for a new image, init coef model from {}!'.format(coef_model_path))

        # load dataset(only one pair of images)
        input1_tensor, input2_tensor = loadSingleData(img1_name = datas['input1']['image'][idx], img2_name = datas['input2']['image'][idx])
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()

        input1_tensor_512 = resize_512(input1_tensor)
        input2_tensor_512 = resize_512(input2_tensor)

        loss_list = []

        print("##################start iteration {} #######################".format(idx+1))

        best_ssim, best_psnr, best_stitch_result = 0, 0, None
        for epoch in range(start_epoch, start_epoch + args.max_iter):
            net.train()

            optimizer.zero_grad()

            batch_out = build_new_ft_model(net, coef_net,  input1_tensor_512, input2_tensor_512)
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']
            rigid_mesh = batch_out['rigid_mesh']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']

            total_loss = cal_lp_loss2(output_tps_ref, output_tps_tgt)
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            scheduler.step()

            current_iter = epoch-start_epoch+1
            
            loss_list.append(total_loss)
            print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))


            # init check
            if current_iter == 1:
                # ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                best_alpha, ssim3,psnr3,stitch_result = ternary_search(net, coef_net, input1_tensor, input2_tensor, low=-1.0, high=2.0, max_iter=5,  max_out_height=4000, tol=1e-6)
                print('init best_alpha:{}, init ssim:{}, psnr:{}'.format(best_alpha,ssim3,psnr3))
       
                if ssim3 > best_ssim:
                    best_ssim = ssim3
                    best_psnr = psnr3
                    best_stitch_result = stitch_result

            if current_iter >= 4:
                if torch.abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= 1e-4 and torch.abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= 1e-4 \
                and torch.abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= 1e-4:
                    
                    # ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                    best_alpha, ssim3,psnr3,stitch_result = ternary_search(net, coef_net, input1_tensor, input2_tensor, low=-1.0, high=2.0, max_iter=5,  max_out_height=4000, tol=1e-6)
                    print('stop early, final best_alpha:{}, current ssim:{}, psnr:{}'.format(best_alpha,ssim3,psnr3))
                    
                    if ssim3 > best_ssim:
                        best_ssim = ssim3
                        best_psnr = psnr3
                        best_stitch_result = stitch_result

                    break

            if current_iter == args.max_iter:
                # ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                best_alpha, ssim3,psnr3,stitch_result = ternary_search(net, coef_net, input1_tensor, input2_tensor, low=-1.0, high=2.0, max_iter=5,  max_out_height=4000, tol=1e-6)
                print('final best_alpha:{}, final ssim:{}, psnr:{}'.format(best_alpha,ssim3,psnr3))
          
                if ssim3 > best_ssim:
                    best_ssim = ssim3
                    best_psnr = psnr3
                    best_stitch_result = stitch_result


            torch.cuda.empty_cache()

        print("##################end iteration {} #######################".format(idx+1))
        fintune_ssim_list.append(best_ssim)
        fintune_psnr_list.append(best_psnr)
        path = path_ave_other_fusion + str(idx+1).zfill(6) + ".jpg"
        cv2.imwrite(path, best_stitch_result)

    fintune_psnr_list.sort(reverse = True)
    list_len = len(fintune_psnr_list)
    print("top 30%", np.mean(fintune_psnr_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(fintune_psnr_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(fintune_psnr_list[int(list_len*0.6):]))
    print('average psnr:', np.mean(fintune_psnr_list))
    
    fintune_ssim_list.sort(reverse = True)
    print("top 30%", np.mean(fintune_ssim_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(fintune_ssim_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(fintune_ssim_list[int(list_len*0.6):]))
    print('average ssim:', np.mean(fintune_ssim_list))


if __name__=="__main__":

    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--woCoefNet_path', type=str, default='/home/my123/pyProjects/RopStitch_FinalCode/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--path', type=str, default='/media/my123/0d52819b-6878-4445-b5be-37548de0a05d/pyProjects/StitchUDIS/stitch_real/')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)

    #nl: rain
    train(args)


