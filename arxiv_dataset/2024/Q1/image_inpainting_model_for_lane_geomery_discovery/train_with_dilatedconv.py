import json
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator
from datasets import ImageDataset
from losses import completion_network_loss
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)
from modified_models import CompletionNetwork_ModifiedDilation #SJ_TEST



#=====================================================
# Train with various Dilated Convolution strategy 
#=====================================================

from torchvision.utils import save_image #SJ_TEST
import torch
from torch.utils.tensorboard import SummaryWriter
import sys

writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('result_dir')
#ORG parser.add_argument('--data_parallel', action='store_true')
parser.add_argument('--data_parallel', action='store_true', default=True) #SJ_TEST : train on multiple GPUs
parser.add_argument('--recursive_search', action='store_true', default=False)
parser.add_argument('--init_model_cn', type=str, default=None)
parser.add_argument('--init_model_cd', type=str, default=None)
parser.add_argument('--steps_1', type=int, default=90000) #ORG 90000  
parser.add_argument('--steps_2', type=int, default=10000) #ORG 10000 
parser.add_argument('--steps_3', type=int, default=400000) #ORG 400000
parser.add_argument('--snaperiod_1', type=int, default=10000) #ORG 10000 
parser.add_argument('--snaperiod_2', type=int, default=5000) #ORG 2000 
parser.add_argument('--snaperiod_3', type=int, default=10000) #ORG 10000 
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--hole_min_w', type=int, default=90) # ORG 48
parser.add_argument('--hole_max_w', type=int, default=160) # ORG 96
parser.add_argument('--hole_min_h', type=int, default=90) # ORG 48
parser.add_argument('--hole_max_h', type=int, default=160) # ORG 96
parser.add_argument('--cn_input_size', type=int, default=256) #ORG default = 160 #SJ_COMMENT : complete network input size
parser.add_argument('--ld_input_size', type=int, default=160) #SJ_COMMENT : ld = local discriminator? 
parser.add_argument('--bsize', type=int, default=8)
parser.add_argument('--bdivs', type=int, default=1)
parser.add_argument('--num_test_completions', type=int, default=6)
parser.add_argument('--mpv', nargs=3, type=float, default=None)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--arc', type=str, choices=['celeba', 'places2'], default='celeba') # ORG : what is arc

# save intermediate image
def tensor_to_image(images_tensor, image_name):
  img1 = images_tensor[0]
  save_image(img1, image_name+'.png') 
  
  

def main(args):
    # ================================================
    # Preparation
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:1')
    
    # create result directory (if necessary)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # load dataset
    trnsfm = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset... (it may take a few minutes)')
    train_dset = ImageDataset(
        os.path.join(args.data_dir, 'train'),
        trnsfm,
        recursive_search=args.recursive_search)
    test_dset = ImageDataset(
        os.path.join(args.data_dir, 'test'),
        trnsfm,
        recursive_search=args.recursive_search)
    train_loader = DataLoader(
        train_dset,
        batch_size=(args.bsize // args.bdivs),
        shuffle=True)

    # compute mpv (mean pixel value) of training dataset
    if args.mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(train_dset.imgpaths),
            desc='computing mean pixel value of training dataset...')
        for imgpath in train_dset.imgpaths:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dset.imgpaths)
        pbar.close()
    else:
        mpv = np.array(args.mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = vars(args)
    args_dict['mpv'] = mpv_json
    with open(os.path.join(
            args.result_dir, 'config.json'),
            mode='w') as f:
        json.dump(args_dict, f)

 
    # make mpv & alpha tensors
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(gpu)
    alpha = torch.tensor(
        args.alpha,
        dtype=torch.float32).to(gpu)

    # to save model only best model
    num_dataset = len(train_dset.imgpaths)
    max = sys.maxsize
    min_loss_of_CN = max 
    min_loss_of_CD = max
    min_loss_of_ph3_CN = max
    min_loss_of_ph3_CD = max
    

    # ================================================
    # Training Phase 1
    # ================================================
    # load completion network
    #ORG model_cn = CompletionNetwork()
    model_cn = CompletionNetwork_ModifiedDilation()
    if args.init_model_cn is not None:
        model_cn.load_state_dict(torch.load(
            args.init_model_cn,
            map_location='cpu'))
    if args.data_parallel:
        #ORG model_cn = DataParallel(model_cn)
        model_cn = DataParallel(model_cn, gpu_ids = [0,1]) #SJ_TEST
    model_cn = model_cn.to(gpu)
    opt_cn = Adadelta(model_cn.parameters())

      
    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_1)
    while pbar.n < args.steps_1:
        
        x_index = 0
        x_sum = 0
        for x in train_loader:
            x_index = x_index + 1
            x_sum = x_sum + len(x)
            # forward
            x = x.to(gpu)
            #tensor_to_image(x, '1_x') # SJ_TEST
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=gen_hole_area(
                    (args.ld_input_size, args.ld_input_size),
                    (x.shape[3], x.shape[2])),
                max_holes=args.max_holes,
            ).to(gpu)
            #tensor_to_image(mask, '2_mask') # SJ_TEST
            x_mask = x - x * mask + mpv * mask
            test1 = x * mask
            #tensor_to_image(test1, '3_x_multi_mask') # SJ_TEST
            test11 = x - x * mask 
            #tensor_to_image(test11, '3_x_minus_x_multi_mask') # SJ_TEST
            test2 = mpv * mask
            #tensor_to_image(test2, '3_mpv_multi_mask') # SJ_TEST        
            #tensor_to_image(x_mask, '3_x_mask') # SJ_TEST
            
            input = torch.cat((x_mask, mask), dim=1)
            #print('debug : input = torch.cat((x_mask, mask), dim=1) ', input) 
            #tensor_to_image(input, '4_input') # SJ_TEST
            #tensor_to_image(input[0], '4_input[0]') #SJ_TEST
            test3 = input[0, 1:]
            #tensor_to_image(test3, '4_test3') #SJ_TEST
            test4 = input[0, 0:1]
            #tensor_to_image(test4, '4_test4') #SJ_TEST
            test5 = input[0, 3:]
            #tensor_to_image(test5, '4_test5') #SJ_TEST
    
            #print('debug : input.shape >> ', input.shape) #SJ_TEST
            output = model_cn(input)
            #tensor_to_image(output, '5_output') # SJ_TEST
            loss = completion_network_loss(x, output, mask)
            #print('debug : loss --> ', loss)

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description('phase 1 | train loss: %.5f' % loss.cpu())
                pbar.update()
                writer.add_scalar("train_loss_phase1", loss, pbar.n)

                # test
                if pbar.n % args.snaperiod_1 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        #tensor_to_image(x, 'test_1_x') # SJ_TEST        
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        #tensor_to_image(mask, 'test_2_mask') # SJ_TEST         
                        x_mask = x - x * mask + mpv * mask
                        #tensor_to_image(x_mask, 'test_3_x_mask') # SJ_TEST     
                        input = torch.cat((x_mask, mask), dim=1)
                        #tensor_to_image(input, 'test_4_input') # SJ_TEST     
                        output = model_cn(input)
                        #tensor_to_image(output, 'test_5_output') # SJ_TEST     
                        completed = poisson_blend(x_mask, output, mask)
                        #tensor_to_image(completed, 'test_6_completed') # SJ_TEST     
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            completed.cpu()), dim=0)
                        #tensor_to_image(imgs, 'test_7_imgs') # SJ_TEST         
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_1',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_1',
                            'model_cn_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        #if args.data_parallel:
                        if args.data_parallel and min_loss_of_CN > loss :
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                        min_loss_of_CN = loss        
                    model_cn.train()
                if pbar.n >= args.steps_1:
                    break
    pbar.close()
    print('[debug] x_index : ', x_index)
    print('[debug] x_sum : ', x_sum)
    
    # ================================================
    # Training Phase 2
    # ================================================
    # load context discriminator
    model_cd = ContextDiscriminator(
        local_input_shape=(3, args.ld_input_size, args.ld_input_size),
        global_input_shape=(3, args.cn_input_size, args.cn_input_size),
        arc=args.arc)
    if args.init_model_cd is not None:
        model_cd.load_state_dict(torch.load(
            args.init_model_cd,
            map_location='cpu'))
    if args.data_parallel:
        #ORG model_cd = DataParallel(model_cd)
        model_cd = DataParallel(model_cd, gpu_ids = [0,1]) #SJ_TEST
    model_cd = model_cd.to(gpu)
    opt_cd = Adadelta(model_cd.parameters())
    bceloss = BCELoss()

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        x2_index = 0
        x2_sum = 0
        
        for x in train_loader:
            x2_index = x2_index + 1
            x2_sum = x2_sum + len(x)
            # fake forward
            x = x.to(gpu)
            #tensor_to_image(x, 'ph2_1_x') #SJ_TEST
            hole_area_fake = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=hole_area_fake,
                max_holes=args.max_holes).to(gpu)
            #tensor_to_image(mask, 'ph2_2_mask') #SJ_TEST    
            fake = torch.zeros((len(x), 1)).to(gpu)
            #print('debug : fake ==> ', fake) #SJ_TEST
            x_mask = x - x * mask + mpv * mask
            #tensor_to_image(x_mask, 'ph2_4_x_mask') #SJ_TEST
            
            input_cn = torch.cat((x_mask, mask), dim=1)
            #tensor_to_image(input_cn, 'ph2_5_input_cn') #SJ_TEST
            
            output_cn = model_cn(input_cn)
            #tensor_to_image(output_cn, 'ph2_6_output_cn') #SJ_TEST
            input_gd_fake = output_cn.detach()
            #tensor_to_image(input_gd_fake, 'ph2_7_input_gd_fake') #SJ_TEST
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            #tensor_to_image(input_ld_fake, 'ph2_8_input_ld_fake') #SJ_TEST
            output_fake = model_cd((
                input_ld_fake.to(gpu),
                input_gd_fake.to(gpu)))
            #print('debug output_fake -->', output_fake) #SJ_TEST    
            loss_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1)).to(gpu)
            input_gd_real = x
            #tensor_to_image(input_gd_real, 'ph2_9_input_gd_real') #SJ_TEST
            input_ld_real = crop(input_gd_real, hole_area_real)
            #tensor_to_image(input_ld_real, 'ph2_10input_ld_real') #SJ_TEST
            output_real = model_cd((input_ld_real, input_gd_real))
            #print('debug output_read --> ', output_real) #SJ_TEST
            loss_real = bceloss(output_real, real)

            # reduce
            loss = (loss_fake + loss_real) / 2.

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cd.step()
                opt_cd.zero_grad()
                pbar.set_description('phase 2 | train loss: %.5f' % loss.cpu())
                pbar.update()
                writer.add_scalar("train_loss_phase2", loss, pbar.n)

                # test
                if pbar.n % args.snaperiod_2 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'step%d.png' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        #if args.data_parallel:
                        if args.data_parallel and min_loss_of_CD > loss :
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                        else:
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                        min_loss_of_CD = loss        
                    model_cn.train()
                if pbar.n >= args.steps_2:
                    break
    pbar.close()
    print('[debug] x2_index : ', x2_index)
    print('[debug] x2_sum : ', x2_sum)

    # ================================================
    # Training Phase 3
    # ================================================
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3:
        x3_index = 0
        x3_sum = 0
       
        for x in train_loader:
            x3_index = x3_index + 1
            x3_sum = x3_sum + len(x)
            # forward model_cd
            x = x.to(gpu)
            #tensor_to_image(x, "p3_1_x") #SJ_TEST
            hole_area_fake = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            mask = gen_input_mask(
                shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                hole_size=(
                    (args.hole_min_w, args.hole_max_w),
                    (args.hole_min_h, args.hole_max_h)),
                hole_area=hole_area_fake,
                max_holes=args.max_holes).to(gpu)
            #tensor_to_image(mask, "p3_2_mask") #SJ_TEST
            # fake forward
            fake = torch.zeros((len(x), 1)).to(gpu)
            #print('debug fake : ', fake) #SJ_TEST
            x_mask = x - x * mask + mpv * mask
            #tensor_to_image(x_mask, "p3_4_x_mask") #SJ_TEST
            input_cn = torch.cat((x_mask, mask), dim=1)
            #tensor_to_image(input_cn, "p3_5_input_cn") #SJ_TEST
            output_cn = model_cn(input_cn)
            #tensor_to_image(output_cn, "p3_6_output_cn") #SJ_TEST
            input_gd_fake = output_cn.detach()
            #tensor_to_image(input_gd_fake, "p3_7_input_gd_fake") #SJ_TEST
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            #tensor_to_image(input_ld_fake, "p3_8_input_ld_fake") #SJ_TEST
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            #print('debug output_fake : ', output_fake) #SJ_TEST
            loss_cd_fake = bceloss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(
                (args.ld_input_size, args.ld_input_size),
                (x.shape[3], x.shape[2]))
            real = torch.ones((len(x), 1)).to(gpu)
            input_gd_real = x
            #tensor_to_image(input_gd_real, "p3_9_input_gd_real") # SJ_TEST
            input_ld_real = crop(input_gd_real, hole_area_real)
            #tensor_to_image(input_ld_real, "p3_10_input_ld_real") #SJ_TEST
            output_real = model_cd((input_ld_real, input_gd_real))
            #print('debug output_real : ', output_real) #SJ_TEST
            loss_cd_real = bceloss(output_real, real)

            # reduce
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.

            # backward model_cd
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                # optimize
                opt_cd.step()
                opt_cd.zero_grad()

            # forward model_cn
            loss_cn_1 = completion_network_loss(x, output_cn, mask)
            input_gd_fake = output_cn
            #tensor_to_image(input_gd_fake, "p3_12_input_gd_fake") 
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            #tensor_to_image(input_ld_fake, "p3_13_input_ld_fake")
            output_fake = model_cd((input_ld_fake, (input_gd_fake)))
            loss_cn_2 = bceloss(output_fake, real)

            # reduce
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.

            # backward model_cn
            loss_cn.backward()
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                pbar.set_description(
                    'phase 3 | train loss (cd): %.5f (cn): %.5f' % (
                        loss_cd.cpu(),
                        loss_cn.cpu()))
                pbar.update()
                writer.add_scalar("train_loss_cd_phase3", loss_cd, pbar.n)
                writer.add_scalar("train_loss_cn_phase3", loss_cn, pbar.n)

                # test
                if pbar.n % args.snaperiod_3 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        x = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions).to(gpu)
                        mask = gen_input_mask(
                            shape=(x.shape[0], 1, x.shape[2], x.shape[3]),
                            hole_size=(
                                (args.hole_min_w, args.hole_max_w),
                                (args.hole_min_h, args.hole_max_h)),
                            hole_area=gen_hole_area(
                                (args.ld_input_size, args.ld_input_size),
                                (x.shape[3], x.shape[2])),
                            max_holes=args.max_holes).to(gpu)
                        x_mask = x - x * mask + mpv * mask
                        input = torch.cat((x_mask, mask), dim=1)
                        output = model_cn(input)
                        completed = poisson_blend(x_mask, output, mask)
                        imgs = torch.cat((
                            x.cpu(),
                            x_mask.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cn_step%d' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(x))
                        #if args.data_parallel:
                        if args.data_parallel and min_loss_of_ph3_CN > loss_cn and min_loss_of_ph3_CD > loss_cd:
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                            min_loss_of_ph3_CN = loss_cn
                            min_loss_of_ph3_CD = loss_cd    
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                    model_cn.train()
                if pbar.n >= args.steps_3:
                    break
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)
    main(args)
