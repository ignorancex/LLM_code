import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import matplotlib
import SimpleITK as sitk
from math import exp
import nibabel as nib
import nibabel.processing
import random
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def norm_to_0_1(img):
    return (img-img.min())/(img.max()-img.min())

class MyDataset_no_fix_v2(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3], self.data[idx][4], self.data[idx][5]
    
def load_data_no_fix_v2(set_name,batch_size,ifshuffle=True):
    
    cur_path = os.getcwd()
    set_path=cur_path+'/dataset/'+set_name
  
    data=np.load(set_path,allow_pickle=True)
    data_set = MyDataset_no_fix_v2(data)
    loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=batch_size, 
                                          shuffle=ifshuffle)
    return loader


def create_folder(path,folder_name):
    folder_path=path+'/'+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return

def create_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'w') as output:
        output.write(str(log) + '\n')
    return

def append_log(log,path,file_name):
    with open(path+"/"+file_name+".txt", 'a+') as output:
        output.write(str(log) + '\n')
    return

def save_sample_any(epoch,img_name,img,img_path):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    torch_img=img.squeeze().float()
    x,y,z=torch_img.shape

    torch_lr=torch_img.permute(0,1,2)
    torch_lr=torch_lr.view(x,1,y,z)

    torch_fb=torch_img.permute(2,0,1)
    torch_fb=torch_fb.view(y,1,x,z)
    torch_fb=torch_fb.permute(0, 1, 2,3).flip(2)

    torch_td=torch_img.permute(1,0,2)
    torch_td=torch_td.view(z,1,x,y)
    torch_td=torch_td.permute(0, 1, 2,3).flip(2)
    
    cat_image=torch.cat((torch_lr[x//2], torch_fb[y//2], torch_td[z//2]))
    cat_image=cat_image.view(3,1,x,y)
    
    name_img=str(epoch)+"_"+img_name+".png"
    image_o=np.transpose(vutils.make_grid(cat_image.to(device), nrow=3, normalize=True).cpu(),(1,2,0)).numpy()
    
    matplotlib.image.imsave(img_path+"/"+name_img,image_o)
    
    return

def save_nii_any(epoch,img_name,img,img_path):
    ref_img_GetOrigin=(0.0, 0.0, 0.0)
    ref_img_GetDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ref_img_GetSpacing=(1.0, 1.0, 1.0)

    img = sitk.GetImageFromArray(img.squeeze().cpu().detach().numpy())
    
    img.SetOrigin(ref_img_GetOrigin)
    img.SetDirection(ref_img_GetDirection)
    img.SetSpacing(ref_img_GetSpacing)
    
    name_img=img_name+"_"+str(epoch)+".nii.gz"
    
    sitk.WriteImage(img, img_path+"/"+name_img)
    
    return

def save_nii_any_v2(epoch,img_name,img,img_path,fixed_set_name,idx_val):
    if fixed_set_name == "CC359":
        fixed_path='./dataset/CC359_96/'
        fixed_name='S'+idx_val+"_"+str(0)+'raw'+'_'+str(96)+'.nii.gz'
        fixed_img=nibabel.load(fixed_path+fixed_name)

    
    elif fixed_set_name == "LPBA40":
        fixed_path='./dataset/LPBA40_96/'
        fixed_name='S'+idx_val+"_"+str(0)+'raw'+'_'+str(96)+'.nii.gz'
        fixed_img=nibabel.load(fixed_path+fixed_name)

    img_np=img.squeeze().cpu().detach().numpy()
    
    img = nib.Nifti1Image(img_np, fixed_img.affine, nib.Nifti1Header())
   
    name_img=img_name+"_"+str(epoch)+".nii.gz"
    
    nib.save(img, img_path+"/"+name_img)
    
    return

def save_nii_any_v3(idx_val,img_name,img,img_path,dataset_path,space):
    
    if space == "orig":
        fixed_name='sub-'+str(idx_val)+"_desc-preproc_T1w_resample_to_96.nii.gz"
        fixed_img=nibabel.load(dataset_path+fixed_name)
    
    elif space == "reg":
        fixed_name="mni_icbm152_t1_tal_nlin_asym_09c_brain_orig_96.nii"
        fixed_img=nibabel.load(dataset_path+fixed_name)

    img_np=img.squeeze().cpu().detach().numpy()
    
    img = nib.Nifti1Image(img_np, fixed_img.affine, nib.Nifti1Header())
   
    name_img=str(idx_val)+"_"+img_name+".nii.gz"
    
    nib.save(img, img_path+"/"+name_img)
    
    return


def f1_loss(y_true, y_pred, is_training=False):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def f1_loss_np(y_true, y_pred):

    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    
    return f1

def jaccard_loss(y_true, y_pred, is_training=False):

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    
    jaccard = tp/(tp+fp+fn+epsilon)
    jaccard.requires_grad = is_training
    return jaccard

def jaccard_loss_v2(y_true, y_pred, is_training=False):
    y_true = y_true.float()
    y_pred = y_pred.float()

    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    
    epsilon = 1e-7
    
    jaccard = tp / (tp + fp + fn + epsilon)
    jaccard.requires_grad = is_training
    return jaccard


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def compute_label_dice_v2(gt, pred, label):
    gt = gt.squeeze()
    pred = pred.squeeze()
    if label == "gm":
        cls_lst = [1., 2., 3.]
    elif label == "aal":
        cls_lst = [ 2001., 2002., 2101., 2102., 2111., 2112., 2201., 2202.,
                   2211., 2212., 2301., 2302., 2311., 2312., 2321., 2322., 2331.,
                   2332., 2401., 2402., 2501., 2502., 2601., 2602., 2611., 2612.,
                   2701., 2702., 3001., 3002., 4001., 4002., 4011., 4012., 4021.,
                   4022., 4101., 4102., 4111., 4112., 4201., 4202., 5001., 5002.,
                   5011., 5012., 5021., 5022., 5101., 5102., 5201., 5202., 5301.,
                   5302., 5401., 5402., 6001., 6002., 6101., 6102., 6201., 6202.,
                   6211., 6212., 6221., 6222., 6301., 6302., 6401., 6402., 7001.,
                   7002., 7011., 7012., 7021., 7022., 7101., 7102., 8101., 8102.,
                   8111., 8112., 8121., 8122., 8201., 8202., 8211., 8212., 8301.,
                   8302., 9001., 9002., 9011., 9012., 9021., 9022., 9031., 9032.,
                   9041., 9042., 9051., 9052., 9061., 9062., 9071., 9072., 9081.,
                   9082., 9100., 9110., 9120., 9130., 9140., 9150., 9160., 9170.]

    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return torch.mean(torch.FloatTensor(dice_lst))

def compute_label_jaccard_v2(gt, pred, label):
    gt = gt.squeeze()
    pred = pred.squeeze()
    if label == "gm":
        cls_lst = [1., 2., 3.]
    elif label == "aal":
        cls_lst = [ 2001., 2002., 2101., 2102., 2111., 2112., 2201., 2202.,
                   2211., 2212., 2301., 2302., 2311., 2312., 2321., 2322., 2331.,
                   2332., 2401., 2402., 2501., 2502., 2601., 2602., 2611., 2612.,
                   2701., 2702., 3001., 3002., 4001., 4002., 4011., 4012., 4021.,
                   4022., 4101., 4102., 4111., 4112., 4201., 4202., 5001., 5002.,
                   5011., 5012., 5021., 5022., 5101., 5102., 5201., 5202., 5301.,
                   5302., 5401., 5402., 6001., 6002., 6101., 6102., 6201., 6202.,
                   6211., 6212., 6221., 6222., 6301., 6302., 6401., 6402., 7001.,
                   7002., 7011., 7012., 7021., 7022., 7101., 7102., 8101., 8102.,
                   8111., 8112., 8121., 8122., 8201., 8202., 8211., 8212., 8301.,
                   8302., 9001., 9002., 9011., 9012., 9021., 9022., 9031., 9032.,
                   9041., 9042., 9051., 9052., 9061., 9062., 9071., 9072., 9081.,
                   9082., 9100., 9110., 9120., 9130., 9140., 9150., 9160., 9170.]

    jaccard_lst = []
    for cls in cls_lst:
        jaccard = jaccard_loss_v2(gt == cls, pred == cls)
        jaccard_lst.append(jaccard)
    return torch.mean(torch.FloatTensor(jaccard_lst))

def get_ave_std(data):
    ave=torch.mean(torch.FloatTensor(data))
    std=torch.std(torch.FloatTensor(data))
    return ave,std


###SSMI
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)



def show_fmri_centroid(img):
    x,y,z=img.shape
    torch_img = torch.from_numpy(img)
    torch_img=torch_img.type(torch.FloatTensor)
    
    torch_lr=torch_img.permute(0,1,2)
    torch_lr=torch_lr.view(x,1,y,z)
    torch_lr=torch_lr.permute(0, 1, 3,2).flip(2)

    torch_fb=torch_img.permute(1,0,2)
    torch_fb=torch_fb.view(y,1,x,z)
    torch_fb=torch_fb.permute(0, 1, 3,2).flip(2)

    torch_td=torch_img.permute(2,0,1)
    torch_td=torch_td.view(z,1,x,y)
    torch_td=torch_td.permute(0, 1, 3,2).flip(2)
    
    cat_image=torch.cat((torch_lr[x//2], torch_fb[y//2], torch_td[z//2]))
    cat_image=cat_image.view(3,1,x,y)
    
    plt.figure(figsize = (12,12))
    plt.imshow(np.transpose(vutils.make_grid(cat_image.to('cuda'), nrow=3, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    return 

def mutual_information(t1_slice,t2_slice):
    """ Mutual information for joint histogram
    """
    t1_slice=t1_slice.cpu().detach().numpy()
    t2_slice=t2_slice.cpu().detach().numpy()
    
    hgram, x_edges, y_edges = np.histogram2d(
     t1_slice.ravel(),
     t2_slice.ravel(),
     bins=2000)
    
    
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_information_np(t1_slice,t2_slice):
    """ Mutual information for joint histogram
    """
    t1_slice=t1_slice
    t2_slice=t2_slice
    
    hgram, x_edges, y_edges = np.histogram2d(
     t1_slice.ravel(),
     t2_slice.ravel(),
     bins=2000)
    
    
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def normalized_cross_correlation(tensor1, tensor2):

    if tensor1.shape != tensor2.shape:
        raise ValueError

    tensor1_mean_zero = tensor1 - tensor1.mean()
    tensor2_mean_zero = tensor2 - tensor2.mean()


    numerator = (tensor1_mean_zero * tensor2_mean_zero).sum()
    denominator = torch.sqrt((tensor1_mean_zero ** 2).sum() * (tensor2_mean_zero ** 2).sum())

    if denominator == 0:
        return 0

    ncc = numerator / denominator
    return ncc.item()


def am_1d_2_Nd_torch(am):
    am=am.squeeze()
    mask_label=torch.unique(am)
    
    multi_d=torch.zeros((len(mask_label),96,96,96))
    for i in range(len(mask_label)):
        label_idx=mask_label[i]
        multi_d[i][am==label_idx]=1.0
    return multi_d.unsqueeze(0).float().cuda()



def get_translation_3D(tx, ty, tz):
    t=torch.zeros(1, 4, 4)
    t[:, :, :4] = torch.tensor([[1, 0, 0, tx],
                                [0, 1, 0, ty],
                                [0, 0, 1, tz],
                                [0, 0, 0, 1]])
    return t
    
def get_rotate_3D_x(angle):
    r_x=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_x[:, :, :4] = torch.tensor([[1, 0, 0, 0],
                                  [0, np.cos(angle), -np.sin(angle), 0],
                                  [0, np.sin(angle),  np.cos(angle), 0],
                                  [0, 0, 0, 1]])
    return r_x

def get_rotate_3D_y(angle):
    r_y=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_y[:, :, :4] = torch.tensor([[np.cos(angle), 0, np.sin(angle), 0],
                                  [0, 1, 0, 0],
                                  [-np.sin(angle), 0,  np.cos(angle), 0],
                                  [0, 0, 0, 1]])
    return r_y

def get_rotate_3D_z(angle):
    r_z=torch.zeros(1, 4, 4)
    angle = np.radians(angle)
    r_z[:, :, :4] = torch.tensor([[np.cos(angle), -np.sin(angle), 0, 0],
                                  [np.sin(angle),  np.cos(angle), 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    return r_z

def get_scale_3D(sx,sy,sz):
    t=torch.zeros(1, 4, 4)
    t[:, :, :4] = torch.tensor([[sx, 0, 0, 0],
                                [0, sy, 0, 0],
                                [0, 0, sz, 0],
                                [0, 0, 0, 1]])
    return t


def random_affine_3D(img_size,degree,voxel,scale_min,scale_max):
    angle_x=random.uniform(-degree,degree)
    angle_y=random.uniform(-degree,degree)
    angle_z=random.uniform(-degree,degree)
    
    tx=random.uniform(-voxel,voxel)    
    ty=random.uniform(-voxel,voxel)    
    tz=random.uniform(-voxel,voxel) 

    sx=random.uniform(scale_min,scale_max)
    sy=random.uniform(scale_min,scale_max)
    sz=random.uniform(scale_min,scale_max)
    
    c=get_translation_3D(-img_size//2, -img_size//2,-img_size//2)
    c_inv=torch.inverse(c)
    
    r_x=get_rotate_3D_x(angle_x)
    r_y=get_rotate_3D_y(angle_y)
    r_z=get_rotate_3D_z(angle_z)
    
    t=get_translation_3D(tx, ty, tz)
    s=get_scale_3D(sx,sy,sz)

    A=c_inv@t@r_x@r_y@r_z@s@c
    return A

def param2theta(param, x,y,z):
    param = np.linalg.inv(param)
    theta = np.zeros([3,4])
    theta[0,0] = param[0,0]
    theta[0,1] = param[0,1]*y/x
    theta[0,2] = param[0,2]*z/x
    theta[0,3] = theta[0,0]+ theta[0,1] +  theta[0,2] + 2*param[0,3]/x -1

    theta[1,0] = param[1,0]*x/y
    theta[1,1] = param[1,1]
    theta[1,2] = param[1,2]*z/y
    theta[1,3] = theta[1,1]+ theta[1,2]+ 2*param[1,3]/y + theta[1,0] -1

    theta[2,0] = param[2,0]*x/z
    theta[2,1] = param[2,1]*y/z
    theta[2,2] = param[2,2]
    theta[2,3] = theta[2,2] + 2*param[2,3]/z + theta[2,0]+theta[2,1] -1

    return theta

def get_random_grid(img,degree,voxel,scale_min,scale_max):
    
    img_size=img.shape[0]

    A=random_affine_3D(img_size,degree,voxel,scale_min,scale_max)
    param=A.cpu().detach().numpy().reshape(4,4)
    theta=param2theta(param, img_size,img_size,img_size)
    theta_torch=torch.from_numpy(theta)
    theta_torch=theta_torch.view(-1,3,4)
    theta_torch=theta_torch.type(torch.FloatTensor)

    grid = F.affine_grid(theta_torch, torch.Size([1 ,1, img_size,img_size,img_size]),align_corners=True)
    
    return grid

def train_aug(img,label,img_size,degree,voxel,scale_min,scale_max):
    A=random_affine_3D(img_size,degree,voxel,scale_min,scale_max)
    param=A.cpu().detach().numpy().reshape(4,4)
    theta=param2theta(param, img_size,img_size,img_size)
    theta_torch=torch.from_numpy(theta)
    theta_torch=theta_torch.view(-1,3,4)
    theta_torch=theta_torch.type(torch.FloatTensor).cuda()
    grid = F.affine_grid(theta_torch, torch.Size([1 ,1, img_size,img_size,img_size]),align_corners=True).cuda()

    img_transformed = F.grid_sample(img, grid,mode='bilinear',align_corners=True,padding_mode="zeros")
    label_transformed = F.grid_sample(label, grid,mode='nearest',align_corners=True,padding_mode="zeros")
    
    return img_transformed,label_transformed

def train_aug_all(img, label_1, label_2, label_3, img_size, degree, voxel, scale_min, scale_max):
    A=random_affine_3D(img_size,degree,voxel,scale_min,scale_max)
    param=A.cpu().detach().numpy().reshape(4,4)
    theta=param2theta(param, img_size,img_size,img_size)
    theta_torch=torch.from_numpy(theta)
    theta_torch=theta_torch.view(-1,3,4)
    theta_torch=theta_torch.type(torch.FloatTensor).cuda()
    grid = F.affine_grid(theta_torch, torch.Size([1 ,1, img_size,img_size,img_size]),align_corners=True).cuda()

    img_transformed = F.grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode="zeros")
    label_1_transformed = F.grid_sample(label_1, grid, mode='nearest', align_corners=True, padding_mode="zeros")
    label_2_transformed = F.grid_sample(label_2, grid, mode='nearest', align_corners=True, padding_mode="zeros")
    label_3_transformed = F.grid_sample(label_3, grid, mode='nearest', align_corners=True, padding_mode="zeros")
    
    return img_transformed, label_1_transformed, label_2_transformed, label_3_transformed

def one_hot_1D_label(data):
    ### e.g. process tensor([   0., 2001., 2002., 2101., 2102., 2111., 2112., 2201., 2202., 2211.,2212., 2301., 2302.]) to 
    ### tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,])
    ### since the predicted am would be 0,1,2,3,4 
    ### correct for dice compute
    nd_data=am_1d_2_Nd_torch(data)
    onehot_1d_val=torch.argmax(nd_data,axis=1)
    return onehot_1d_val


def save_adjacency_matrix(epoch, img_name, adj_matrix_tensor, img_path):

    adj_matrix = adj_matrix_tensor.squeeze().cpu().detach().numpy()
    
    # Plot the adjacency matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, interpolation='none')
    plt.colorbar(label='Similarity value')
    plt.title("Adjacency Matrix")
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    
    # Adjusting the labels to start from 1 instead of 0
    num_nodes = adj_matrix.shape[0]
    plt.xticks(ticks=np.arange(0, num_nodes, step=10), labels=np.arange(1, num_nodes + 1, step=10))
    plt.yticks(ticks=np.arange(0, num_nodes, step=10), labels=np.arange(1, num_nodes + 1, step=10))
    
    plt.grid(visible=False)  # Turn off the grid that was shown in the uploaded image

    # Save the plot to disk
    file_name = f"{img_name}_{epoch}.png"
    np_file_name = f"{img_name}_{epoch}.npy"
    
    plt.savefig(os.path.join(img_path, file_name))
    plt.close()
    
    np.save(os.path.join(img_path, np_file_name),adj_matrix)
    
    return 

def show_adjacency_matrix(id_num, label, Predicted, adj_matrix_tensor):
    adj_matrix = adj_matrix_tensor.squeeze().cpu().detach().numpy()


    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, interpolation='none')
    plt.colorbar(label='Similarity value')
    plt.title(f"Adjacency Matrix - {id_num} label: {label} Predicted: {Predicted}")
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    

    num_nodes = adj_matrix.shape[0]
    plt.xticks(ticks=np.arange(0, num_nodes, step=10), labels=np.arange(1, num_nodes + 1, step=10))
    plt.yticks(ticks=np.arange(0, num_nodes, step=10), labels=np.arange(1, num_nodes + 1, step=10))
    
    plt.grid(visible=False) 

    plt.show()
    
    

