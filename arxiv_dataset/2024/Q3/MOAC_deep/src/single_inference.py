import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np
from restore_module import pre_deconv,wiener_deconv,TinyUNet
from networks.fcn import FCN    # optional for experiments
# PIL read image
# y_img = y_img/255.
# y_samples = y_img[:,:,0] +1j*y_img[:,:,1] 
# y_samples = y_samples[:L_symbols*M_users].reshape(L_symbols,M_users).T    # 从序列读入

def inference(y_samples,pth_tar_location,M_users, L_symbols, h_coff, SNR_db):
    ''' 
    Restore summed samples(L,) from misaligned y(MxL+M-1,)
    ---
    input_samples: M*(L+1)-1长复数序列
    pth_tar_location: 模型文件地址
    M_users: 用户数
    L_symbols: 每个用户发送数据长度
    h_coff: 信道增益，需要已知
    SNR_db: 信道质量，需要已知

    return: L长序列, M*(L+1)-1长复数序列
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model2 = TinyUNet(4,1,bilinear=True).to(device)
    model2 = FCN(4,1).to(device)
    model2.load_state_dict(torch.load(pth_tar_location, map_location='cpu'))
    # y_samples = (y_samples-np.min(y_samples))/(np.max(y_samples)-np.min(y_samples)) # 归一化
    _,s_deconv = pre_deconv(y_samples, M_users, L_symbols, h_coff, SNR_db=SNR_db)
    # s_sum_est = np.sum(s_deconv,axis=0) # debug
    # s_estimate = s_deconv   # debug
    y_out_mat = y_samples[:M_users*L_symbols].reshape(L_symbols,M_users).T
    # s_deconv = np.array([[s_deconv.real,s_deconv.imag, y_out_mat.real, y_out_mat.imag]], dtype=float)    # 1x2xHxW
    s_deconv = np.array([[y_out_mat.real, y_out_mat.imag, s_deconv.real,s_deconv.imag, ]], dtype=float)    # 1x4xHxW
    # factor = np.max(s_deconv)
    # s_deconv = (s_deconv-np.min(s_deconv))/(np.max(s_deconv)-np.min(s_deconv)) # 归一化
    s_deconv = torch.from_numpy(s_deconv).float().to(device)
    model2.eval()
    with torch.no_grad():
        s_estimate = model2(s_deconv) 
        # s_estimate,_,_,_ = model2(s_deconv,s_deconv[:,0,:,:].unsqueeze(1)) # for DAE only
    s_estimate = s_estimate.squeeze().cpu().detach().numpy()
    s_sum_est = np.sum(s_estimate,axis=0)  # (W,)
    s_sum_est = s_sum_est + 0j*s_sum_est    # 目前只接受实部调制数据
    # s_sum_est = s_sum_est*factor    # 反归一化
    return s_sum_est,s_estimate