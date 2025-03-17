# Test Script
# %%
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from single_inference import inference
import numpy as np

## MAIN SETTINGS ##
M_users = 14
L_symbols = 256
loaded_image = Image.open('../test_images/data_1.png').convert('L')
loaded_array = np.array(loaded_image)
# d_data = (loaded_array.astype('float32') - 128) / 20.  # 将0-255的整数转换回小数
d_data = loaded_array.astype('float32') / 255.
# h_coff = np.exp(1j*np.random.uniform(0,1,(M_users,1)) * 4* np.pi /4)
# SNR_db_global = 0
mse_record = np.zeros((5,5,3))  # 储存结果，重要 -20dB-20dB的5组测试值，每组测三次
SNR_all = [-20,-10,0,10,20]
## UTILS ##
def awgn(x, snr):
    '''Add AWGN(complex)
    x: numpy array
    snr: int, dB
    '''
    len_x = x.flatten().shape[0]
    Ps = np.sum(np.abs(x)**2) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise_r = np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(Pn)/np.sqrt(2)
    noise_i = np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(Pn)/np.sqrt(2)
    noise = noise_r + 1j*noise_i
    return x + noise

def get_pn(x, snr):
    len_x = x.flatten().shape[0]
    Ps = np.sum(np.abs(x)**2) / len_x
    Pn = Ps / (np.power(10, snr / 10))    
    return Pn

# 用优化后的方法生成噪声信号：
def distortion_func(image:np.ndarray,h_coff,SNR_db=0):
    # 添加混叠，以及信道系数
#     np.random.seed()   # 保证噪声随机化
    # image = image.numpy()
#     image = (image-np.min(image))/(80.-np.min(image)) # 归一化
    blur_kernel = np.zeros((2*M_users-1,3))
    kernel_height = 2*M_users-1
    blur_kernel[0:M_users,1] = 1.
    blur_kernel[M_users:,0] = 1.
    blur_kernel_1d = np.array([1.]*M_users)
    blur_kernel_r = np.rot90(blur_kernel,2)*(1+0j)   # 注意，必须将kernel先旋转180°，才能使实际卷积函数按照“对应位置相乘”运行卷积
    # print(blur_kernel_r)
    d_data_col = np.hstack((image,np.zeros((M_users,1))))   # 额外加一列

    # 乘信道系数
    h_coff = h_coff.reshape(-1,1)
    # print(h_coff)   # debug
    hh_expand = np.tile(h_coff,(1,d_data_col.shape[1]))    # 每行是hi
    d_data_h = d_data_col*hh_expand
    # print(hh_expand[:,:10]) # debug
    # 加卷积
    convSame = signal.convolve2d(d_data_h, blur_kernel_r, mode='same')# 等效为一维线卷积，实现更快
    # 加噪
    convSame = awgn(convSame,SNR_db) # add white gaussian noise
    convSame = np.complex64(convSame)   # 32bit for real and imag
    return convSame
from numpy import fft
def wiener_deconv(input, kernel, SNR_db=0):       
    ''' 
    Winner filter with given blur kernel
    ---
    input: M长序列，可能包含周边padding
    kernel: 补全至M的卷积核
    '''
    K = 1 / (np.power(10, SNR_db / 10))
    input_fft = fft.fft(input)
    kernel_fft = fft.fft(kernel)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = fft.ifft(input_fft * kernel_fft_1)
    # result = np.abs(fft.fftshift(result))
    return result


## BEGIN LOOP TEST PRAT##
for db_iter in range(5):
    for avg_iter in range(3):
        h_coff = np.exp(1j*np.random.uniform(0,1,(M_users,1)) * 4* np.pi /4)
        SNR_db_global = SNR_all[db_iter]
        
        y_mat = distortion_func(d_data,h_coff,SNR_db_global)
        noised_seq = y_mat.T.reshape(-1) # M x L -> ML x 1 seq
        y_out = noised_seq[:-1]  # 得到过采样有噪失真序列

        # 1. aligned estimation

        MthFiltersIndex = (np.arange(L_symbols) + 1) * M_users - 1
        output = y_out[MthFiltersIndex]
        mse_val = np.mean(np.abs(output-np.sum(d_data,axis=0))**2)
        mse_record[db_iter,0,avg_iter] = mse_val

        # 2. ML estimation
        import scipy.sparse.linalg as srlg
        from scipy.sparse import csr_matrix
        D_array = np.zeros([M_users*(L_symbols+1)-1, M_users*L_symbols])*(1+0j)
        for idx in range(M_users*L_symbols):    # 按列依次赋值
            D_array[np.arange(M_users)+idx, idx] = h_coff[np.mod(idx,M_users)]
        # print(D_array.shape,y_out.shape)
        # restored = np.linalg.lstsq(D_array,y_out)[0]    # 直接求解：5000x5000~29s
        D_array_csr = csr_matrix(D_array)               # 稀疏矩阵求解：5000x5000~2.7s
        pn = get_pn(y_out.flatten(),SNR_db_global)
        noise_sigma = csr_matrix(np.diag([pn]*(M_users*(L_symbols+1)-1)))
        noise_sigma_inv = srlg.inv(noise_sigma)
        # d_inv_csr = srlg.inv(D_array_csr.conj().T@D_array_csr)
        # restored = d_inv_csr @ D_array_csr.conj().T @ y_out  
        d_inv_csr = srlg.inv(D_array_csr.conj().T@noise_sigma_inv@D_array_csr)
        restored = d_inv_csr @ D_array_csr.conj().T @noise_sigma_inv@ y_out  
        x_re_mat0 = restored.flatten()[:L_symbols*M_users].reshape(L_symbols,M_users).T
        x_re_mat0 = np.real(x_re_mat0)
        mse_val = np.mean(np.abs(np.sum(x_re_mat0,axis=0)-np.sum(d_data,axis=0))**2)
        mse_record[db_iter,1,avg_iter] = mse_val
        # MxL>5000时，运行缓慢

        # 3. LMMSE estimation

        # 计算每一行的均值
        mean_vector = np.mean(d_data, axis=1)
        mu_vec = csr_matrix(np.tile(mean_vector.flatten(),L_symbols))

        # 计算每一行的方差
        variance_vector = np.var(d_data, axis=1)
        d_vector = np.tile(variance_vector.flatten(),L_symbols)
        d_hat_mat = csr_matrix(np.diag(d_vector))


        pn = get_pn(y_out.flatten(),SNR_db_global)
        noise_sigma = csr_matrix(np.diag([1.]*(M_users*(L_symbols+1)-1)))
        tmp_inv_mat = srlg.inv(D_array_csr @ d_hat_mat @ D_array_csr.T.conj() + noise_sigma)
        a_mat = d_hat_mat @ D_array_csr.T.conj() @ tmp_inv_mat
        I_mat = np.diag([1.]*(M_users*L_symbols))
        # print(a_mat.shape)
        restored = a_mat @ y_out.reshape(-1,1) + (I_mat - a_mat @ D_array_csr) @ mu_vec.reshape(-1,1)
        # print(restored.shape)
        x_re_mat1 = restored.flatten()[:L_symbols*M_users].reshape(L_symbols,M_users).T
        x_re_mat1 = np.real(x_re_mat1)
        # print(x_re_mat1.shape,d_data.shape)
        sum_est = np.array(np.sum(x_re_mat1,axis=0))    # matrix to array
        sum_gt = np.sum(d_data,axis=0)

        # print(np.mean(np.abs(sum_est-sum_gt)**2))
        mse_val = np.mean(np.abs(sum_est-sum_gt)**2)
        mse_record[db_iter,2,avg_iter] = mse_val
        # 4. Our estimation

        output,image = inference(y_out.flatten(),'../model_Umod5.pth',M_users,L_symbols,h_coff,SNR_db_global)
        image = image.real
        # print('Our result:',np.mean(np.abs(np.sum(image,axis=0)-np.sum(d_data,axis=0))**2))
        mse_val = np.mean(np.abs(np.sum(image,axis=0)-np.sum(d_data,axis=0))**2)
        mse_record[db_iter,3,avg_iter] = mse_val

        # 5. Wiener itself

        y_out1 = y_out.flatten()
        pad_len = len(y_out1)
        blur_kernel_1d = np.array([1.]*M_users)
        blur_kernel_1d = np.hstack((blur_kernel_1d,np.array([0]*(pad_len-len(blur_kernel_1d)))))
        x_re_pad = wiener_deconv(y_out1,blur_kernel_1d,SNR_db_global)   # 低SNR下，用更低的SNR估计取得效果更好
        x_re_mat = x_re_pad[:L_symbols*M_users].reshape(L_symbols,M_users).T
        hh_expand = np.tile(h_coff,(1,L_symbols))    # 每行是hi
        x_re_mat = x_re_mat/hh_expand
        x_re_mat = np.real(x_re_mat)
        # print(hh_expand[:5,:5])
        # print(np.mean(np.abs(np.sum(x_re_mat,axis=0)-np.sum(d_data,axis=0))**2))
        mse_val = np.mean(np.abs(np.sum(x_re_mat,axis=0)-np.sum(d_data,axis=0))**2)
        mse_record[db_iter,4,avg_iter] = mse_val
    print('SNR=',SNR_db_global,mse_record[db_iter,:,0])

mse_record_mean = np.mean(mse_record,axis=2)
np.savetxt('MSE_result.csv',mse_record_mean,delimiter=",")