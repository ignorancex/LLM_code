import os
# if train_or_eval = True then 训练 else 测试
train_or_eval = False
#train_or_eval = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 注意修改train文件里的部分代码以适应并行
print('GPU ID:', os.environ['CUDA_VISIBLE_DEVICES'])

if train_or_eval is not True:
    # 测试的配置
    print('test......')
    task = 'denoising'
    # dataset_dir = r'D:\Users\mxy\Data\J4R_old\frames_light_test_JPEG'  # 测试图片包括边缘图的路径
    # dataset_gtc_dir = r'D:\Users\mxy\Data\J4R_old\frames_light_test_JPEG'
    # # # 相对应的gtc路径(使用了训练集中的gtc)，所有设置的路径都是00006/1这种文件夹的父文件夹才可以
    # pathlistfile = r'D:\Users\mxy\Data\J4R_old\test_light.txt'  # 测试的图片的具体路径
    # dataset_dir = '/data8T/mxy/data/frames_heavy_test_JPEG/'
    # dataset_gtc_dir = '/data8T/mxy/data/frames_heavy_test_JPEG'
    # pathlistfile = '/data8T/mxy/data/test_heavy.txt'
    dataset_dir = '/root/huancun/frames_heavy_test_JPEG/'
    dataset_gtc_dir = '/root/huancun/frames_heavy_test_JPEG/'
    pathlistfile = '/root/huancun/test_heavy.txt'
    dataset_dir = r'E:\G\Data\J4R_old\frames_heavy_test_JPEG'
    dataset_gtc_dir = r'E:\G\Data\J4R_old\frames_heavy_test_JPEG'
    pathlistfile = r'E:\G\Data\J4R_old\test_heavy.txt'
    # dataset_dir = './train/visual/L'
    # dataset_gtc_dir = './train/visual/L'
    # pathlistfile = './train/test_visual.txt'
    # dataset_dir = '/data/mxy/data/Dataset_Testing_Synthetic/'
    # dataset_gtc_dir = '/data/mxy/data/Dataset_Testing_Synthetic/GTC/'
    # pathlistfile = '/data/mxy/data/test_spac.txt'
    out_img_dir = './evaluate/epoch100'  # 实验结果存放位置

    model_path = './toflow_models_mine/denoising100.pkl'  # 1maxoper的新模型

    gpuID = 0  # 不起作用
    map_location = 'cuda:0'
    BATCH_SIZE = 1
    h = 496  # 500 报错 496?
    w = 888  # 8 89 报错
    N = 3  # 5张图片
    C = 64
else:
    # 训练的配置
    print('train......')
    mode = 'train'
    task = 'denoising'
    # edited_img_dir = '/data/mxy/data/Dataset_Training_Synthetic/training/'  # 训练输入的图片的文件夹
    # dataset_dir = '/data/mxy/data/Dataset_Training_Synthetic/training/'
    # rfc_dir = '/root/data/mengxiangyu/data/heavy_train'
    # gtc_dir = '/root/data/mengxiangyu/data/heavy_train'
    # pathlistfile = '/root/data/mengxiangyu/data/train_heavy.txt'
    # val_rfc_dir = '/root/data/mengxiangyu/data/frames_heavy_test_JPEG'
    # val_gtc_dir = '/root/data/mengxiangyu/data/frames_heavy_test_JPEG'
    # val_pathlistfile = './train/test_heavy.txt'
    rfc_dir = '/root/huancun/heavy_train'
    gtc_dir = '/root/huancun/heavy_train'
    pathlistfile = '/root/huancun/train_heavy.txt'
    val_rfc_dir = '/root/huancun/frames_heavy_test_JPEG/'
    val_gtc_dir = '/root/huancun/frames_heavy_test_JPEG/'
    val_pathlistfile = './train/test_heavy.txt'
    # rfc_dir = '/data8T/mxy/data/heavy_train'
    # gtc_dir = '/data8T/mxy/data/heavy_train'
    # pathlistfile = '/data8T/mxy/data/train_heavy.txt'
    # val_rfc_dir = '/data8T/mxy/data/frames_heavy_test_JPEG/'
    # val_gtc_dir = '/data8T/mxy/data/frames_heavy_test_JPEG/'
    # val_pathlistfile = './train/test_heavy.txt'
    # edited_img_dir = '/home/mengxiangyu/data/TPAMI/Video_rain_synthesis_train/Rain'
    # dataset_dir = '/home/mengxiangyu/data/TPAMI/Video_rain_synthesis_train/GT'
    # pathlistfile = '/home/mengxiangyu/data/TPAMI/train_pami.txt'
    # pathlistfile = '/data/mxy/data/Dataset_Training_Synthetic/training/train_spac.txt'
    # edited_img_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'  # 训练输入的图片的文件夹
    # dataset_dir = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\rained'
    # pathlistfile = r'D:\Users\mxy\Data\Ours\vimeo_test_clean\sep_testlist_train.txt'

    visualize_root = './visualization_mine/'  # 存放展示结果的文件夹
    visualize_pathlist = ['2500000']  # 需要展示训练结果的训练图片所在的小文件夹
    checkpoints_root = './checkpoints_mine'  # 训练过程中产生的检查点的存放位置
    model_besaved_root = 'toflow_models_mine'  # best_model 和 final_model 的参数的保存位置
    model_best_name = '_best.pkl'
    model_val_best_name = '_val_best.pkl'
    VAL_AFTER_EVERY = 20000     # evaluate frequency, larger num denotes no eval
    model_final_name = '_final.pkl'

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 注意修改train文件里的部分代码以适应并行
    # print('GPU ID:', os.environ['CUDA_VISIBLE_DEVICES'])
    gpuID = 1   # 不起作用
    # print('gpuID: ', gpuID)

    # Hyper Parameters
    if task == 'interp':
        LR = 3 * 1e-5
    elif task in ['denoise', 'denoising', 'sr', 'super-resolution']:
        LR = 1 * 1e-4
        # LR = 2e-4   # 2e-4
        # LR_MIN = 1e-6   #
    C = 64
    EPOCH = 201
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 1
    num_workers = 1
    LR_strategy = []
    val_h = 480  # evaluation height
    val_w = 640  #
    h = 320      # not use
    w = 320      # not use
    N = 3  # 一次使用3张图片
    sample_frames = 3  # dataset对一个视频采样的帧数
    scale_min = 1  # 0.4     set 1 for no resizing
    scale_max = 1  # 2
    crop_size = 256  # 起作用  裁剪大小，raft 是4的倍数
    size_multiplier = 2 ** 6  # ?
    geometry_aug = True
    order_aug = True

    w_SSIM = 11.0
    w_L1 = 7.5
    w_VGG = 1.0
    w_Style = 0.0
    w_ST = 1.0
    w_LT = 0.0
    alpha = 50.0

    use_checkpoint = False  # 一开始不使用已有的检查点
    checkpoint_exited_path = './checkpoints_mine/checkpoints_100epoch.ckpt'  # 已有的检查点
    work_place = '.'
    model_name = task
    model_houzhui = '.pkl'
    Training_pic_path = 'toflow_models_mine/Training_result_mine_maxoper.jpg'
    val_pic_path = 'toflow_models_mine/val.jpg'
    model_information_txt = model_name + '_information.txt'
    val_info_txt = model_name + '_val_info.txt'

# # if __name__ == '__main__':
# #     print('#### Test Case ###')
# #     import torch
# #
# #     with torch.cuda.device(3):
# #         x = torch.rand(1, 1500, 3, 480, 512).cuda()
# #         str = input("请输入0：")
# #         print("你输入的内容是: ", str)
#
# import numpy as np
# # from thop import profile
# from torchstat import stat
# # https://blog.csdn.net/weixin_43519707/article/details/108512921
# import torch
# # from models.modules.Sakuya_arch import LunaTokis
# import Network
#
#
# def count_parameters_in_MB(model):
#     return np.sum(np.fromiter((np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name),
#                               np.float)) / 1e6
#
#
# if __name__ == '__main__':
#     with torch.cuda.device(2):
#         x = torch.rand(1, 3, 3, 256, 256).cuda()
#         # y = torch.rand(1, 3, 256, 256).cuda()
#         net = Network.TOFlow(h=1, w=1, task='.', cuda_flag=True).cuda()
#         net2 = Network.TOFlow(h=1, w=1, task='.', cuda_flag=True).cuda()
#         print('parm(MB): ', count_parameters_in_MB(net))  # parm(MB):
#
#         print(isinstance(net2, torch.nn.Module))
#         # https: // blog.csdn.net / weixin_43519707 / article / details / 108512921
#         stat(net2, (6, 224, 224))   # compute FLOPs abd Param input CxHxW, change it to GPU version
#
#         out, _, _ = net(x)
#         print('out: ', out.shape)
#         str = input("请输入：")
#         print("你输入的内容是: ", str)
# #         # parm(MB) of feat_extract: 0.36928
# #         # parm(MB) of pcd_align: 2.224016
# #         # parm(MB) of reconstruct: 1.7232
