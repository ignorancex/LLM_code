# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp00-SECOND_ResNet_NoGraph_CatFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py', 
    '../_base_/default_runtime.py'
    ]

load_from = "work_dirs/0505_xnet_exp00-kitti_3d_3class-cyclic_40e/epoch_20.pth"
