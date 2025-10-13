import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from matplotlib import rcParams
from torchsummary import summary
# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
class sam2hiera(nn.Module):
    def __init__(self, config_file=None, ckpt_path=None) -> None:
        super().__init__()
        if config_file is None:
            print("No config file provided, using default config")
            config_file = "./sam2_configs/sam2.1_hiera_l.yaml"
        if ckpt_path is None:
            model = build_sam2(config_file)
        else:
            model = build_sam2(config_file, ckpt_path)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.sam_encoder = model.image_encoder.trunk

        for param in self.sam_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        out = self.sam_encoder(x)
        return out
    
if __name__ == "__main__":
    config_file = r"../sam2_configs/sam2.1_hiera_l.yaml"
    ckpt_path = r"../checkpoints/sam2.1_hiera_large.pt"

    x = torch.randn(1, 3, 352, 352).cuda()
    with torch.no_grad():
        model = sam2hiera(config_file, ckpt_path).cuda()
        # x = torch.randn(1, 3, 352, 352).cuda()
        out= model(x)
        print(len(out))
        print(out[0].shape)
        print(out[1].shape)
        print(out[2].shape)
        print(out[3].shape)
        # print(model)
        # summary(model,(3, 352, 352))
        # for key in out.keys():
        #     print(key, type(out[key]))
        # print(out["vision_features"].shape)
        # # out["vision_pos_enc"] 是一个包含三个张量的列表
        # if isinstance(out["vision_pos_enc"], list) and len(out["vision_pos_enc"]) == 4:
        #     print("out['vision_pos_enc'] is a list of length 4")
        #     for i, tensor in enumerate(out["vision_pos_enc"]):
        #         print(f"Tensor {i} shape: {tensor.shape}")
        # else:
        #     print("out['tensor_list'] is not a list of length 4")
        # # out["vision_pos_enc"] 是一个包含三个张量的列表
        # if isinstance(out["backbone_fpn"], list) and len(out["backbone_fpn"]) == 4:
        #     print("out['backbone_fpn'] is a list of length 4")
        #     for i, tensor in enumerate(out["backbone_fpn"]):
        #         print(f"Tensor {i} shape: {tensor.shape}")
        # else:
        #     print("out['tensor_list'] is not a list of length 4")
        #############################################输出结果##################################################

        # vision_features <class 'torch.Tensor'>
        # vision_pos_enc <class 'list'>
        # backbone_fpn <class 'list'>
        # torch.Size([1, 256, 22, 22])
        # out['vision_pos_enc'] is a list of length 3
        # Tensor 0 shape: torch.Size([1, 256, 88, 88])
        # Tensor 1 shape: torch.Size([1, 256, 44, 44])
        # Tensor 2 shape: torch.Size([1, 256, 22, 22])
        # out['backbone_fpn'] is a list of length 3
        # Tensor 0 shape: torch.Size([1, 256, 88, 88])
        # Tensor 1 shape: torch.Size([1, 256, 44, 44])
        # Tensor 2 shape: torch.Size([1, 256, 22, 22])

        ###############################################END#####################################################
        # if torch.equal(out['backbone_fpn'][-1], out['vision_features']):
        #     print("vision_features is the last element of backbone_fpn")

