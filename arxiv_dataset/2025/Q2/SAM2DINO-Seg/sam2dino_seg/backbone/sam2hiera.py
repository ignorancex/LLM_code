import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from matplotlib import rcParams
from sam2dino_seg.self_transforms.preprocess_image import transforms_image
from sam2dino_seg.modules import adapter
from visualize.features_vis import visualize_feature_maps_mean, visualize_feature_maps_pca, visualize_feature_maps_tsne


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
        # Adapter
        blocks = []
        for block in self.sam_encoder.blocks:
            blocks.append(
                adapter.Adapter(block)
            )
        self.sam_encoder.blocks = nn.Sequential(
            *blocks
        )
    def forward(self, x):
        out = self.sam_encoder(x)
        return out
    
if __name__ == "__main__":
    config_file = r"G:\MyProjectCode\SAM2DINO-Seg\sam2_configs\sam2.1_hiera_l.yaml"
    ckpt_path = r"G:\MyProjectCode\SAM2DINO-Seg\checkpoints\sam2.1_hiera_large.pt"
    # 预处理图像
    image_path = r"G:\MyProjectCode\SAM2DINO-Seg\data\images\COD10K-CAM-1-Aquatic-3-Crab-29.jpg"  # 替换为您的图像路径
    x = transforms_image(image_path, image_size=352)
    with torch.no_grad():
        model = sam2hiera(config_file, ckpt_path).cuda()
        if torch.cuda.is_available():
            x = x.cuda()
        out= model(x)
        # 组合为字典
        # features = {
        #     'high_level': out['backbone_fpn'][2],
        #     'mid_level': out['backbone_fpn'][1],
        #     'low_level': out['backbone_fpn'][0]
        # }
        features = {
            'top_level': out[3],
            'high_level': out[2],
            'mid_level': out[1],
            'low_level': out[0]
        }

        # 打印各特征形状
        print(f"顶级特征形状 (全局尺度): {features['top_level'].shape}")
        # print(f"高级特征形状 (高等尺度): {features['high_level']}")
        print(f"高级特征形状 (高等尺度): {features['high_level'].shape}")
        print(f"中级特征形状 (中等尺度): {features['mid_level'].shape}")
        print(f"低级特征形状 (局部尺度): {features['low_level'].shape}")

        # 均值可视化特征
        visualize_feature_maps_mean(features,backbone_name='SAM2')

        # PCA可视化
        visualize_feature_maps_pca(features,backbone_name='SAM2')

        # T-SNE可视化
        visualize_feature_maps_tsne(features, backbone_name='SAM2')

        print("Hiera多尺度特征提取完成!")