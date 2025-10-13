import torch
import torch.nn as nn
import numpy as np
from matplotlib import rcParams
from sam2dino_seg.self_transforms.preprocess_image import transforms_image

# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name=None, hub_dir=None) -> None:
        super().__init__()
        if hub_dir is None:
            print("No hub_dir specified, using default")
            hub_dir = 'facebookresearch/dinov2'
        if model_name is None:
            print("No model_name specified, using default")
            model_name = 'dinov2_vitl14'
        model = torch.hub.load(hub_dir, model_name, pretrained=True)
        self.dino_encoder = model
        self.patchsize = 14

        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.dino_encoder.forward_features(x)
        dino_feature = output['x_norm_patchtokens']
        # print(dino_feature.shape)
        # 转换为空间特征图
        img_size = int(x.shape[-1])
        batch_size = int(x.shape[0])
        feature_size = int((img_size / self.patchsize) ** 2)
        # 验证获取的特征大小
        assert dino_feature.shape[1] == feature_size, f"特征大小不匹配: {dino_feature.shape[1]} vs {feature_size}"
        # 重新构建为2D特征图
        side_length = int(np.sqrt(feature_size))
        dino_feature_map = dino_feature.reshape(batch_size, side_length, side_length, -1).permute(0, 3, 1, 2)

        return dino_feature_map
# 示例使用
if __name__ == "__main__":
    # 预处理图像
    # image_path = r"G:\MyProjectCode\SAM2DINO-Seg\data\images\R-C.jpg"  # 替换为您的图像路径
    # x = transforms_image(image_path, image_size=518)
    x = torch.randn(12, 3, 518, 518)
    with torch.no_grad():
        model = DinoV2FeatureExtractor().cuda()
        if torch.cuda.is_available():
            x = x.cuda()
        out = model(x)
        print(out.shape)
        # print(out)