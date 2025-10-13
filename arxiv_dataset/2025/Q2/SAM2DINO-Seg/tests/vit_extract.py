import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor as HFViTFeatureExtractor
# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

class ViTCustomFeatureExtractor(nn.Module):
    def __init__(self, model_name='google/vit-large-patch16-224'):
        """
        初始化ViT特征提取器
        Args:
            model_name: 预训练ViT模型名称
        """
        super(ViTCustomFeatureExtractor, self).__init__()
        # 加载预训练ViT模型
        self.model = ViTModel.from_pretrained(model_name, output_hidden_states=True)
        self.feature_extractor = HFViTFeatureExtractor.from_pretrained(model_name)

    def forward(self, x):
        """
        前向传播，提取三个级别的特征
        Args:
            x: 输入图像张量
        Returns:
            low_level_features: 低级特征 (较浅层特征)
            mid_level_features: 中级特征
            high_level_features: 高级特征 (最深层特征)
        """
        # 获取所有隐藏状态
        outputs = self.model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # 选择三个不同层次的特征
        # 假设hidden_states包含输入嵌入 + 12个transformer层的隐藏状态
        # 我们选择第1层作为低级特征，第6层作为中级特征，第12层作为高级特征
        low_level_features = hidden_states[1]  # 第1层
        mid_level_features = hidden_states[6]  # 第6层
        high_level_features = hidden_states[12]  # 第12层 (最后一层)

        return low_level_features, mid_level_features, high_level_features


def preprocess_image(image_path, feature_extractor):
    """
    预处理图像以适应ViT模型
    Args:
        image_path: 图像路径
        feature_extractor: 特征提取器
    Returns:
        preprocessed_img: 预处理后的图像张量
    """
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs.pixel_values


def visualize_features(features, level_name):
    """
    可视化特征图
    Args:
        features: 特征张量 [1, n_patches, hidden_dim]
        level_name: 特征级别名称
    """
    # 将特征重塑成图像格式
    # 对于ViT-base，图像被分成14x14个patch，每个patch的特征向量维度为768
    batch_size, n_patches, hidden_dim = features.shape

    # 移除CLS token，只保留patch tokens
    patch_features = features[:, 1:, :]

    # 重塑为方形网格 (不包括CLS token，所以是从1开始)
    patch_size = int(np.sqrt(n_patches - 1))

    # 使用PCA降维，将高维特征降到3维以便可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    reshaped_features = patch_features.reshape(-1, hidden_dim).detach().numpy()
    pca_features = pca.fit_transform(reshaped_features)

    # 归一化到[0,1]范围
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    # pca_features = np.clip(pca_features, 0, 1)
    # 重塑为图像格式
    pca_image = pca_features.reshape(patch_size, patch_size, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(pca_image, cmap='viridis')
    plt.title(f'{level_name} 特征图')
    plt.axis('off')
    plt.colorbar()
    plt.savefig('../visualize/pic/Vit_multiscale_features_pca_{level_name}.png'.format(level_name=level_name))
    plt.show()


def main():
    # 初始化ViT特征提取器
    vit_extractor = ViTCustomFeatureExtractor(model_name='google/vit-large-patch16-224')

    # 设置图像路径
    image_path = r"G:\MyProjectCode\SAM2DINO-Seg\data\images\R-C.jpg"  # 替换为您自己的图像路径

    # 预处理图像
    preprocessed_img = preprocess_image(image_path, vit_extractor.feature_extractor)

    # 提取三个级别的特征
    with torch.no_grad():
        low_features, mid_features, high_features = vit_extractor(preprocessed_img)

    # 可视化特征
    visualize_features(low_features, '低级')
    visualize_features(mid_features, '中级')
    visualize_features(high_features, '高级')

    # 保存特征张量
    torch.save({
        'low_level': low_features,
        'mid_level': mid_features,
        'high_level': high_features
    }, 'vit_features.pt')

    print(f"低级特征形状: {low_features.shape}")
    print(f"中级特征形状: {mid_features.shape}")
    print(f"高级特征形状: {high_features.shape}")
    # print("特征已提取并保存至 'vit_features.pt'")


if __name__ == "__main__":
    main()