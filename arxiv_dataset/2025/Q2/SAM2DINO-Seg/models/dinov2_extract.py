import torch
import numpy as np
from matplotlib import rcParams
from sam2dino_seg.self_transforms.preprocess_image import transforms_image
from visualize.features_vis import visualize_feature_maps_mean,visualize_feature_maps_pca,visualize_feature_maps_tsne

# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 导入DINOV2模型
def load_dinov2_model(model_name="dinov2_vitl14"):
    """
    加载预训练的DINOV2模型
    可选模型: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    """
    model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# 提取多尺度特征
def extract_multiscale_features(model, image, patch_size=14):
    """
    提取三个不同尺度的特征图:
    1. 全局特征: 最终的CLS token特征
    2. 中等尺度特征: 中间层的特征图
    3. 局部特征: 最底层的特征图
    """
    if torch.cuda.is_available():
        image = image.cuda()
    
    # 注册钩子来获取中间层特征
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook
    
    # 检查blocks数量
    num_blocks = len(model.blocks)
    print(f"模型中的blocks数量: {num_blocks}")
    # 注册钩子到不同的块以获取多尺度特征
    # 针对不同尺度的特征，我们选择不同深度的块
    model.blocks[1].register_forward_hook(get_features('low_level'))
    model.blocks[7].register_forward_hook(get_features('mid_level'))
    model.blocks[17].register_forward_hook(get_features('high_level'))
    model.blocks[23].register_forward_hook(get_features('top_level'))
    # 前向传播获取特征
    with torch.no_grad():
        output = model.forward_features(image)
    print(output.keys())
    print(output['x_norm_clstoken'].shape)
    print(output['x_norm_regtokens'].shape)
    print(output['x_norm_patchtokens'].shape)
    print(output['x_norm_patchtokens'])
    print(output['x_prenorm'].shape)
    print(output['masks'])
    print(features.keys())
    # 提取高等级别特征并重新构建为特征图
    top_level_feature = features['top_level'][:, 1:, :]  # 移除CLS token

    # 提取高等级别特征并重新构建为特征图
    high_level_feature = features['high_level'][:, 1:, :]  # 移除CLS token

    # 提取中等级别特征并重新构建为特征图
    mid_level_feature = features['mid_level'][:, 1:, :]  # 移除CLS token
    
    # 提取低级别特征并重新构建为特征图
    low_level_feature = features['low_level'][:, 1:, :]  # 移除CLS token
    
    # 转换为空间特征图
    img_size = int(image.shape[-1])
    feature_size = int((img_size / patch_size) ** 2)
    
    # 验证获取的特征大小
    assert mid_level_feature.shape[1] == feature_size, f"特征大小不匹配: {mid_level_feature.shape[1]} vs {feature_size}"
    
    # 重新构建为2D特征图
    side_length = int(np.sqrt(feature_size))

    top_level_map = top_level_feature.reshape(1, side_length, side_length, -1).permute(0, 3, 1, 2)
    high_level_map = high_level_feature.reshape(1, side_length, side_length, -1).permute(0, 3, 1, 2)
    mid_level_map = mid_level_feature.reshape(1, side_length, side_length, -1).permute(0, 3, 1, 2)
    low_level_map = low_level_feature.reshape(1, side_length, side_length, -1).permute(0, 3, 1, 2)
    
    return {
        'top_level': top_level_map,                   # 全局特征图
        'high_level': high_level_map,                 # 高等尺度特征图
        'mid_level': mid_level_map,                   # 中等尺度特征图
        'low_level': low_level_map                    # 局部尺度特征图
    }

# 示例使用
if __name__ == "__main__":
    # 加载模型 - 使用DINOV2的Large模型
    model = load_dinov2_model("dinov2_vitl14")
    
    # 预处理图像
    image_path = r"G:\MyProjectCode\SAM2DINO-Seg\data\images\COD10K-CAM-1-Aquatic-3-Crab-29.jpg"  # 替换为您的图像路径
    image = transforms_image(image_path)

    # 提取特征
    features = extract_multiscale_features(model, image)
    
    # 打印各特征形状
    print(f"顶级特征形状 (全局尺度): {features['top_level'].shape}")
    print(f"顶级特征形状 (全局尺度): {features['top_level']}")
    print(f"高级特征形状 (高等尺度): {features['high_level'].shape}")
    print(f"中级特征形状 (中等尺度): {features['mid_level'].shape}")
    print(f"低级特征形状 (局部尺度): {features['low_level'].shape}")
    
    # 可视化特征
    visualize_feature_maps_mean(features,backbone_name="DINOv2")

    # PCA可视化
    visualize_feature_maps_pca(features,backbone_name="DINOv2")

    # T-SNE可视化
    visualize_feature_maps_tsne(features,backbone_name="DINOv2")

    print("DINOv2多尺度特征提取完成!")