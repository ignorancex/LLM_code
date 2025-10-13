import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

def visualize_feature_maps_mean(features,backbone_name = "default"):
    """
    可视化三个尺度的特征图
    """
    plt.figure(figsize=(24, 6))
    
    # # 处理高级特征 (1D)
    # high_level = features['high_level'].cpu().numpy()[0]

    # 处理高级特征图
    top_level = features['top_level'].cpu()
    # 对特征通道进行降维以便可视化
    top_level_pca = top_level.mean(dim=1)[0]

    # 处理高级特征图
    high_level = features['high_level'].cpu()
    # 对特征通道进行降维以便可视化
    high_level_pca = high_level.mean(dim=1)[0]

    # 处理中级特征图
    mid_level = features['mid_level'].cpu()
    # 对特征通道进行降维以便可视化
    mid_level_pca = mid_level.mean(dim=1)[0]
    
    # 处理低级特征图
    low_level = features['low_level'].cpu()
    # 对特征通道进行降维以便可视化
    low_level_pca = low_level.mean(dim=1)[0]

    # 绘制顶级特征图
    plt.subplot(1, 4, 1)
    plt.imshow(top_level_pca.numpy(), cmap='viridis')
    plt.title('Level3 Feature Map')
    plt.colorbar()

    # 绘制高级特征图
    plt.subplot(1, 4, 2)
    plt.imshow(high_level_pca.numpy(), cmap='viridis')
    plt.title('Level2 Feature Map')
    plt.colorbar()
    
    # 绘制中级特征图
    plt.subplot(1, 4, 3)
    plt.imshow(mid_level_pca.numpy(), cmap='viridis')
    plt.title('Level1 Feature Map')
    plt.colorbar()
    
    # 绘制低级特征图
    plt.subplot(1, 4, 4)
    plt.imshow(low_level_pca.numpy(), cmap='viridis')
    plt.title('Level0 Feature Map')
    plt.colorbar()

    plt.suptitle('Mean Feature Map Visualization', fontsize=16)

    plt.tight_layout()
    plt.savefig('../visualize/pic/{backbone_name}_multiscale_features_mean.png'.format(backbone_name=backbone_name))
    plt.show()

def visualize_feature_maps_tsne(features, backbone_name = "default"):
    """
    可视化三个尺度的特征图
    """
    plt.figure(figsize=(24, 6))

    # 处理顶级特征图
    top_level = features['top_level'].cpu().numpy()[0]
    top_level_flat = top_level.reshape(top_level.shape[0], -1).T  # (N, C) -> (C, N)
    tsne = TSNE(n_components=2, random_state=42)
    top_level_tsne = tsne.fit_transform(top_level_flat)

    # 处理高级特征图
    high_level = features['high_level'].cpu().numpy()[0]
    high_level_flat = high_level.reshape(high_level.shape[0], -1).T  # (N, C) -> (C, N)
    tsne = TSNE(n_components=2, random_state=42)
    high_level_tsne = tsne.fit_transform(high_level_flat)
    
    # 处理中级特征图
    mid_level = features['mid_level'].cpu().numpy()[0]
    mid_level_flat = mid_level.reshape(mid_level.shape[0], -1).T  # (N, C) -> (C, N)
    tsne = TSNE(n_components=2, random_state=42)
    mid_level_tsne = tsne.fit_transform(mid_level_flat)
    
    # 处理低级特征图
    low_level = features['low_level'].cpu().numpy()[0]
    low_level_flat = low_level.reshape(low_level.shape[0], -1).T  # (N, C) -> (C, N)
    tsne = TSNE(n_components=2, random_state=42)
    low_level_tsne = tsne.fit_transform(low_level_flat)

    # 绘制顶级特征图
    plt.subplot(1, 4, 1)
    plt.scatter(top_level_tsne[:, 0], top_level_tsne[:, 1], c='blue', s=1)
    plt.title('Level3 Feature Map')
    plt.colorbar()

    # 绘制高级特征图
    plt.subplot(1, 4, 2)
    plt.scatter(high_level_tsne[:, 0], high_level_tsne[:, 1], c='blue', s=1)
    plt.title('Level2 Feature Map')
    plt.colorbar()
    
    # 绘制中级特征图
    plt.subplot(1, 4, 3)
    plt.scatter(mid_level_tsne[:, 0], mid_level_tsne[:, 1], c='green', s=1)
    plt.title('Level1 Feature Map')
    plt.colorbar()
    
    # 绘制低级特征图
    plt.subplot(1, 4, 4)
    plt.scatter(low_level_tsne[:, 0], low_level_tsne[:, 1], c='red', s=1)
    plt.title('Level0 Feature Map')
    plt.colorbar()

    plt.suptitle('T-SNE Feature Map Visualization', fontsize=16)

    plt.tight_layout()
    plt.savefig('../visualize/pic/{backbone_name}_multiscale_features_tsne.png'.format(backbone_name=backbone_name))
    plt.show()

def visualize_feature_maps_pca(features,backbone_name = "default"):
    """
    可视化三个尺度的特征图
    """
    plt.figure(figsize=(24, 6))

    # 处理顶级特征图
    top_level = features['top_level'].cpu().numpy()[0]
    top_level_flat = top_level.reshape(top_level.shape[0], -1)
    pca = PCA(n_components=1)
    top_level_pca = pca.fit_transform(top_level_flat.T).flatten()

    # 处理高级特征图
    high_level = features['high_level'].cpu().numpy()[0]
    high_level_flat = high_level.reshape(high_level.shape[0], -1)
    pca = PCA(n_components=1)
    high_level_pca = pca.fit_transform(high_level_flat.T).flatten()
    
    # 处理中级特征图
    mid_level = features['mid_level'].cpu().numpy()[0]
    mid_level_flat = mid_level.reshape(mid_level.shape[0], -1)
    pca = PCA(n_components=1)
    mid_level_pca = pca.fit_transform(mid_level_flat.T).flatten()
    
    # 处理低级特征图
    low_level = features['low_level'].cpu().numpy()[0]
    low_level_flat = low_level.reshape(low_level.shape[0], -1)
    pca = PCA(n_components=1)
    low_level_pca = pca.fit_transform(low_level_flat.T).flatten()

    # 绘制顶级特征图
    plt.subplot(1, 4, 1)
    plt.imshow(top_level_pca.reshape(int(np.sqrt(len(top_level_pca))), -1), cmap='viridis')
    plt.title('Level3 Feature Map')
    plt.colorbar()

    # 绘制高级特征图
    plt.subplot(1, 4, 2)
    plt.imshow(high_level_pca.reshape(int(np.sqrt(len(high_level_pca))), -1), cmap='viridis')
    plt.title('Level2 Feature Map')
    plt.colorbar()
    
    # 绘制中级特征图
    plt.subplot(1, 4, 3)
    plt.imshow(mid_level_pca.reshape(int(np.sqrt(len(mid_level_pca))), -1), cmap='viridis')
    plt.title('Level1 Feature Map')
    plt.colorbar()
    
    # 绘制低级特征图
    plt.subplot(1, 4, 4)
    plt.imshow(low_level_pca.reshape(int(np.sqrt(len(low_level_pca))), -1), cmap='viridis')
    plt.title('Level0 Feature Map')
    plt.colorbar()

    plt.suptitle('PCA Feature Map Visualization', fontsize=16)

    plt.tight_layout()
    plt.savefig('../visualize/pic/{backbone_name}_multiscale_features_pca.png'.format(backbone_name=backbone_name))
    plt.show()