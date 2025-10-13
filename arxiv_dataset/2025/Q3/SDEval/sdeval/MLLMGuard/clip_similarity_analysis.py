import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import CLIPImageProcessor, CLIPModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPFeatureExtractor:
    def __init__(self, model_id="openai/clip-vit-large-patch14", cache_dir=None):
        """初始化CLIP模型和处理器"""
        print(f"加载CLIP模型: {model_id}")
        self.model = CLIPModel.from_pretrained('/model/clip', cache_dir=cache_dir).to(device)
        self.processor = CLIPImageProcessor.from_pretrained('/model/clip', cache_dir=cache_dir)
        print(f"模型加载完成，运行在 {device} 上")
    
    def process_image(self, image_path):
        """处理单张图像，返回预处理后的张量"""
        from PIL import Image as PImage
        try:
            if image_path.startswith("http"):
                # 从URL加载图像
                
                import requests
                image = PImage.open(requests.get(image_path, stream=True).raw).convert("RGB")
            else:
                # 从本地加载图像
                image = PImage.open(image_path).convert("RGB")
            
            # 预处理图像
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(device)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_feature(self, image_tensor):
        """从图像张量中提取CLIP特征"""
        with torch.no_grad():
            features = self.model.get_image_features(image_tensor)
            # 归一化特征
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features
    
    def batch_extract_features(self, image_paths, batch_size=32):
        """批量提取图像特征"""
        features = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                tensor = self.process_image(path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
            
            if batch_tensors:
                batch_tensor = torch.cat(batch_tensors)
                batch_features = self.extract_feature(batch_tensor)
                features.append(batch_features)
        
        if features:
            return torch.cat(features), valid_paths
        else:
            return None, []

class SimilarityAnalyzer:
    def __init__(self):
        """相似度分析器"""
        pass
    
    def compute_similarity(self, features_a, features_b=None):
        """
        计算特征之间的余弦相似度
        
        如果只提供features_a，则计算内部相似度矩阵
        如果同时提供features_a和features_b，则计算它们之间的相似度矩阵
        """
        if features_b is None:
            # 计算内部相似度
            return torch.matmul(features_a, features_a.transpose(0, 1)).cpu().numpy()
        else:
            # 计算两个数据集之间的相似度
            return torch.matmul(features_a, features_b.transpose(0, 1)).cpu().numpy()
    
    def analyze_similarity(self, similarity_matrix, threshold=0.5):
        """分析相似度矩阵，统计超过阈值的比例"""
        if similarity_matrix.ndim == 2:
            # 对于两个数据集之间的相似度
            above_threshold = (similarity_matrix > threshold).mean()
            max_similarities = np.max(similarity_matrix, axis=1)
            avg_max_similarity = max_similarities.mean()
            return {
                "average_similarity": similarity_matrix.mean(),
                "proportion_above_threshold": above_threshold,
                "average_max_similarity": avg_max_similarity,
                "similarity_matrix": similarity_matrix
            }
        else:
            # 对于单个数据集内部的相似度（排除对角线）
            np.fill_diagonal(similarity_matrix, -1)  # 排除自身比较
            above_threshold = (similarity_matrix > threshold).mean()
            max_similarities = np.max(similarity_matrix, axis=1)
            avg_max_similarity = max_similarities.mean()
            return {
                "average_similarity": similarity_matrix[similarity_matrix != -1].mean(),
                "proportion_above_threshold": above_threshold,
                "average_max_similarity": avg_max_similarity,
                "similarity_matrix": similarity_matrix
            }
    
    def visualize_similarity(self, similarity_matrix, image_paths=None, top_k=5, output_dir="results"):
        """可视化相似度结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        if similarity_matrix.ndim == 2 and similarity_matrix.shape[0] == similarity_matrix.shape[1]:
            # 内部相似度可视化
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='余弦相似度')
            plt.title('数据集内部相似度矩阵')
            plt.savefig(os.path.join(output_dir, 'internal_similarity_matrix.png'))
            
            # 绘制相似度分布
            sim_values = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
            plt.figure(figsize=(10, 6))
            plt.hist(sim_values, bins=50, alpha=0.7)
            plt.axvline(x=0.5, color='r', linestyle='--', label='阈值=0.5')
            plt.xlabel('余弦相似度')
            plt.ylabel('频次')
            plt.title('数据集内部相似度分布')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'internal_similarity_distribution.png'))
            
            # 如果提供了图像路径，找出最相似的样本对
            if image_paths is not None:
                with open(os.path.join(output_dir, 'most_similar_pairs.txt'), 'w') as f:
                    f.write("最相似的样本对:\n")
                    indices = np.triu_indices(similarity_matrix.shape[0], k=1)
                    flat_indices = np.ravel_multi_index(indices, similarity_matrix.shape)
                    sorted_indices = np.argsort(-similarity_matrix[indices])[:top_k]
                    
                    for i in sorted_indices:
                        idx1, idx2 = indices[0][i], indices[1][i]
                        sim = similarity_matrix[idx1, idx2]
                        f.write(f"相似度: {sim:.4f}\n")
                        f.write(f"  Image 1: {image_paths[idx1]}\n")
                        f.write(f"  Image 2: {image_paths[idx2]}\n\n")
        else:
            # 两个数据集之间的相似度可视化
            plt.figure(figsize=(12, 10))
            plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='余弦相似度')
            plt.xlabel('数据集B样本')
            plt.ylabel('数据集A样本')
            plt.title('两个数据集之间的相似度矩阵')
            plt.savefig(os.path.join(output_dir, 'cross_similarity_matrix.png'))
            
            # 绘制相似度分布
            plt.figure(figsize=(10, 6))
            plt.hist(similarity_matrix.flatten(), bins=50, alpha=0.7)
            plt.axvline(x=0.5, color='r', linestyle='--', label='阈值=0.5')
            plt.xlabel('余弦相似度')
            plt.ylabel('频次')
            plt.title('数据集间相似度分布')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'cross_similarity_distribution.png'))
    
    def visualize_tsne(self, features, labels=None, output_dir="results"):
        """使用t-SNE可视化特征空间"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 降维
        features_np = features.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features_np)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                plt.scatter(
                    reduced_features[labels == label, 0], 
                    reduced_features[labels == label, 1], 
                    c=[color], 
                    label=str(label),
                    alpha=0.7,
                    s=50
                )
            plt.legend(title='数据集')
        else:
            plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7, s=50)
        
        plt.title('CLIP特征的t-SNE可视化')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))

def load_image_paths_from_dir(data_dir):
    """从目录加载图像路径"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_paths = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def load_image_paths_from_json(json_path):
    """从JSON文件加载图像路径"""
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='CLIP特征提取与相似度分析工具')
    parser.add_argument('--data_dir', type=str, default='/Datasets/add_texts', help='数据集目录')
    parser.add_argument('--json_path', type=str, help='包含图像路径的JSON文件')
    parser.add_argument('--data_dir_b', type=str,default='/Datasets/MM-SafetyBench(imgs)', help='第二个数据集目录（用于跨数据集比较）')
    parser.add_argument('--json_path_b', type=str, help='第二个数据集的JSON文件')
    parser.add_argument('--model_id', type=str, default="openai/clip-vit-large-patch14", 
                        help='CLIP模型ID')
    parser.add_argument('--cache_dir', type=str, help='模型缓存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--threshold', type=float, default=0.95, help='相似度阈值')
    parser.add_argument('--output_dir', type=str, default="/root/code/MLLMGuard/mmguarl_sim", 
                        help='结果输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化特征提取器
    extractor = CLIPFeatureExtractor(args.model_id, args.cache_dir)
    
    # 加载数据集A的图像路径
    if args.data_dir:
        image_paths_a = load_image_paths_from_dir(args.data_dir)
    elif args.json_path:
        image_paths_a = load_image_paths_from_json(args.json_path)
    else:
        raise ValueError("必须提供 --data_dir 或 --json_path 参数")
    
    print(f"数据集A包含 {len(image_paths_a)} 张图像")
    
    # 提取数据集A的特征
    features_a, valid_paths_a = extractor.batch_extract_features(image_paths_a, args.batch_size)
    
    if features_a is None:
        print("未能提取任何有效特征，程序退出")
        return
    
    # 保存特征
    features_dict_a = {path: feat for path, feat in zip(valid_paths_a, features_a)}
    torch.save(features_dict_a, os.path.join(args.output_dir, 'features_dataset_a.pt'))
    
    # 初始化相似度分析器
    analyzer = SimilarityAnalyzer()
    
    if args.data_dir_b or args.json_path_b:
        # 跨数据集比较
        print("\n处理第二个数据集...")
        
        # 加载数据集B的图像路径
        if args.data_dir_b:
            image_paths_b = load_image_paths_from_dir(args.data_dir_b)
        else:
            image_paths_b = load_image_paths_from_json(args.json_path_b)
        
        print(f"数据集B包含 {len(image_paths_b)} 张图像")
        
        # 提取数据集B的特征
        features_b, valid_paths_b = extractor.batch_extract_features(image_paths_b, args.batch_size)
        
        if features_b is None:
            print("未能提取数据集B的有效特征，仅分析数据集A")
        else:
            # 保存特征
            features_dict_b = {path: feat for path, feat in zip(valid_paths_b, features_b)}
            torch.save(features_dict_b, os.path.join(args.output_dir, 'features_dataset_b.pt'))
            
            # 计算跨数据集相似度
            print("\n计算两个数据集之间的相似度...")
            cross_similarity = analyzer.compute_similarity(features_a, features_b)
            cross_analysis = analyzer.analyze_similarity(cross_similarity, args.threshold)
            
            # 输出结果
            print("\n跨数据集相似度分析结果:")
            print(f"- 平均相似度: {cross_analysis['average_similarity']:.4f}")
            print(f"- 相似度超过 {args.threshold} 的样本比例: {cross_analysis['proportion_above_threshold']:.4f}")
            print(f"- 平均最大相似度: {cross_analysis['average_max_similarity']:.4f}")
            
            # 保存结果
            with open(os.path.join(args.output_dir, 'cross_similarity_results.json'), 'w') as f:
                json.dump({
                    "average_similarity": float(cross_analysis['average_similarity']),
                    "proportion_above_threshold": float(cross_analysis['proportion_above_threshold']),
                    "average_max_similarity": float(cross_analysis['average_max_similarity']),
                    "threshold": args.threshold
                }, f, indent=2)
            
            # 可视化
            if args.visualize:
                print("生成可视化结果...")
                analyzer.visualize_similarity(cross_similarity, output_dir=args.output_dir)
                
                # 合并特征进行t-SNE可视化
                combined_features = torch.cat([features_a, features_b])
                labels = np.array(['数据集A'] * len(features_a) + ['数据集B'] * len(features_b))
                analyzer.visualize_tsne(combined_features, labels, output_dir=args.output_dir)
    

    
    # 可视化
    if args.visualize:
        print("生成可视化结果...")
        analyzer.visualize_similarity(internal_similarity, valid_paths_a, output_dir=args.output_dir)
        analyzer.visualize_tsne(features_a, output_dir=args.output_dir)
    
    print(f"\n所有结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main()