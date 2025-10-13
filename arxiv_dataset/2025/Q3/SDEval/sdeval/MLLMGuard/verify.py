import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import json
import os
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPModel, CLIPProcessor
import argparse

def load_local_features(feature_path):
    """加载本地数据集的特征"""
    return torch.load(feature_path)

def compute_feature_hash(feature, bits=64):
    """计算特征的哈希值（与LAION-400M使用的方法兼容）"""
    binary = (feature > 0).cpu().numpy().astype(np.uint8).flatten()
    hash_value = 0
    for bit in binary[:bits]:
        hash_value = (hash_value << 1) | bit
    return hash_value

def compare_with_laion(local_features, laion_dataset, bits=64, sample_ratio=1.0):
    """比较本地特征与LAION-400M的特征哈希"""
    potential_matches = []
    
    # 计算本地特征的哈希值
    local_hashes = {
        path: compute_feature_hash(feat, bits)
        for path, feat in tqdm(local_features.items(), desc="计算本地特征哈希")
    }
    
    # 对LAION数据集进行抽样（如果需要）
    if sample_ratio < 1.0:
        laion_size = int(len(laion_dataset) * sample_ratio)
        laion_dataset = laion_dataset.shuffle(seed=42).select(range(laion_size))
        print(f"使用LAION-400M的{sample_ratio:.0%}样本进行比较 ({laion_size}条)")
    
    # 构建LAION哈希到样本的映射
    laion_hash_to_idx = {}
    for i, sample in enumerate(tqdm(laion_dataset, desc="处理LAION元数据")):
        laion_hash = sample["CLIP_hash"]  # LAION-400M中的特征哈希字段
        if laion_hash in laion_hash_to_idx:
            laion_hash_to_idx[laion_hash].append(i)
        else:
            laion_hash_to_idx[laion_hash] = [i]
    
    # 查找匹配的哈希值
    for local_path, local_hash in tqdm(local_hashes.items(), desc="查找匹配项"):
        if local_hash in laion_hash_to_idx:
            for laion_idx in laion_hash_to_idx[local_hash]:
                laion_sample = laion_dataset[laion_idx]
                potential_matches.append({
                    "local_path": local_path,
                    "laion_url": laion_sample["URL"],
                    "laion_caption": laion_sample["TEXT"],
                    "laion_hash": laion_sample["CLIP_hash"],
                    "local_hash": local_hash
                })
    
    return potential_matches

def verify_single_match(match, local_features, clip_model, clip_processor, device="cuda", timeout=10):
    """验证单个匹配项的精确相似度"""
    try:
        # 从网络加载LAION图像
        response = requests.get(match["laion_url"], timeout=timeout)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # 提取LAION图像的特征
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            laion_feature = clip_model.get_image_features(**inputs)
            laion_feature = torch.nn.functional.normalize(laion_feature, p=2, dim=1)
        
        # 加载本地特征
        local_feature = local_features[match["local_path"]].to(device)
        
        # 计算精确相似度
        similarity = torch.nn.functional.cosine_similarity(local_feature, laion_feature).item()
        
        return {**match, "similarity": similarity} if similarity >= args.threshold else None
    
    except Exception as e:
        # print(f"Error verifying {match['laion_url']}: {e}")
        return None

def parallel_verify_matches(potential_matches, local_features, clip_model, clip_processor, 
                           device="cuda", max_workers=10, chunksize=100):
    """并行验证匹配项"""
    verified_matches = []
    
    # 分块处理，避免内存溢出
    for i in tqdm(range(0, len(potential_matches), chunksize), desc="并行验证"):
        chunk = potential_matches[i:i+chunksize]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_match = {executor.submit(
                verify_single_match, 
                match, 
                local_features, 
                clip_model, 
                clip_processor,
                device
            ): match for match in chunk}
            
            for future in as_completed(future_to_match):
                result = future.result()
                if result is not None:
                    verified_matches.append(result)
    
    return verified_matches
def main():
    parser = argparse.ArgumentParser(description='计算本地数据集与LAION-400M的重叠度')
    parser.add_argument('--local_features', type=str, default='/root/code/MLLMGuard/features_dataset_a.pt', help='本地特征文件路径')
    parser.add_argument('--output_dir', type=str, default='/root/code/MLLMGuard/mmguarl_sim', help='结果输出目录')
    parser.add_argument('--threshold', type=float, default=0.8, help='相似度阈值')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='LAION-400M抽样比例')
    parser.add_argument('--max_workers', type=int, default=10, help='并行线程数')
    parser.add_argument('--chunksize', type=int, default=100, help='每批处理的匹配项数')
    parser.add_argument('--timeout', type=int, default=10, help='网络请求超时时间（秒）')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备（cuda或cpu）')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载本地特征
    print("加载本地数据集特征...")
    local_features = load_local_features(args.local_features)
    print(f"加载完成: {len(local_features)} 个样本")
    
    # 加载LAION-400M元数据
    print("加载LAION-400M元数据...")
    laion_dataset = load_dataset('laion/laion400m', split='train')
    print(f"加载完成: {len(laion_dataset)} 个样本")
    
    # 比较特征哈希
    print("比较本地特征与LAION-400M特征哈希...")
    potential_matches = compare_with_laion(
        local_features, laion_dataset, sample_ratio=args.sample_ratio
    )
    
    # 保存潜在匹配结果
    with open(os.path.join(args.output_dir, 'potential_matches.json'), 'w') as f:
        json.dump(potential_matches, f, indent=2)
    
    print(f"找到 {len(potential_matches)} 个潜在匹配项")
    
    # 如果有潜在匹配项，验证精确相似度
    if potential_matches:
        print("加载CLIP模型用于验证...")
        clip_model = CLIPModel.from_pretrained("/model/clip").to(args.device)
        clip_processor = CLIPProcessor.from_pretrained("/model/clip")
        
        print(f"使用{args.max_workers}个线程并行验证潜在匹配项...")
        verified_matches = parallel_verify_matches(
            potential_matches, 
            local_features, 
            clip_model, 
            clip_processor,
            device=args.device,
            max_workers=args.max_workers,
            chunksize=args.chunksize
        )
        
        # 保存验证结果
        with open(os.path.join(args.output_dir, 'verified_matches.json'), 'w') as f:
            json.dump(verified_matches, f, indent=2)
        
        print(f"验证完成: {len(verified_matches)} 个匹配项相似度超过阈值 {args.threshold}")
        
        # 计算重叠度
        overlap_ratio = len(verified_matches) / len(local_features)
        print(f"本地数据集与LAION-400M的重叠度: {overlap_ratio:.2%}")
        
        # 保存统计结果
        with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
            json.dump({
                "local_samples": len(local_features),
                "laion_samples_checked": int(len(laion_dataset) * args.sample_ratio),
                "potential_matches": len(potential_matches),
                "verified_matches": len(verified_matches),
                "overlap_ratio": overlap_ratio,
                "threshold": args.threshold
            }, f, indent=2)

if __name__ == "__main__":
    main()