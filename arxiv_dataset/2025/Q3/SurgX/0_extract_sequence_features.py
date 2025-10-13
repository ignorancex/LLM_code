import os
import glob
import pickle
from typing import List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from mmengine.config import Config
import SurgVLP.surgvlp as surgvlp

# =========================
# PeskaVLP image encoder 로드
# =========================
def load_peskavlp_image_encoder(config_path: str, device: str = None):
    """
    config_path: 예) './config_peskavlp.py'
    반환: (model, preprocess_transform, feature_dim)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = Config.fromfile(config_path, lazy_import=False)['config']
    model, _ = surgvlp.load(configs.model_config)
    model = model.to(device)
    model.eval()

    # 전처리: 일반적으로 ImageNet 통계 (제로샷 코드의 unnormalize 참고)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 보통 img_emb 차원은 768 또는 1024 구성에 따라 다름.
    # 안전하게 한 번 더미로 확인할 수도 있지만, 여기선 768로 가정.
    feature_dim = 768
    return model, preprocess, feature_dim, device

# =========================
# 경로/전처리/인코딩 유틸
# =========================
def tuple_to_paths(vid: int, frame_1fps: int) -> List[str]:
    """
    1fps → 25fps로 변환(base=f*25+1) 후,
    현재 프레임 포함 5초(=125프레임) 간격으로 9장 더해 최대 10장(존재 파일만)
    """
    vid += 1
    base = frame_1fps * 25 + 1
    paths = []
    for i in range(10):
        fnum = base - i * 125
        if fnum <= 0:
            break
        p = f"/data2/local_datasets/cholec80/cholec_split_360x640_1fps/video{vid:02d}/video{vid:02d}_{fnum:06d}.png"
        if os.path.exists(p):
            paths.append(p)
    return paths

def load_images_as_batch(paths: List[str], preprocess, device: str) -> torch.Tensor:
    """
    여러 이미지를 한 번에 로드하여 (N,3,224,224) 텐서로 반환
    """
    batch = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
    if len(batch) == 0:
        return torch.empty(0, 3, 224, 224, device=device)
    batch = torch.stack(batch, dim=0).to(device)
    return batch

@torch.no_grad()
def peskavlp_encode_images(model, images: torch.Tensor) -> torch.Tensor:
    """
    images: (N,3,224,224)
    반환: (N, D) img_emb (L2 정규화)
    """
    if images.numel() == 0:
        return images.new_zeros((0, 768))
    out = model(images, None, mode='video')   # 제로샷 코드와 동일한 호출
    img_emb = out['img_emb']                  # (N, D)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb

@torch.no_grad()
def extract_tuple_feature(model, preprocess, device: str, vid: int, frame_1fps: int, feature_dim: int) -> torch.Tensor:
    """
    하나의 (vid, frame_1fps) 튜플에 대해 10장(존재하는 것만) feature 평균 (D,) 반환
    """
    paths = tuple_to_paths(vid, frame_1fps)
    if not paths:
        return torch.zeros(feature_dim, device=device)
    images = load_images_as_batch(paths, preprocess, device)  # (N,3,224,224)
    feats = peskavlp_encode_images(model, images)             # (N,D)
    return feats.mean(dim=0)                                  # (D,)

# =========================
# 메인 처리 루틴
# =========================
def process_one_pkl(input_pkl: str, output_dir: str, model, preprocess, device: str, feature_dim: int):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)  # [ [ (vid, frame, val), ... ], [ ... ], ... ]  (뉴런 단위 리스트)

    per_neuron_tensors = []  # 각 원소: (num_tuples, D)

    for neuron in tqdm(data, desc=f"[{os.path.basename(input_pkl)}] Neurons", position=0, leave=True):
        tuple_feat_list = []
        for tup in tqdm(neuron, desc="  Tuples", position=1, leave=False):
            if not isinstance(tup, (list, tuple)) or len(tup) < 2:
                continue
            vid, frame_1fps = int(tup[0]), int(tup[1])
            feat = extract_tuple_feature(model, preprocess, device, vid, frame_1fps, feature_dim)  # (D,)
            tuple_feat_list.append(feat)

        if len(tuple_feat_list) == 0:
            neuron_tensor = torch.empty(0, feature_dim)
        else:
            neuron_tensor = torch.stack(tuple_feat_list, dim=0)  # (num_tuples, D)

        per_neuron_tensors.append(neuron_tensor.cpu())

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_pkl))
    for p in per_neuron_tensors:
        print(p.shape)
    with open(out_path, "wb") as f:
        pickle.dump(per_neuron_tensors, f)

    print(f"✔ Saved -> {out_path}")

def main():
    input_dir  = "spr_models/ASFormer/representative_frames"
    output_dir = "extracted_features/sequence"
    config_path = "SurgVLP/tests/config_peskavlp.py"   # ← 제로샷 코드와 동일한 설정 파일 경로 사용

    model, preprocess, feature_dim, device = load_peskavlp_image_encoder(config_path)
    pkl_files = glob.glob(os.path.join(input_dir, "*.pkl"))

    for idx, pkl_path in enumerate(pkl_files, start=1):
        print(f"\n=== [{idx}/{len(pkl_files)}] Processing {os.path.basename(pkl_path)} ===")
        process_one_pkl(pkl_path, output_dir, model, preprocess, device, feature_dim)

if __name__ == "__main__":
    main()
