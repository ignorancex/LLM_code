import os
import glob
import pickle
from typing import List, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm
import json

# =============== 기존 tuple_to_paths 그대로 사용 ===============
def tuple_to_paths(vid: int, frame_1fps: int) -> List[str]:
    vid += 1
    base = frame_1fps * 25 + 1
    paths = []
    for i in range(50):
        fnum = base - i * 25
        if fnum <= 0:
            break
        p = f"/data2/local_datasets/cholec80/cholec_split_360x640_1fps/video{vid:02d}/video{vid:02d}_{fnum:06d}.png"
        if os.path.exists(p):
            paths.append(p)
    return paths  # 최신이 앞에 옴


# =============== 새 시각화 유틸 (GIF 생성) ===============
def load_image_safe(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def make_gif_sequence(paths: List[str],
                      out_gif: str,
                      num_frames: int = 10,
                      target_size: Tuple[int, int] = None,
                      duration_ms: int = 200):
    """
    paths: 최신 프레임이 앞에 오는 리스트 (tuple_to_paths 결과)
    - GIF는 오래된→최신 순으로 재생되도록 정렬
    - 프레임이 num_frames 미만이면, 앞쪽을 검정 프레임으로 패딩
    - 모든 프레임 크기를 target_size로 통일 (없으면 첫 유효 프레임 크기, 더 없으면 (640, 360))
    """
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)

    # 기준 크기 결정
    ref_img = None
    for p in paths:
        try:
            ref_img = load_image_safe(p)
            break
        except Exception:
            continue
    if target_size is None:
        if ref_img is not None:
            target_size = ref_img.size  # (W, H)
        else:
            target_size = (640, 360)

    W, H = target_size
    black = Image.new("RGB", (W, H), (0, 0, 0))

    # 오래된→최신 순으로 정렬 (paths는 최신이 앞이므로 뒤집기)
    ordered = list(reversed(paths[:num_frames]))

    # 앞쪽 검정 패딩 개수
    pad_count = num_frames - len(ordered)
    frames = []

    # 앞쪽 패딩
    for _ in range(max(0, pad_count)):
        frames.append(black.copy())

    # 실제 프레임들
    for p in ordered:
        try:
            img = load_image_safe(p)
            # 비율 유지 + 레터박스 패딩(contain)
            img_contained = ImageOps.contain(img, (W, H), method=Image.Resampling.LANCZOS)
            bg = black.copy()
            bg.paste(img_contained, ((W - img_contained.width) // 2,
                                     (H - img_contained.height) // 2))
            frames.append(bg)
        except Exception:
            frames.append(black.copy())

    # 안전장치: 정확히 num_frames로 맞춤
    frames = frames[:num_frames]

    if not frames:
        return

    # GIF 저장 (무한 반복)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


# =============== 메인 루틴 (GIF 저장용) ===============
def process_one_pkl(input_pkl: str, neuron_concept: List, output_dir: str):
    """
    input_pkl: representative_frames *.pkl
      구조: [ [ (vid, frame, val), ... ], [ ... ], ... ]  (뉴런 단위 리스트)
    output: visualized_neuron_concept/{pkl_basename}/neuronXXX/seqYYY.gif
    """
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    base = os.path.splitext(os.path.basename(input_pkl))[0]
    out_dir = os.path.join(output_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    for n_idx, neuron in enumerate(tqdm(data, desc=f"[{os.path.basename(input_pkl)}] Neurons", dynamic_ncols=True)):
        for s_idx, tup in enumerate(tqdm(neuron, desc="  Seqs", leave=False, dynamic_ncols=True)):
            if not isinstance(tup, (list, tuple)) or len(tup) < 2:
                continue
            vid, frame_1fps = int(tup[0]), int(tup[1])
            paths = tuple_to_paths(vid, frame_1fps)

            # out 경로 (GIF)
            out_gif = os.path.join(out_dir, f"neuron{n_idx:03d}", f"seq{s_idx:03d}.gif")

            # 항상 10프레임 GIF로 저장, 비어있는 부분은 검정으로
            make_gif_sequence(paths, out_gif, num_frames=10, target_size=None, duration_ms=10)

    print(f"[OK] GIF sequences saved under: {out_dir}")


def main():
    spr_model = "ASFormer"
    selection_method = "4_Video_wise_Top1"
    concept_sets = "ChoLec-270"
    neuron_concepts = f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"
    
    input_dir  = f"spr_models/{spr_model}/representative_frames"
    output_dir = f"visualized_neuron_concept/{spr_model}"
    os.makedirs(output_dir, exist_ok=True)

    with open(neuron_concepts, "r", encoding="utf-8") as f:
        neuron_concepts = json.load(f)

    pkl_files = glob.glob(os.path.join(input_dir, f"{selection_method}.pkl"))
    for idx, pkl_path in enumerate(sorted(pkl_files), start=0):
        print(f"\n=== [{idx}/{len(pkl_files)}] {os.path.basename(pkl_path)} ===")
        neuron_concept = next(
            nc["selected_concepts_sorted"] 
            for nc in neuron_concepts 
            if isinstance(nc, dict) and nc.get("neuron_idx") == idx
        )
        process_one_pkl(pkl_path, neuron_concept, output_dir)

if __name__ == "__main__":
    main()
