#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageSequence, ImageDraw, UnidentifiedImageError

# ===== 경로 =====
SRC_ROOT = Path("./visualized_neuron_concept_20_with_concept_info_integrity/ASFormer/4_Video_wise_Top1")
DST_ROOT = Path("./visualized_neuron_concept_20_with_concept_info_final/ASFormer/4_Video_wise_Top1")
DST_ROOT.mkdir(parents=True, exist_ok=True)

# ===== 설정 =====
DEFAULT_DURATION = 100  # ms
PRINT_SKIPPED = True

# imageio 사용 가능하면 최우선
try:
    import imageio.v3 as iio  # 최신
    HAS_IMAGEIO_V3 = True
except Exception:
    HAS_IMAGEIO_V3 = False
try:
    import imageio  # 구버전 fallback
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False


# -----------------------------
# 정렬/탐색 유틸
# -----------------------------
def natural_key(p: Path):
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_neuron_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("neuron")], key=natural_key)

def list_seq_gifs(neuron_dir: Path) -> List[Path]:
    gifs = [p for p in neuron_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".gif" and p.name.startswith("seq")]
    return sorted(gifs, key=natural_key)


# -----------------------------
# Pillow 엄격 복원기 (fallback)
# -----------------------------
def _frame_bbox(im: Image.Image) -> Tuple[int, int, int, int]:
    bbox = getattr(im, "dispose_extent", None)
    if bbox and isinstance(bbox, tuple) and len(bbox) == 4:
        return bbox
    try:
        if im.tile and len(im.tile) > 0:
            _, region, _ = im.tile[0][:3]
            if region and len(region) >= 4:
                return region[:4]
    except Exception:
        pass
    return (0, 0, im.width, im.height)

def _palette_index_to_rgba(im: Image.Image, idx: int) -> Tuple[int, int, int, int]:
    pal = im.getpalette()
    if pal is None:
        return (0, 0, 0, 255)
    base = 3 * idx
    r = pal[base + 0] if base + 0 < len(pal) else 0
    g = pal[base + 1] if base + 1 < len(pal) else 0
    b = pal[base + 2] if base + 2 < len(pal) else 0
    return (r, g, b, 255)

def _gif_background_rgba(im: Image.Image) -> Tuple[int, int, int, int]:
    # GIF에 background 인덱스가 있으면 해당 팔레트 색으로, 없으면 불투명 검정
    bg_idx = im.info.get("background", None)
    if isinstance(bg_idx, int):
        try:
            return _palette_index_to_rgba(im, bg_idx)
        except Exception:
            pass
    return (0, 0, 0, 255)

def decode_gif_full_frames_pillow(gif_path: Path) -> Tuple[List[Image.Image], List[int]]:
    """
    Pillow만으로 GIF를 '풀 캔버스 RGBA 프레임들'로 엄격 복원.
    disposal 2/3, 투명, 오프셋(region) 모두 처리.
    """
    frames: List[Image.Image] = []
    durations: List[int] = []

    with Image.open(gif_path) as im:
        canvas_w, canvas_h = im.size
        bg_rgba = _gif_background_rgba(im)

        accum = Image.new("RGBA", (canvas_w, canvas_h), bg_rgba)
        accum_prev = None

        n = getattr(im, "n_frames", 1)
        for i in range(n):
            im.seek(i)
            # 오프셋/영역
            x0, y0, x1, y1 = _frame_bbox(im)
            # 현재 프레임 RGBA (팔레트/투명 반영)
            cur = im.convert("RGBA")

            # disposal=3 대비
            accum_prev = accum.copy()

            # 정확한 위치에 합성
            accum.paste(cur, (x0, y0), cur)

            # 이 시점의 풀프레임 캡쳐
            frames.append(accum.copy())

            # duration
            dur = im.info.get("duration", DEFAULT_DURATION)
            durations.append(int(dur) if dur and dur > 0 else DEFAULT_DURATION)

            # disposal 처리
            disposal = getattr(im, "disposal_method", None)
            if disposal is None:
                disposal = getattr(im, "disposal", 0)
            if disposal == 2:
                draw = ImageDraw.Draw(accum)
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=bg_rgba)
                del draw
            elif disposal == 3 and accum_prev is not None:
                accum = accum_prev

    return frames, durations


# -----------------------------
# imageio 기반 정확 복원 (권장)
# -----------------------------
def decode_gif_full_frames_imageio(gif_path: Path) -> Tuple[List[Image.Image], List[int]]:
    """
    imageio가 내부에서 disposal/offset/투명 처리하여 완전한 프레임 시퀀스를 반환.
    v3가 있으면 그걸, 아니면 v2 mimread.
    """
    frames: List[Image.Image] = []
    durations: List[int] = []

    if HAS_IMAGEIO_V3:
        arr = iio.imread(gif_path, index=None)  # (n, H, W, 4 or 3)
        meta = iio.immeta(gif_path) or {}
        # v3 메타에서 durations(ms) 추출 시 프레임별 정보 지원이 제한적일 수 있어 통일/보정
        dur = int(meta.get("duration", DEFAULT_DURATION)) or DEFAULT_DURATION
        for a in arr:
            # RGBA 보장
            if a.shape[-1] == 3:
                # RGB → RGBA(불투명)
                from numpy import dstack, uint8, ones
                a = dstack([a, 255 * ones(a[..., :1].shape, dtype=uint8)])
            frames.append(Image.fromarray(a, "RGBA"))
            durations.append(dur)
        return frames, durations

    if HAS_IMAGEIO:
        reader = imageio.get_reader(str(gif_path))
        try:
            meta = reader.get_meta_data() or {}
            dur = int(meta.get("duration", DEFAULT_DURATION)) or DEFAULT_DURATION
        except Exception:
            dur = DEFAULT_DURATION
        try:
            for im in reader:
                pil = Image.fromarray(im)  # imageio가 이미 disposal 반영한 풀프레임
                if pil.mode != "RGBA":
                    pil = pil.convert("RGBA")
                frames.append(pil)
                durations.append(dur)
        finally:
            reader.close()
        return frames, durations

    # imageio가 없으면 Pillow fallback
    return decode_gif_full_frames_pillow(gif_path)

# ... (이전 import 및 유틸 함수들은 동일) ...

# -----------------------------
# 메인: 시퀀스 복원 → 이어붙이기 → 저장
# -----------------------------
def concat_sequences_after_full_restore(neuron_dir: Path, out_path: Path):
    seq_paths = list_seq_gifs(neuron_dir)
    if not seq_paths:
        if PRINT_SKIPPED:
            print(f"[SKIP] {neuron_dir.name}: seq*.gif 없음")
        return

    # 먼저 모든 시퀀스를 '풀프레임 RGBA 리스트'로 복원
    restored_frames_all: List[Image.Image] = []
    durations_all: List[int] = []

    iterable = seq_paths
    if use_tqdm:
        iterable = tqdm(seq_paths, desc=f"{neuron_dir.name}", leave=False)

    canvas_w = canvas_h = None
    for gif_path in iterable:
        try:
            frames, durs = decode_gif_full_frames_imageio(gif_path)
        except Exception:
            # imageio 실패 시 Pillow로 재시도
            frames, durs = decode_gif_full_frames_pillow(gif_path)

        if not frames:
            print(f"[WARN] 프레임 복원 실패: {gif_path}")
            continue

        # 캔버스 크기 갱신(최대)
        w, h = frames[0].size
        if canvas_w is None:
            canvas_w, canvas_h = w, h
        else:
            canvas_w = max(canvas_w, w)
            canvas_h = max(canvas_h, h)

        restored_frames_all.extend(frames)
        durations_all.extend(durs)

    if not restored_frames_all:
        if PRINT_SKIPPED:
            print(f"[SKIP] {neuron_dir.name}: 복원 프레임 없음")
        return

    # ===== 핵심 수정 부분 (투명도 제거 강화) =====
    target_size = (canvas_w, canvas_h)
    final_frames_rgb = [] # RGB 모드로 저장할 최종 프레임 리스트

    # 모든 복원 프레임을 불투명 검정색 캔버스에 합성하여 RGB로 변환
    for fr in restored_frames_all:
        # 1. 불투명 검정색 RGBA 캔버스 생성
        black_background_canvas = Image.new("RGBA", target_size, (0, 0, 0, 255))

        # 2. 원본 프레임(fr)을 검정 캔버스 위에 합성
        #    - fr.convert("RGBA")를 사용하여 일관된 RGBA 모드를 보장
        #    - mask=fr을 명시적으로 사용하여 fr의 알파 채널을 마스크로 활용
        #    - 원본 fr의 크기가 target_size와 다를 경우, 정 중앙에 붙여넣기 (선택 사항)
        #      x_offset = (target_size[0] - fr.width) // 2
        #      y_offset = (target_size[1] - fr.height) // 2
        #      black_background_canvas.paste(fr.convert("RGBA"), (x_offset, y_offset), mask=fr.convert("RGBA"))
        #    - 일단은 (0,0)에 붙여넣기로 유지
        black_background_canvas.paste(fr, (0, 0), mask=fr) # mask=fr 추가!

        # 3. 최종적으로 알파 채널을 제거하여 RGB 모드로 변환
        final_frames_rgb.append(black_background_canvas.convert("RGB"))
    # ============================================

    # 저장: disposal=1(Do not dispose), optimize=False — 이미 풀프레임이므로 안전
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = final_frames_rgb[0], final_frames_rgb[1:]

    # Pillow에 GIF 배경색 정보를 직접 전달 (가장 중요한 부분)
    # background=0은 팔레트의 첫 번째 색상(보통 검정)을 의미합니다.
    # 하지만 RGB 모드에서는 직접 튜플(0,0,0)을 전달하는 것이 더 안전할 수 있습니다.
    # 여기서는 RGB 모드이므로 background를 명시적으로 (0,0,0)으로 지정합니다.
    # Pillow 9.0.0 이후 버전부터는 save 함수의 background 인자에 RGB 튜플이 가능합니다.
    # 만약 에러가 발생하면 0으로 변경해 보세요.
    save_options = {
        "save_all": True,
        "append_images": rest,
        "duration": durations_all,
        "loop": 0,
        "optimize": False, # RGB 모드에서는 True로 해도 괜찮지만, 안전하게 False 유지
        "disposal": 1,
        "background": (0, 0, 0), # <--- 이 부분이 핵심입니다! GIF 헤더에 검정색 배경 명시
    }

    try:
        first.save(out_path, **save_options)
    except TypeError as e:
        print(f"[WARN] Pillow background=(0,0,0) 오류 발생: {e}. background=0으로 재시도.")
        save_options["background"] = 0 # 팔레트 0번 인덱스를 배경색으로 지정 (보통 검정)
        first.save(out_path, **save_options)
    
    print(f"[OK]  {neuron_dir.name} -> {out_path}")

def main():
    if not SRC_ROOT.exists():
        print(f"[ERROR] 입력 경로 없음: {SRC_ROOT.resolve()}")
        return
    neuron_dirs = list_neuron_dirs(SRC_ROOT)
    if not neuron_dirs:
        print(f"[ERROR] 뉴런 폴더를 찾을 수 없음: {SRC_ROOT}")
        return

    iterable = neuron_dirs
    if use_tqdm:
        iterable = tqdm(neuron_dirs, desc="Restoring & concatenating")

    for nd in iterable:
        out_path = DST_ROOT / f"{nd.name}.gif"
        concat_sequences_after_full_restore(nd, out_path)

if __name__ == "__main__":
    main()