import os
import re
import glob
from typing import List, Tuple
from PIL import Image, ImageOps, ImageSequence
from tqdm import tqdm


spr_model = "ASFormer"
selection_method = "4_Video_wise_Top1"
concept_sets = "ChoLec-270"
neuron_concepts = f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"


ROWS, COLS = 4, 5
CELLS_PER_GRID = ROWS * COLS  
PAD = 0
DURATION_MS = 200

SRC_BASE = f"visualized_neuron_concept/{spr_model}/{selection_method}"
DST_BASE = f"visualized_neuron_concept_{ROWS*COLS}/{spr_model}/{selection_method}"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_seq_num(path: str) -> int:
    m = re.search(r"seq(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def to_rgb_on_black(im: Image.Image) -> Image.Image:
    """투명/반투명 → 검정 배경에 합성(알파만 처리, 색상은 건드리지 않음)."""
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info) or (im.mode == "P"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        composited = Image.alpha_composite(bg, rgba)
        return composited.convert("RGB")
    return im.convert("RGB")


def load_gif_frames(path: str) -> Tuple[List[Image.Image], int]:
    """GIF → RGB 프레임 리스트(+ 대표 duration)"""
    try:
        im = Image.open(path)
    except Exception:
        return [], DURATION_MS

    frames, durations = [], []
    for fr in ImageSequence.Iterator(im):
        rgb = to_rgb_on_black(fr)  # 투명만 검정으로
        frames.append(rgb.copy())
        d = fr.info.get("duration", 0) or 0
        durations.append(d)

    if not frames:
        return [], DURATION_MS

    valid = [d for d in durations if d and d > 0]
    duration = int(round(sum(valid) / len(valid))) if valid else DURATION_MS
    return frames, duration


def pad_without_resize(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """스케일 0: 리사이즈 없이 검정 캔버스에 '중앙 패딩'만."""
    W, H = target_size
    bg = Image.new("RGB", (W, H), (0, 0, 0))
    x = (W - img.width) // 2
    y = (H - img.height) // 2
    bg.paste(img, (x, y))
    return bg


def make_global_palette(frames: List[Image.Image], max_samples: int = 24) -> Image.Image:
    """
    전역 팔레트 생성을 위해 여러 프레임을 한 캔버스에 모아
    adaptive 256 팔레트로 변환한 'P' 이미지 반환.
    """
    if not frames:
        # 안전장치: 빈 팔레트 방지
        tmp = Image.new("RGB", (16, 16), (0, 0, 0))
        return tmp.convert("P", palette=Image.ADAPTIVE, colors=256)

    n = min(len(frames), max_samples)
    # 균등 샘플
    idxs = [round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)]
    samples = [frames[i] for i in idxs]

    w, h = samples[0].size
    sheet = Image.new("RGB", (w, h * n))
    for i, f in enumerate(samples):
        sheet.paste(f, (0, i * h))

    pal_img = sheet.convert("P", palette=Image.ADAPTIVE, colors=256)

    # ✅ 팔레트 0번을 확실히 검정(0,0,0)으로 고정
    pal = pal_img.getpalette()
    pal[0:3] = [0, 0, 0]
    pal_img.putpalette(pal)

    # 혹시 모를 투명 메타 제거
    pal_img.info.pop("transparency", None)
    return pal_img


def build_grid_animation(gif_paths: List[str], out_path: str):
    """
    gif_paths (<=12)를 3x4 그리드로 합친 애니메이션 GIF 저장.
    - 리사이즈 금지(무손실 지향): 셀 크기는 입력 프레임 중 최대 (W,H)
    - 부족한 칸은 검정 프레임
    - 전역 팔레트 256, dither 없음
    """
    ensure_dir(os.path.dirname(out_path))

    # 입력 GIF 로드
    loaded: List[List[Image.Image]] = []
    durations = []
    for p in gif_paths:
        frames, dur = load_gif_frames(p)
        loaded.append(frames)
        durations.append(dur)

    # 셀 크기 = 모든 입력 프레임의 최대 (W,H)  → 스케일링 없이 패딩만
    max_w, max_h = 0, 0
    for frames in loaded:
        if frames:
            w, h = frames[0].size  # 일반적으로 GIF 내 동일
            max_w = max(max_w, w)
            max_h = max(max_h, h)
    if max_w == 0 or max_h == 0:
        max_w, max_h = 640, 360  # fallback

    cell_size = (max_w, max_h)

    # 패딩만 적용 (리사이즈 없음)
    padded_sets: List[List[Image.Image]] = []
    for frames in loaded:
        if not frames:
            padded_sets.append([])
            continue
        ps = [pad_without_resize(f, cell_size) for f in frames]
        padded_sets.append(ps)

    while len(padded_sets) < CELLS_PER_GRID:
        padded_sets.append([])

    max_T = max((len(ps) for ps in padded_sets), default=1) or 1

    cell_w, cell_h = cell_size
    W = COLS * cell_w + (COLS - 1) * PAD
    H = ROWS * cell_h + (ROWS - 1) * PAD
    black_cell = Image.new("RGB", cell_size, (0, 0, 0))
    frames_out: List[Image.Image] = []

    for t in range(max_T):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        for idx in range(CELLS_PER_GRID):
            r, c = divmod(idx, COLS)
            x = c * (cell_w + PAD)
            y = r * (cell_h + PAD)
            ps = padded_sets[idx]
            fr = ps[t % len(ps)] if ps else black_cell
            canvas.paste(fr, (x, y))
        frames_out.append(canvas)

    # 전역 팔레트 생성(프레임 간 색상 일관성 ↑, 디더링 0)
    pal_img = make_global_palette(frames_out, max_samples=24)

    # ✅ 프레임별 양자화 + transparency 메타 제거
    quantized = []
    for im in frames_out:
        q = im.quantize(palette=pal_img, dither=Image.Dither.NONE)
        q.info.pop("transparency", None)  # 투명 인덱스 제거
        quantized.append(q)

    # 대표 duration: 입력 중 유효 최대값(원본 타이밍 유지 지향)
    duration_out = max([d for d in durations if d and d > 0], default=DURATION_MS)

    # GIF 저장 (최대한 보수적 설정)
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_out,
        loop=0,
        optimize=False,
        disposal=2,
        background=0,           # ✅ 배경 인덱스 = 0 (검정)
        # transparency는 전달하지 않음(설정 금지)
    )


    # ---- 완전 무손실이 꼭 필요하면 APNG 출력(주석 해제) ----
    # frames_out[0].save(
    #     out_path.replace(".gif", ".png"),
    #     save_all=True,
    #     append_images=frames_out[1:],
    #     duration=duration_out,
    #     loop=0,
    #     format="PNG",  # APNG로 저장 (Pillow 9.2+)
    # )


def main():
    ensure_dir(DST_BASE)
    neuron_dirs = sorted(glob.glob(os.path.join(SRC_BASE, "neuron*")))
    if not neuron_dirs:
        print(f"[WARN] No neuron folders under: {SRC_BASE}")
        return

    for ndir in tqdm(neuron_dirs, desc="Neurons", dynamic_ncols=True):
        gifs = sorted(glob.glob(os.path.join(ndir, "*.gif")), key=parse_seq_num)
        if not gifs:
            continue

        rel = os.path.relpath(ndir, SRC_BASE)  # neuronXXX
        out_neuron_dir = os.path.join(DST_BASE, rel)
        ensure_dir(out_neuron_dir)

        for i in range(0, len(gifs), CELLS_PER_GRID):
            chunk = gifs[i:i + CELLS_PER_GRID]
            start_num = parse_seq_num(chunk[0]) if chunk else i
            end_num = parse_seq_num(chunk[-1]) if chunk else (i + len(chunk) - 1)
            out_name = f"seq{start_num:03d}-{end_num:03d}.gif"
            out_path = os.path.join(out_neuron_dir, out_name)
            build_grid_animation(chunk, out_path)

    print(f"[OK] Saved 3x4 grid GIFs under: {DST_BASE}")


if __name__ == "__main__":
    main()
