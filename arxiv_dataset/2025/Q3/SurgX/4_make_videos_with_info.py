import os
import re
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageSequence

spr_model = "ASFormer"
selection_method = "4_Video_wise_Top1"
concept_sets = "ChoLec-270"
neuron_concepts = f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"


# ========== 경로 설정 ==========
SRC_ROOT  = Path(f"visualized_neuron_concept_20/{spr_model}/{selection_method}")
DST_ROOT  = Path(f"visualized_neuron_concept_20_with_concept_info_integrity/{spr_model}/{selection_method}")

# 업로드 파일(노트북 등)에 맞춰 필요시 바꾸세요
JSON_PATHS_TRY = [
    Path(f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json"),
    Path(f"/mnt/data/{selection_method}_and_{concept_sets}.json"),
]

# ========== 표시 옵션 ==========
SCALE = 3

PADDING_X = 24 * SCALE
PADDING_Y = 16 * SCALE
LINE_SPACING = 6 * SCALE
MAX_WIDTH_RATIO = 0.94
FONT_SIZE_TITLE = 26 * SCALE
FONT_SIZE_BODY  = 22 * SCALE
ACT_ROUND = 3
MARGIN_MIN = 80 * SCALE

PALETTE = [                  # 예쁜 색상 팔레트(라인별 순환)
    (235, 111, 146),  # 핑크
    (64, 160, 255),   # 블루
    (64, 192, 87),    # 그린
    (250, 179, 135),  # 오렌지
    (245, 219, 71),   # 옐로
    (136, 57, 239),   # 퍼플
]

# 시스템에 폰트가 없다면 기본 폰트 사용
def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_TITLE = load_font(FONT_SIZE_TITLE)
FONT_BODY  = load_font(FONT_SIZE_BODY)

NEURON_DIR_RE = re.compile(r"^neuron(?P<idx>\d+)$", re.IGNORECASE)

# ... (다른 함수들은 변경 없음) ...
def load_concept_map(json_paths: List[Path]) -> Dict[int, List[Tuple[str, float]]]:
    path = None
    for jp in json_paths:
        if jp.exists(): path = jp; break
    if path is None: raise FileNotFoundError(f"JSON not found in {json_paths}")
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    mapping: Dict[int, List[Tuple[str, float]]] = {}
    for item in data:
        idx = int(item.get("neuron_idx"))
        pairs = []
        for c in item.get("selected_concepts_sorted", []):
            name = c.get("name", "").strip()
            score = float(c.get("avg_cos_sim", 0.0))
            pairs.append((name, score))
        mapping[idx] = pairs
    return mapping

def compute_fixed_margin(concept_map: Dict[int, List[Tuple[str, float]]]) -> int:
    tmp_img = Image.new("RGB", (1920, 1080)); tmp_draw = ImageDraw.Draw(tmp_img)
    max_text_h = 0
    for n_idx, pairs in concept_map.items():
        lines = format_concept_lines(n_idx, pairs)
        lines_fmt = []
        for li, text in enumerate(lines):
            font = FONT_TITLE if li == 0 else FONT_BODY
            lines_fmt.append((text, font, None))
        _, text_h = measure_text_block(tmp_draw, lines_fmt, LINE_SPACING)
        max_text_h = max(max_text_h, text_h)
    if max_text_h == 0:
        fallback_lines_fmt = [("NEURON 000 ...", FONT_TITLE, None), ("1. (no learned concepts)", FONT_BODY, None)]
        _, max_text_h = measure_text_block(tmp_draw, fallback_lines_fmt, LINE_SPACING)
    return max(MARGIN_MIN, max_text_h + 2 * PADDING_Y)

def format_concept_lines(neuron_idx: int, pairs: List[Tuple[str, float]]) -> List[str]:
    header = f"NEURON {neuron_idx:03d} learned concepts (avg cos sim)"
    if not pairs: return [header, "1. (no learned concepts)"]
    lines = [header]
    for i, (name, score) in enumerate(pairs, start=1):
        lines.append(f"{i}. {name} ({round(score, ACT_ROUND)})")
    return lines

def measure_text_block(draw: ImageDraw.ImageDraw, lines_fmt, line_spacing: int) -> Tuple[int, int]:
    widths, heights = [], []
    for text, font, _ in lines_fmt:
        bbox = draw.textbbox((0, 0), text, font=font)
        widths.append(bbox[2] - bbox[0]); heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_spacing * (len(heights) - 1)
    return (max(widths) if widths else 0, total_h)

def draw_text_with_shadow(draw: ImageDraw.ImageDraw, xy, text: str, font, fill: Tuple[int,int,int]):
    x, y = xy; shadow = (0, 0, 0)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)

def process_gif(gif_path: Path, dst_path: Path, lines_fmt, fixed_margin_h: int):
    with Image.open(gif_path) as im0:
        w, h = im0.size
        frames_rgba = []
        durations = []
        loop = im0.info.get("loop", 0)

        margin_h = fixed_margin_h

        for frame in ImageSequence.Iterator(im0):
            durations.append(frame.info.get("duration", 40))
            fr = frame.convert("RGBA")
            canvas = Image.new("RGBA", (w, h + margin_h), (0, 0, 0, 255))
            canvas.paste(fr, (0, 0))
            draw = ImageDraw.Draw(canvas)
            x0, cursor_y = PADDING_X, h + PADDING_Y
            for (text, font, color) in lines_fmt:
                draw_text_with_shadow(draw, (x0, cursor_y), text, font=font, fill=color)
                bbox = draw.textbbox((x0, cursor_y), text, font=font)
                cursor_y += (bbox[3] - bbox[1]) + LINE_SPACING
            frames_rgba.append(canvas)

        # ========== 여기부터 수정 ==========
        # 1. 모든 프레임을 합친 대표 이미지로 255색 전역 팔레트 생성
        sheet = Image.new("RGBA", (w * len(frames_rgba), h))
        for i, frame in enumerate(frames_rgba):
            sheet.paste(frame, (i * w, 0))
        palette_image = sheet.convert("RGB").quantize(colors=255, dither=Image.Dither.NONE)

        # 2. 각 프레임을 전역 팔레트로 변환하고, 투명도를 인덱스로 직접 처리
        p_frames = []
        for frame_rgba in frames_rgba:
            # 우선 팔레트 모드로 변환
            p_frame = frame_rgba.convert("RGB").quantize(palette=palette_image, dither=Image.Dither.NONE)
            
            # 원본의 알파 채널을 가져와 마스크 생성
            # (알파값이 128 미만이면 투명 처리)
            alpha = frame_rgba.split()[-1]
            mask = Image.eval(alpha, lambda a: 255 if a < 128 else 0)
            
            # 마스크를 이용해 투명해야 할 부분에 255번 인덱스를 덮어씌움
            p_frame.paste(255, mask=mask)
            
            p_frames.append(p_frame)
        
        # 3. 팔레트 모드로 변환된 프레임들과 transparency 옵션을 사용해 저장
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        p_frames[0].save(
            dst_path,
            save_all=True,
            append_images=p_frames[1:],
            duration=durations,
            loop=loop,
            optimize=False,
            disposal=2,
            transparency=255 # 255번 인덱스를 투명색으로 지정
        )
        # ========== 여기까지 수정 ==========

def main():
    concept_map = load_concept_map(JSON_PATHS_TRY)
    fixed_margin_h = compute_fixed_margin(concept_map)

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SRC_ROOT}")

    neuron_dirs = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir() and NEURON_DIR_RE.match(p.name)],
                         key=lambda p: int(NEURON_DIR_RE.match(p.name)["idx"]))

    for ndir in neuron_dirs:
        m = NEURON_DIR_RE.match(ndir.name)
        n_idx = int(m["idx"])
        pairs = concept_map.get(n_idx, [])
        lines = format_concept_lines(n_idx, pairs)
        wrapped_lines = []
        for li, text in enumerate(lines):
            font = FONT_TITLE if li == 0 else FONT_BODY
            color = (255,255,255) if li == 0 else PALETTE[(li - 1) % len(PALETTE)]
            wrapped_lines.append((text, font, color))

        gifs = sorted([p for p in ndir.glob("*.gif")])
        for idx, gif_path in enumerate(gifs):
            dst_dir = DST_ROOT / ndir.name
            dst_name = f"seq{idx:03d}.gif"
            dst_path = dst_dir / dst_name
            process_gif(gif_path, dst_path, wrapped_lines, fixed_margin_h)

    print("[DONE] 모든 GIF 처리 완료!")

if __name__ == "__main__":
    main()