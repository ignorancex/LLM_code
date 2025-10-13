from pathlib import Path
import re
import imageio.v2 as imageio  # pip install imageio imageio-ffmpeg
from PIL import Image
import numpy as np

SRC_ROOT = Path("explanation")
OUT_ROOT = Path("explanation_video_mp4_3")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FRAME_DURATION_SEC = 0.1   # 한 프레임을 보여줄 시간(초) ← 0.5면 2fps와 동일한 효과
CHUNK_SIZE = 300            # 60장씩 저장
OUTPUT_FPS = 30            # 실제 파일의 fps (프레임을 repeat하여 체류시간을 맞춤)
CODEC = "libx264"          # H.264

# 파일명에서 프레임 번호 추출: {videoName}_{######}.png → ######
num_pat = re.compile(r"_(\d+)\.png$", re.IGNORECASE)

def get_frame_number(p: Path, fallback: int) -> int:
    m = num_pat.search(p.name)
    return int(m.group(1)) if m else fallback

def ensure_even_size(img: Image.Image) -> Image.Image:
    """H.264는 가로/세로가 짝수인 것이 안전. 홀수면 마지막 행/열 1px 크롭."""
    w, h = img.size
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return img

# 출력 FPS에서 프레임당 체류시간을 맞추기 위한 반복 횟수
# 예) OUTPUT_FPS=30, FRAME_DURATION_SEC=0.5 → repeat=15
REPEAT = max(1, int(round(FRAME_DURATION_SEC * OUTPUT_FPS)))

for video_dir in sorted(SRC_ROOT.glob("video*")):
    if not video_dir.is_dir():
        continue

    frames = sorted(video_dir.glob("*.png"))
    if not frames:
        print(f"⚠️ {video_dir.name}: PNG 없음")
        continue

    total = len(frames)
    print(f"▶ {video_dir.name}: {total} frames → {CHUNK_SIZE}장씩 MP4 생성")

    # 60장씩 청크 분할
    for start_idx in range(0, total, CHUNK_SIZE):
        chunk = frames[start_idx : start_idx + CHUNK_SIZE]
        start_num = get_frame_number(chunk[0], start_idx + 1)
        end_num   = get_frame_number(chunk[-1], start_idx + len(chunk))

        (OUT_ROOT / f"{video_dir.name}").mkdir(parents=True, exist_ok=True)
        out_mp4 = OUT_ROOT / f"{video_dir.name}/{video_dir.name}_{start_num:06d}-{end_num:06d}.mp4"

        # 첫 프레임으로 해상도 결정(+짝수 보정)
        with Image.open(chunk[0]) as im0:
            im0 = ensure_even_size(im0.convert("RGB"))
            width, height = im0.size

        # ffmpeg 기반 writer
        # macro_block_size=1 → 리사이즈 강제 방지(우리가 직접 짝수로 보정함)
        writer = imageio.get_writer(
            out_mp4,
            fps=OUTPUT_FPS,
            codec=CODEC,
            format="ffmpeg",
            macro_block_size=1,
            quality=8,  # 0(저품질)~10(고품질); 또는 bitrate="4M" 등으로 제어 가능
        )

        try:
            for idx, p in enumerate(chunk, 1):
                with Image.open(p) as pil_img:
                    pil_img = ensure_even_size(pil_img.convert("RGB"))
                    # 프레임당 체류시간을 위해 REPEAT만큼 같은 프레임을 기록
                    frame_np = np.asarray(pil_img)
                    for _ in range(REPEAT):
                        writer.append_data(frame_np)

                if idx % 30 == 0:
                    print(f"… {video_dir.name}: {idx}/{len(chunk)}")

        finally:
            writer.close()

        effective_fps = OUTPUT_FPS / REPEAT  # 체감 FPS(= 1/FRAME_DURATION_SEC와 같아야 함)
        print(f"  ✅ {out_mp4.name} 생성 ({len(chunk)} frames, 체류 {FRAME_DURATION_SEC}s/frame, 체감FPS≈{effective_fps:.3f})")

    print(f"✓ {video_dir.name} 완료\n")
