import argparse
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm 
import os

def get_video_dimensions(filename):
    video_clip = VideoFileClip(filename)
    width, height = video_clip.size
    duration = video_clip.duration
    fps = video_clip.fps
    num_frames = int(duration * fps)
    
    print(f"Width: {width}, Height: {height}")
    print(f"Duration: {duration} seconds")
    print(f"Frames per second (FPS): {fps}")
    print(f"Total number of frames: {num_frames}")
    return width, height, num_frames

def truncate_video(filename, output, num_frames):
    video_clip = VideoFileClip(filename)
    fps = video_clip.fps
    duration = num_frames / fps
    
    truncated_clip = video_clip.subclip(0, duration)
    truncated_clip.write_videofile(output, codec="libx264")
    print(f"Video truncated to {num_frames} frames and saved as {output}")

def generate_png_image(w1, w2, h):
    image = Image.new("RGB", (w2, h), color="white")
    for y in range(h):
        for x in range((w2 - w1) // 2, (w2 + w1) // 2):
            image.putpixel((x, y), (0, 0, 0))
    
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Generaetion")
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("-k", "--divider", type=int, required=True)
    parser.add_argument("--max_frames", type=int, default=512)
    args = parser.parse_args()

    name, k = args.video, args.divider
    max_frames = args.max_frames
    mp4_file_path = os.path.join(f"{name}/video/", os.listdir(f"{name}/video/")[0])
    output_directory = f"{name}/mask_1_{k}"
    os.makedirs(output_directory, exist_ok=True)

    w, h, num_frames = get_video_dimensions(mp4_file_path)
    image = generate_png_image(w // k * (k-1), w, h)

    for i in tqdm(range(max_frames)):
        mask_name = str(i).zfill(5) + ".png"
        mask_path = os.path.join(output_directory, mask_name)
        image.save(mask_path)
