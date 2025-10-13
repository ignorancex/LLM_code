import os
import subprocess

# Set your input folder path
input_folder = "./output"  # change this

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".mp4"):
        mp4_path = os.path.join(input_folder, filename)
        gif_path = os.path.splitext(mp4_path)[0] + ".gif"

        print(f"Converting {filename} to GIF...")

        subprocess.run([
            "ffmpeg", "-i", mp4_path,
            "-vf", "fps=10",  # only set frame rate, no resizing
            "-c:v", "gif", gif_path
        ])

        print(f"Saved: {gif_path}")

