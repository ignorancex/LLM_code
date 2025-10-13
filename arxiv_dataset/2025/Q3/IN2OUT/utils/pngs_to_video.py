import cv2
import os

def images_to_video(img_dir, output_video_name, fps=30):
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

    files.sort()
    if not files:
        raise ValueError("No PNG images found in the specified directory!")

    frame = cv2.imread(files[0])
    h, w, layers = frame.shape
    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_name, fourcc, fps, size)
    for file in files:
        img = cv2.imread(file)
        out.write(img)

    out.release()

img_directory = 'results_ours_no_fcm/e2fgvi_hq_davis'
output_diretory = 'results_ours_davis_mp4'
if not os.path.exists(output_diretory):
    os.mkdir(output_diretory)

for vid in os.listdir(img_directory):
    if os.path.isdir(os.path.join(img_directory, vid)):
        images_to_video(os.path.join(img_directory, vid), f'{output_diretory}/{vid}.mp4', fps=20)
