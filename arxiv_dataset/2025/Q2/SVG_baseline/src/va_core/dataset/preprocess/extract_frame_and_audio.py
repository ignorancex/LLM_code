import argparse
import csv
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm

from torchvision import transforms
import torch


@dataclass
class CsvRow():
    relpath: str
    label: str
    url: str
    start_time: str

def get_subclip_frames_audio(row: CsvRow, basedir, subclip_duration, fps, resolution):
    # Note(2023/06/23):
    #  When we do write_xxx() with logger=None for silent execution, there is an error with ffmpeg encoding.
    #  I'm not sure why but current stdout will be messy due to this issue.

    # read mp4
    orig_vc = VideoFileClip(os.path.join(basedir, row.relpath))
    filename = os.path.basename(row.relpath).split(".")[0]

    # extract subclip and save
    s = int(row.start_time)
    e = min(s + subclip_duration, orig_vc.duration)
    subclip_vc: VideoFileClip = orig_vc.subclip(t_start=s, t_end=e)
    mp4_relpath = f"subclips_{resolution}x{resolution}/orig_res/{filename}_{s}_{e}.mp4"
    subclip_vc.write_videofile(os.path.join(basedir, mp4_relpath), fps=fps)
    
    # save audio
    subclip_ac = subclip_vc.audio
    wav_relpath = f"subclips_{resolution}x{resolution}/{filename}_{s}_{e}.wav"
    subclip_ac.write_audiofile(os.path.join(basedir, wav_relpath))

    # save frames as npy
    cap = cv2.VideoCapture(os.path.join(basedir, mp4_relpath))
    assert cap.isOpened(), f"cannot open video: {mp4_relpath}"
    print(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        is_success, frame = cap.read()
        if not is_success:
            break

        # convert color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)
    
    # Resize and crop frames here to reduce file size
    tf = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution)
        ]
    )
    frames_pt = torch.from_numpy(np.asarray(frames).transpose(0, 3, 1, 2))
    frames = tf(frames_pt).cpu().detach().numpy().transpose(0, 2, 3, 1)

    npy_relpath = f"subclips_{resolution}x{resolution}/{filename}_{s}_{e}.npz"
    np.savez(os.path.join(basedir, npy_relpath), frames)

    # save mp4 with the specified resolution
    subclip_cr = ImageSequenceClip(list(frames), fps=fps)
    subclip_cr = subclip_cr.set_audio(subclip_ac)
    mp4_cr_relpath = f"subclips_{resolution}x{resolution}/{filename}_{s}_{e}.mp4"
    subclip_cr.write_videofile(os.path.join(basedir, mp4_cr_relpath))

    return mp4_relpath, mp4_cr_relpath, npy_relpath, wav_relpath, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--subclip_duration", default=20, type=int)
    parser.add_argument("--num_threads", default=8, type=int)
    parser.add_argument("--video_fps", default=None, type=int)
    parser.add_argument("--resolution", default=512, type=int)
    args = parser.parse_args()

    # check input csv file exists and read data
    assert os.path.exists(args.csv), \
        f"csv file ({args.csv}) is not found."
    with open(args.csv) as f:
        # assure csv doesn't have names of colums in the first row.
        reader = csv.reader(f)

        csvrows = []
        for row in reader:
            csvrows.append(CsvRow(*row))

    # prepare output directory
    basedir = os.path.dirname(args.csv)
    os.makedirs(os.path.join(basedir, f"subclips_{args.resolution}x{args.resolution}"), exist_ok=True)
    os.makedirs(os.path.join(basedir, f"subclips_{args.resolution}x{args.resolution}/orig_res"), exist_ok=True)

    # main process
    func = partial(get_subclip_frames_audio,
                   basedir=basedir,
                   subclip_duration=args.subclip_duration,
                   fps=args.video_fps,
                   resolution=args.resolution)

    if args.num_threads == 1:
        results = []
        for csvrow in tqdm(csvrows, total=len(csvrows)):
            results.append(func(csvrow))

    else:
        with Pool(args.num_threads) as pool:
            imap = pool.imap(func, csvrows)
            results = list(tqdm(imap, total=len(csvrows)))

    with open(os.path.join(basedir, "frame_audio_list.txt"), "w") as f:
        for result in results:
            mp4_relpath, mp4_cr_relpath, npy_relpath, wav_relpath, csvrow = result
            f.write(",".join([mp4_cr_relpath,
                              npy_relpath,
                              wav_relpath,
                              csvrow.label,
                              csvrow.url,
                              csvrow.start_time,
                              str(int(csvrow.start_time) + args.subclip_duration)]) + "\n")

if __name__ == "__main__":
    main()