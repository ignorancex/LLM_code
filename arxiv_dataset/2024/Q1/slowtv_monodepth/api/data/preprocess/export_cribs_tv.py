import os
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import src.devkits.cribs_tv as ctv
from src.paths import DATA_PATHS as PATHS
from src.utils import io


def _load_img(file):
    return Image.open(file).reduce(5)


def split_scenes(overwrite: bool = False) -> None:
    """Split each video into individual scene cuts. Requires http://www.scenedetect.com/en/latest/"""
    root = PATHS['cribs_tv']
    vid_files = ctv.get_vid_files()

    for f in tqdm(vid_files):
        if not overwrite and io.has_contents(root/f.stem):
            print(f'-> Skipping scene splitting for {f.stem}...')
            continue

        subprocess.call([
            'scenedetect', '-i', f, '-m', '1s', '--drop-short-scenes',
            'detect-content', 'split-video', '-o', f'{root}/{f.stem}', '-f', '$SCENE_NUMBER'
        ])


def export_videos(fps: Union[str, list[str]] = '1/1', overwrite: bool = False) -> None:
    """Split each video and each scene into individual frames."""
    root = PATHS['cribs_tv']
    seqs = ctv.get_seqs()

    if isinstance(fps, str): fps = [fps]*len(seqs)
    else: assert isinstance(fps, list) and len(fps) == len(seqs)

    for seq, f in tqdm(zip(seqs, fps)):
        vids = io.get_files(root/seq, key=lambda f: f.suffix == '.mp4')

        for vid in tqdm(vids):
            dst = vid.parent/f'{int(vid.stem)-1:010}'
            io.mkdirs(dst)

            if not overwrite and io.has_contents(dst):
                print(f'-> Skipping video "{vid}"...')
                continue

            subprocess.call(['ffmpeg', '-i', vid, '-r', f, dst/'%010d.png', '-hide_banner',  '-loglevel', 'error'])


def create_thumbnails(overwrite=False):
    """Create thumbnail for each video, showing the first frame from each scene."""
    root = PATHS['cribs_tv']
    dst_dir = PATHS['cribs_tv']/'thumbnails'
    io.mkdirs(dst_dir)

    seqs = ctv.get_seqs()

    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=24)

    for seq in tqdm(seqs):
        target = dst_dir/f'{seq}.png'
        if not overwrite and target.is_file():
            print(f'-> Skipping thumbnail creation for "{seq}"...')
            continue

        files = sorted((root/seq).glob(f'./*/{1:010}.png'))
        with Pool() as p: imgs = list(p.imap(_load_img, tqdm(files)))

        n = 5
        pad = n - (len(imgs) % n)
        if pad != n: imgs += [Image.new('RGB', imgs[0].size) for _ in range(pad)]

        [ImageDraw.Draw(img).text((0, 0), str(int(f.parent.stem)), fill=(255, 255, 255), font=font)
         for img, f in zip(imgs, files)]

        # Arrange as a grid of images, with `n` columns
        imgs = [imgs[i:i+n] for i in range(0, len(imgs), n)]
        img = np.concatenate([np.concatenate([np.array(i) for i in img], axis=1) for img in imgs], axis=0)
        Image.fromarray(img).save(target)


def convert_blacklist():
    """Utility to convert a range of scenes into a list of individual scenes."""
    file = ctv.get_blacklist_file()
    lines = io.readlines(file, split=True)

    for i, line in enumerate(lines):
        new_line = [line[0]]
        for item in line[1:]:
            if '-' not in item:
                new_line.append(f'{int(item):010}')
            else:
                start, end = item.split('-')
                new_line += [f'{i:010}' for i in range(int(start), int(end)+1)]

        lines[i] = new_line

    lines = [' '.join(line)+'\n' for line in lines]
    with open(file, 'w') as f: f.writelines(lines)


def create_splits(n_train=0.95):
    """Create list of train/val files for all videos/scenes."""
    root = PATHS['cribs_tv']
    train_file = ctv.Item.get_split_file('train')
    val_file = ctv.Item.get_split_file('val')

    train_file.unlink(missing_ok=True), val_file.unlink(missing_ok=True)

    seqs = ctv.get_seqs()

    blacklist = ctv.load_blacklist()

    for seq in tqdm(seqs):
        scenes = io.get_dirs(root/seq)
        print(f'-> Found {len(scenes)} scenes in "{seq}"')
        scenes = [s for s in scenes if s.stem not in blacklist.get(seq, [])]
        print(f'-> Reduced to {len(scenes)}')

        split = int(len(scenes)*n_train)

        train_scenes, val_scenes = scenes[:split], scenes[split:]
        print(f'-> Adding {len(train_scenes)} train scenes and {len(val_scenes)} val scenes from "{seq}"')

        n_train = 0
        for scene in tqdm(train_scenes):
            files = io.get_files(scene)
            n_train += len(files)
            lines = [f'{seq} {scene.stem} {f.stem}\n' for f in files]
            with open(train_file, 'a') as f: f.writelines(lines)
        print(f'\t-> Added {n_train} train images from "{seq}"')

        n_val = 0
        for scene in tqdm(val_scenes):
            files = io.get_files(scene)
            n_val += len(files)
            lines = [f'{seq} {scene.stem} {f.stem}\n' for f in files]
            with open(val_file, 'a') as f: f.writelines(lines)
        print(f'\t-> Added {n_val} val images from "{seq}"')


def main(fps, n_train, overwrite=False):
    split_scenes(overwrite)
    export_videos(fps=fps, overwrite=overwrite)
    create_thumbnails(overwrite=False)
    create_splits(n_train=n_train)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fps', type=str, default='10', help='FPS to extract videos at.')
    parser.add_argument('--n-train', type=float, default=0.95, help='Fraction of images used for training.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing files.')
    args = parser.parse_args()

    main(args.fps, args.n_train, args.overwrite)
