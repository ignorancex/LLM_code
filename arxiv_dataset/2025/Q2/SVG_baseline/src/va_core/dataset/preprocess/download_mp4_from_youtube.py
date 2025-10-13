import logging
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace
from typing import List

import click
import moviepy.video.fx.all as vfx
from download_utils import MetaData, create_metadata_list_from_dataset_name
from moviepy.editor import VideoClip, VideoFileClip
from tqdm import tqdm

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

@dataclass
class SavedInfo(object):
    preprocessed_mp4_path: str
    orig_mp4_path: str
    duration: int

    def exists_all_path_from(self, basepath):
        return os.path.exists(os.path.join(basepath, self.mp4_relpath))

def resize_crop_video(orig_vc: VideoFileClip,
                          resolution: int) -> SavedInfo:
    orig_w, orig_h = orig_vc.size

    if orig_w < resolution or orig_h < resolution:
        return None

    vc: VideoClip
    if orig_w < orig_h:
        vc = vfx.resize(orig_vc, width=resolution)
    else:
        vc = vfx.resize(orig_vc, height=resolution)
    
    resized_w, resized_h = vc.size
    x1 = (resized_w - resolution) // 2
    y1 = (resized_h - resolution) // 2

    vc = vfx.crop(vc, 
                  x1=x1, 
                  x2=x1 + resolution, 
                  y1=y1, 
                  y2=y1 + resolution)

    return vc

def youtube_download(meta: MetaData,
                     backend: str):
    video_url = f"https://www.youtube.com/watch?v={meta.youtube_id}"

    if backend == "yt_dlp":
        ## using yt_dlp
        from yt_dlp import YoutubeDL
        ydl_args = {
            "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]",
            "outtmpl": "%(id)s.%(ext)s"
        }
        with YoutubeDL(params=ydl_args) as ydl:
            ydl.download(video_url)

        saved_path = os.path.join(os.getcwd(), f"{meta.youtube_id}")
        if not saved_path.endswith(".mp4"):
            saved_path += ".mp4"

    elif backend == "pytube":
        # using pytube
        from pytube import YouTube
        yt = YouTube(video_url)
        yt.streams.filter(progressive=True, file_extension="mp4")\
            .order_by("resolution").desc().first()\
            .download(os.getcwd(), filename=f"{meta.youtube_id}.mp4", timeout=10)
        
        saved_path = os.path.join(os.getcwd(), f"{meta.youtube_id}.mp4")
    
    return saved_path

def prepare_video(meta: MetaData,
                  orig_video_dir: str,
                  preprocessed_video_dir: str,
                  video_duration: int,
                  video_resolution: int,
                  video_fps: int,
                  backend="yt_dlp",
                  retry_info=0,
                  ):
    assert backend in ("yt_dlp", "pytube")

    # pack all arguments for retry
    kwargs = {
        "meta": meta,
        "orig_video_dir": orig_video_dir,
        "preprocessed_video_dir": preprocessed_video_dir,
        "video_duration": video_duration,
        "video_resolution": video_resolution,
        "video_fps": video_fps,
        "backend": backend,
    }

    # not sure this is needed.
    meta.youtube_id = meta.youtube_id[:11]

    # setting path for output files
    filename = f"{meta.data_id:06}.mp4"

    # return object
    saved_info = SavedInfo(
        preprocessed_mp4_path=os.path.join(preprocessed_video_dir, filename),
        orig_mp4_path=os.path.join(orig_video_dir, filename),
        duration=-1
    )

    # Dataset preparation will proceedq in the following order.
    #   1. download a raw video (full length & full resolution, will be deleted)
    #   2. clip duration of the raw video and get subclip (desired length & full resolution, will be kept)
    #   3. crop height and width of the subclip and change fps (desired length and res, will be kept)
    # By keeping 2 and 3, we can skip each step if they already exist.

    # early exit if the video is already downloaded and preprocessed
    if os.path.exists(saved_info.preprocessed_mp4_path):
        logging.info(f"Skip processing for {meta.data_id} because it already exists.")
        try:
            vc = VideoFileClip(saved_info.preprocessed_mp4_path)
        except:
            os.remove(saved_info.preprocessed_mp4_path)
            if retry_info & 1:
                logging.info(f"Created preprocessed mp4 twice but got an error. Skip processing for {meta.data_id}.")
                return ("ERROR", meta)

            return prepare_video(retry_info=(retry_info | 1),
                                 **kwargs)
            
        saved_info.duration = vc.duration
        return (saved_info, meta)

    orig_vc: VideoFileClip
    if not os.path.exists(saved_info.orig_mp4_path): 
        # 1. download video from youtube
        try:
            raw_video_path = youtube_download(meta, backend)
        except Exception as e:
            logging.info(f"Download is failed for {meta.data_id}: {e}")
            
            return ("ERROR", meta)

        # 2. create subclip for the target duration
        failed = False
        try:
            raw_vc = VideoFileClip(raw_video_path)

            s = int(meta.start_time)
            e = min(s + video_duration, raw_vc.duration)
            orig_vc: VideoFileClip = raw_vc.subclip(t_start=s, t_end=e)
            
            # save subclip
            # sub clip will be reloaded from the file below to remove original mp4.
            orig_vc.write_videofile(saved_info.orig_mp4_path,
                                    audio=True,
                                    threads=4)
        except Exception as e:
            logging.info(f"Loading raw video is failed for {meta.data_id}: {e}")
            failed = True
            
        # remove raw video
        if os.path.exists(raw_video_path):
            os.remove(raw_video_path)

        if failed:
            return ("ERROR", meta)
    
    # load subclip
    try:
        orig_vc = VideoFileClip(saved_info.orig_mp4_path)
    except:
        os.remove(saved_info.orig_mp4_path)
        if (retry_info >> 1) & 1:
            logging.info(f"Created original mp4 twice but got an error. Skip processing for {meta.data_id}.")
            return ("ERROR", meta)
        
        return prepare_video(retry_info=(retry_info | 2),
                             **kwargs)

    # 3. crop
    resized_subclip_vc: VideoClip = resize_crop_video(orig_vc,
                                                      resolution=video_resolution)

    # if the original video is smaller than target resolution, skip this video and return as an ERROR.
    if resized_subclip_vc is None:
        return ("ERROR", meta)
    
    # save
    resized_subclip_vc.write_videofile(saved_info.preprocessed_mp4_path,
                                       fps=video_fps,
                                       audio=True,
                                       threads=4)

    saved_info.duration = resized_subclip_vc.duration

    return (saved_info, meta)


def get_resume_subsection_index(failed_list_path,
                                data_list_path,
                                data_size,
                                num_subsection):
    if not os.path.exists(failed_list_path) or not os.path.exists(data_list_path):
        return 0
    
    max_index = 0

    # check failed list
    with open(failed_list_path, "r") as f:
        failed_lines = f.readlines()
    
    for line in failed_lines:
        idx = line.strip().split(",")[-1]
        max_index = max(max_index, int(idx))
    
    # check datalist
    with open(data_list_path, "r") as f:
        success_lines = f.readlines()
    
    # assume the first line is header.
    for line in success_lines[1:]:
        idx = line.strip().split(",")[-1]
        max_index = max(max_index, int(idx))
    
    num_ss_data = (data_size + num_subsection - 1) // num_subsection
    resume_start_idx = None
    for ss_idx in range(num_subsection):
        ss_start_id = num_ss_data * ss_idx
        ss_end_id = ss_start_id + num_ss_data - 1
        
        if max_index == ss_end_id:
            resume_start_idx = ss_idx + 1
            break

        if ss_start_id <= max_index < ss_end_id:
            resume_start_idx = ss_idx
            break
    
    assert resume_start_idx is not None

    # remove lines
    failed = ""
    for line in failed_lines:
        idx = int(line.strip().split(",")[-1])
        if idx < num_ss_data * resume_start_idx:
            failed += line
    
    with open(failed_list_path, "w") as f:
        f.write(failed)
    
    # the first line must be header
    success = success_lines[0]
    for line in success_lines[1:]:
        idx = int(line.strip().split(",")[-1])
        if idx < num_ss_data * resume_start_idx:
            success += line

    with open(data_list_path, "w") as f:
        f.write(success)
    
    return resume_start_idx


@click.command()
@click.option("--output_dir", required=True, type=str)
@click.option("--num_threads", type=int, default=20)
@click.option("--video_resolution", type=int, default=512)
@click.option("--video_duration", type=int, default=10)
@click.option("--video_fps", type=int, default=3)
@click.option("--dataset_name", required=True, type=click.Choice(["audiocaps", "vggsound", "audioset"]))
@click.option("--dataset_type",required=True, type=str)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    args.output_dir = os.path.join(args.output_dir, args.dataset_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    from pprint import pprint
    pprint(args)

    metadata: List[MetaData] = create_metadata_list_from_dataset_name(args.dataset_name,
                                                                      args.dataset_type)

    # for resuming
    num_ss = 10
    ss = (len(metadata) + num_ss - 1) // num_ss

    failed_list_path = os.path.join(args.output_dir, "failed.txt")
    data_list_path = os.path.join(args.output_dir, "datalist.txt")

    ss_start = get_resume_subsection_index(failed_list_path, 
                                           data_list_path, 
                                           data_size=len(metadata), 
                                           num_subsection=num_ss)

    # write header to csv
    headers = [
        "mp4_relpath",
        "orig_mp4_relpath",
        "caption",
        "youtube_id",
        "start_time",
        "end_time",
        "data_id"
    ]
    header_write = ",".join(headers)

    if ss_start > 0:
        # read header already written and check they are the same
        with open(data_list_path, "r") as f:
            header_read = f.readline()
        
        assert header_read.strip() == header_write, \
            "header already witten in csv file is not compatible. " \
            f"To be written: {header_write} != Already written: {header_read}."
    elif ss_start == 0:
        # write header
        with open(data_list_path, "w") as f:
            f.write(header_write + "\n")

    # output directories
    res = args.video_resolution
    orig_video_dir = os.path.join(args.output_dir, "orig_mp4")
    preprocessed_video_dir = os.path.join(args.output_dir, 
                                          f"mp4_{res}x{res}_fps_{args.video_fps}")
    os.makedirs(orig_video_dir, exist_ok=True)
    os.makedirs(preprocessed_video_dir, exist_ok=True)

    # prepare dataset
    common_kwargs = {
        "preprocessed_video_dir": preprocessed_video_dir,
        "orig_video_dir": orig_video_dir,
        "video_duration": args.video_duration,
        "video_resolution": args.video_resolution,
        "video_fps": args.video_fps,
    }
    for i in range(ss_start, num_ss):
        logging.info(f"Start Download for {i + 1} / {num_ss}.")
        if args.num_threads == 1:
            results = []
            for meta in tqdm(metadata[i*ss:(i+1)*ss]):
                res = prepare_video(meta,
                                    **common_kwargs)
                results.append(res)
        else:
            with Pool(args.num_threads) as pool:
                imap = pool.imap(partial(prepare_video, 
                                         **common_kwargs),
                                metadata[i*ss:(i+1)*ss])
                results = list(tqdm(imap, total=ss))

        # log for failed failes
        with open(failed_list_path, "w" if i == 0 else "a") as f:
            for result in results:
                if result[0] == "ERROR":
                    meta = result[1]
                    f.write(",".join([str(meta.youtube_id),
                                      str(meta.data_id)]) + "\n")

        # list up all videos successfully downloaded
        with open(data_list_path, "a") as f:
            for result in results:
                if result[0] == "ERROR":
                    continue
                si: SavedInfo = result[0]
                meta: MetaData = result[1]

                contents = [
                    os.path.relpath(si.preprocessed_mp4_path, start=args.output_dir),
                    os.path.relpath(si.orig_mp4_path, start=args.output_dir),
                    meta.caption,
                    meta.youtube_id,
                    str(meta.start_time),
                    str(meta.start_time + si.duration),
                    str(meta.data_id)
                ]

                assert len(headers) == len(contents), \
                    f"The # of entries in header and content must be the same. ({len(headers)} != {len(contents)})"
                
                f.write(",".join(contents) + "\n")

if __name__ == "__main__":
    main()