import argparse
import csv
import os
import re

from functools import partial
from multiprocessing import Pool
from pytube import YouTube
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class MetaData():
    label: str
    url: str
    start_time: str
    id: int

def youtube_download(meta: MetaData, output_dir: str):
    try:
        yt = YouTube(meta.url)

        name = f"{meta.id:06}"
        # (2023/06/27) this doesn't work because some titles contain non-English words.
        # name = "_".join(re.sub(r"[^\w_\s]+", "", yt.title).split())

        if not output_dir.endswith(("mp4", "mp4/")):
            mp4_dir = os.path.join(output_dir, "mp4")

        yt.streams.filter(progressive=True, file_extension="mp4")\
            .order_by("resolution").desc().first()\
            .download(mp4_dir, filename=name+".mp4")
        
        output_mp4_path = os.path.join(mp4_dir, name + ".mp4")
        return (os.path.relpath(output_mp4_path, output_dir), meta)

    except:
        return ("ERROR", meta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--num_threads", type=int, default=8)
    args = parser.parse_args()

    assert os.path.exists(args.input_csv), \
        f"csv file ({args.input_csv}) is not found."
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # read csv to get all url list
    with open(args.input_csv) as f:
        reader = csv.reader(f)
        next(reader, None) # first row is the names of each column

        urls = set()
        metadata = []
        for row in reader:
            if row[1] in urls:
                print(row, "is deplicated. skip")
            else:
                metadata.append(MetaData(row[0], row[1], row[2], len(metadata)))
            
            urls.add(row[1])

    with Pool(args.num_threads) as pool:
        imap = pool.imap(partial(youtube_download, output_dir=args.output_dir), metadata)
        results = list(tqdm(imap, total=len(metadata)))

    # log for failed failes
    with open(os.path.join(args.output_dir, "failed.txt"), "w") as f:
        for result in results:
            if result[0] == "ERROR":
                meta = result[1]
                f.write(f"{meta.url}\n")

    # list up all videos successfully downloaded
    with open(os.path.join(args.output_dir, "datalist.txt"), "w") as f:
        for result in results:
            if result[0] == "ERROR":
                continue
            relpath = result[0]
            meta = result[1]
            f.write(",".join([relpath, meta.label, meta.url, meta.start_time]) + "\n")

if __name__ == "__main__":
    main()