import os
import json

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MetaData():
    data_id: int
    youtube_id: str
    start_time: float
    caption: str
    end_time: float = None


def create_metadata_list_from_dataset_name(dataset: str, dataset_type: str):
    """
    Create a list of MetaData for downloading mp4 from youtube.

    Args:
        dataset (str): The name of dataset.
        dataset_type (str): E.g. train, test, val, etc.

    """
    dataset = dataset.lower()

    factory = {
        "audiocaps": _audiocaps_metadata_list,
        "vggsound": _vggsound_metadata_list,
        "audioset": _audioset_metadata_list
    }

    if dataset not in factory:
        raise ValueError(f"'{dataset}' is not supported. The dataset name must be one of {factory.keys()}.")

    return factory[dataset](dataset_type)


# metadata factory for the each dataset

# AudioCaps
def _audiocaps_metadata_list(dataset_type):
    from datasets import load_dataset

    supported_types = ["train", "test", "validation"]
    assert dataset_type in supported_types, \
        f"dataset_type must be one of {supported_types}, but '{dataset_type}' is given."

    ds = load_dataset("d0rj/audiocaps")

    metadata_list = [
        MetaData(data_id=idx,
                 youtube_id=elm["youtube_id"],
                 start_time=elm["start_time"],
                 caption=elm["caption"].replace(",", "_")
                 )
        for idx, elm in enumerate(ds[dataset_type])
    ]
    
    return metadata_list

# VGGSound
def _vggsound_metadata_list(dataset_type):
    supported_types = ["train", "test"]
    assert dataset_type in supported_types, \
        f"dataset_type must be one of {supported_types}, but '{dataset_type}' is given."

    csv_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                            f"vggsound/{dataset_type}.csv")
    
    if not os.path.exists(csv_path):
        from urllib.request import urlretrieve
        urlretrieve(f"https://github.com/hche11/VGGSound/raw/master/data/{dataset_type}.csv", csv_path)
    
    with open(csv_path, "r") as f:
        lines = f.readlines()
    
    metadata_list = []

    for idx, line in enumerate(lines):
        youtube_id, caption = line.strip().split(",", maxsplit=1)
        metadata = MetaData(data_id=idx,
                            youtube_id=youtube_id,
                            start_time=0,
                            caption=caption.replace(",", "_"))

        metadata_list.append(metadata)
    
    return metadata_list

# AudioSet
@dataclass
class _AudioSetOntologyElem(object):
    id: str
    name: str
    children: List[str]


def _load_audioset_ontology(path) -> Dict[str, _AudioSetOntologyElem]:
    with open(path, "r") as f:
        raw = json.load(f)
    
    ontology = {}

    for data in raw:
        ontology[data["id"]] = _AudioSetOntologyElem(
            id = data["id"],
            name = data["name"],
            children = data["child_ids"],
        )
    
    return ontology

def _get_all_child_labels_from_ontology(ontology, 
                                        root_id):
    visit = set()
    def recursive(cur: _AudioSetOntologyElem):
        if len(cur.children) == 0:
            return [(cur.name, cur.id)]
        
        ret = []
        for child_id in cur.children:
            if child_id in visit:
                continue

            visit.add(child_id)
            child = ontology[child_id]
            ret += recursive(child)
        
        ret.append((cur.name, cur.id))

        return ret

    labels = recursive(ontology[root_id])
    return labels

def _audioset_metadata_list(dataset_type,
                            num_music_video=80000,
                            ignore_speech=True):
    # if dataset_type is train, both balanced_train and unbalanced_train will be downloaded.
    supported_types = ["train", "eval"]
    assert dataset_type in supported_types, \
        f"dataset_type must be one of {supported_types}, but '{dataset_type}' is given."
    
    # 1: Load all meta data from csv files
    metadata_list: List[MetaData] = []
    for dstype in ["balanced_train", "unbalanced_train"] if dataset_type == "train" else ["eval"]:
        csv_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                f"audioset/{dstype}.csv")
        
        if not os.path.exists(csv_path):
            from urllib.request import urlretrieve
            urlretrieve(f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{dstype}_segments.csv", csv_path)

        with open(csv_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#"):
                continue
            
            ytid, start, end, labels = line.strip().split(", ", maxsplit=3)
            labels = labels[1:-1]

            meta = MetaData(data_id=len(metadata_list),
                            youtube_id=ytid,
                            start_time=float(start),
                            end_time=float(end),
                            caption=labels)
            metadata_list.append(meta)

    # 2: Load ontology
    ontology_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
                                 "audioset/ontology.json")
    if not os.path.exists(ontology_path):
        from urllib.request import urlretrieve
        urlretrieve("https://raw.githubusercontent.com/audioset/ontology/master/ontology.json", ontology_path)
    
    ontology: Dict[str, _AudioSetOntologyElem] = _load_audioset_ontology(ontology_path)
    name2id = {v.name: k for k, v in ontology.items()}

    speech_id = name2id["Speech"]
    music_labels = [x[1] for x in _get_all_child_labels_from_ontology(ontology, name2id["Music"])]

    # 3: filter audiosets for V2A gen.
    others = []
    music_videos: List[MetaData] = []
    for meta in metadata_list:
        is_speech = False
        is_music = False

        labels = meta.caption.split(",")
        for label in labels:
            if label == speech_id:
                is_speech = True
            
            if label in music_labels:
                is_music = True

        # 3.1: ignore speech video
        if is_speech:
            continue

        if is_music:
            music_videos.append(meta)
        else:
            others.append(meta)

    # 3.2: Randomly pick out music videos up to "num_music_video"
    
    # Decide # of videos to pick out for each label based on historgram
    from numpy.random import RandomState
    rs = RandomState(seed=83)

    from collections import defaultdict
    label2count = defaultdict(int)

    # Count all videos per label
    for meta in music_videos:
        labels = meta.caption.split(",")
        for label in labels:
            if label in music_labels:
                label2count[label] += 1

    # Assign each video to the label that has fewest videos
    label2music_videos = defaultdict(list)
    for meta in music_videos:
        target_label = None
        max_count = 10_000_000_000
        labels = meta.caption.split(",")
        for label in labels:
            if label not in music_labels:
                continue
            
            if label2count[label] < max_count:
                target_label = label
                max_count = label2count[label]
        
        assert target_label is not None
        label2music_videos[target_label].append(meta)
    
    # Ascending order
    label_wise_videos = sorted(label2music_videos.values(), key=lambda v: len(v))

    used_music_videos = []
    num_labels = len(label_wise_videos)
    num_each_videos = None

    for i, vs in enumerate(label_wise_videos):
        num_cur_videos = len(vs)
        cur_total = len(used_music_videos)

        if num_each_videos is not None:
            print(f"select {num_each_videos} videos from {i}")
            # randomly pick out videos up to num_each_videos
            rs.shuffle(vs)
            used_music_videos += vs[:num_each_videos]

            continue

        if cur_total + num_cur_videos * (num_labels - i) >= num_music_video:
            # find the mimimul integer meets the cond. above
            divisor = num_labels - i
            num_each_videos = (num_music_video - cur_total + divisor - 1) // divisor

            rs.shuffle(vs)
            used_music_videos += vs[:num_each_videos]
            print(f"[Filtering music videos] select {num_each_videos} videos from {i}")
        else:
            # use all videos for this label
            used_music_videos += vs
            print(f"[Filtering music videos] select {len(vs)} videos from {i}")

    # prepare return
    filtered = others + used_music_videos
    for i in range(len(filtered)):
        filtered[i].data_id = i
        filtered[i].caption = "_".join([ontology[x].name.strip() for x in filtered[i].caption.split(",")])
    
    print(f"{len(filtered)} videos in total will be prepared.")
    
    return filtered
