from collections.abc import Sequence
from absl import app
import numpy as np
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer 
import os
from pathlib import Path
import cv2

dataset_dir = 'data/handle/train_handle'

def batch_process(batch):
    processed_data = []
    for img in batch:
        processed_img = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        processed_data.append(processed_img)
    return np.array(processed_data)

def load_data(data_type, dataset_dir):
    data_by_episode = {}
    data_path = Path(dataset_dir)
    run_folders = sorted(data_path.glob('run_*'))
    
    for run_folder in run_folders:
        episode_id = int(run_folder.name.split('_')[1])
        image_folder = run_folder / 'images_linear'
        image_files = sorted(image_folder.glob('*.jpg'))
        images = batch_process(image_files)

        if episode_id not in data_by_episode:
            data_by_episode[episode_id] = []
        data_by_episode[episode_id].append(images)
    
    return data_by_episode

def add_episode_to_buffer(buffer, episode_data):
    episode_length = len(episode_data['image'])
    if episode_length == 0:
        return  # No data to add

    episode_data = {key: np.array(value) for key, value in episode_data.items()}
    buffer.add_episode(episode_data, compressors="disk")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    output_dir = "data/handle_reply_buffer"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    zarr_path = str(output_dir.joinpath("replay_buffer.zarr").absolute())
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")
    image_data = load_data('color', dataset_dir)

    episode_ids = set(image_data) 

    for episode_id in episode_ids:
        print((image_data.get(episode_id,[]))[0].shape)
        episode_data = {
            'image': image_data.get(episode_id, [])[0],
        }
        add_episode_to_buffer(replay_buffer, episode_data)

if __name__ == '__main__':
    app.run(main)
