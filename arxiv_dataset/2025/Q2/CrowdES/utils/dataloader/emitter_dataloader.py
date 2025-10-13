import os
import numpy as np
import torch
import pickle
from tqdm import tqdm

from utils.dataloader.base_dataloader import BaseDataset


class EmitterDataset(BaseDataset):
    r"""Dataloader for crowd emitter diffusion model"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        # Load from cache if exists
        cache_path = os.path.join(config.dataset.dataset_preprocessed_path, 'cache', f'{phase}_emitter.pickle')
        if os.path.exists(cache_path):
            # load all self variables from cache
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                for key, value in cache.items():
                    setattr(self, key, value)
            return

        pbar = tqdm(range(len(self.scene_list) * 2 + 2))
        pbar.set_description(f'Emitter Dataset {phase}: Load data')
        self.load_scene_image()
        self.load_scene_segmentation()
        self.load_appearance_map()
        self.load_population_map()
        self.load_flow_map()
        self.load_scene_size()
        self.load_origin_goal()
        self.load_trajectory()
        pbar.update()

        # Resize scene image guidance
        pbar.set_description(f'Emitter Dataset {phase}: Prepare input data')
        image_size = tuple(config.crowd_emitter.emitter.image_size)
        input_types = config.crowd_emitter.emitter.input_types

        self.input_data = {}
        for scene in self.scene_list:
            input = [getattr(self, it)[scene] for it in input_types]
            input = [(i[:, :, None] if i.ndim == 2 else i) for i in input]
            input = np.concatenate(input, axis=2)
            input = torch.FloatTensor(input).permute(2, 0, 1)
            input = torch.nn.functional.interpolate(input.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)  # shape=(C, H, W)
            input_dim = input.shape[0]
            self.input_data[scene] = input
            pbar.update()
        self.image_size = image_size
        self.input_dim = input_dim + 1  # Add agent distribution channel
        self.output_dim = 7  # agent_type, frame_origin, preferred_speed, origin_x, origin_y, goal_x, goal_y

        # Prepare output data
        pbar.set_description(f'Emitter Dataset {phase}: Prepare output data')
        window_size, window_stride = config.crowd_emitter.emitter.window_size, config.crowd_emitter.emitter.window_stride
        max_num_agents_pad = config.crowd_emitter.emitter.max_num_agents_pad

        self.window_size, self.window_stride = window_size, window_stride
        self.max_num_agents_pad = max_num_agents_pad

        self.seq_start_end = {}
        self.frame_start_end = {}
        self.num_seq_scene = {}
        self.agent_dist_scene = {}

        for scene in self.scene_list:
            origin_goal = self.origin_goal[scene]
            # Normalize xy size
            width, height = self.scene_size[scene]['width'], self.scene_size[scene]['height']
            length, fps, sim_fps = self.scene_size[scene]['length'], self.scene_size[scene]['fps'], self.scene_size[scene]['sim_fps']
            window_size_frame = int(round(self.window_size * fps))
            window_stride_frame = int(round(self.window_stride * fps))

            origin_goal['origin_x'] = origin_goal['origin_x'] / (width - 1)
            origin_goal['origin_y'] = origin_goal['origin_y'] / (height - 1)
            origin_goal['goal_x'] = origin_goal['goal_x'] / (width - 1)
            origin_goal['goal_y'] = origin_goal['goal_y'] / (height - 1)

            # Sort by frame_origin
            origin_goal = origin_goal[['agent_type', 'frame_origin', 'preferred_speed', 'origin_x', 'origin_y', 'goal_x', 'goal_y']]
            origin_goal = origin_goal.sort_values(by='frame_origin')
            origin_goal = origin_goal.reset_index(drop=True).values

            # Create frame range with window_size and stride
            frame_start = np.arange(window_stride_frame, length - window_stride_frame, window_stride_frame)
            frame_start_end = np.stack([frame_start, frame_start + window_size_frame], axis=1)
            frame_origin = origin_goal[:, 1]

            # Find the first and last index where frame_origin within frame_start and frame_end
            seq_start_end = np.searchsorted(frame_origin, frame_start_end, side='left')

            # Filter out where the start_idx and end_idx are the same
            mask = seq_start_end[:, 0] != seq_start_end[:, 1]
            num_seq_scene = mask.sum()
            self.num_seq_scene[scene] = num_seq_scene
            self.seq_start_end[scene] = seq_start_end[mask]
            self.frame_start_end[scene] = torch.LongTensor(frame_start_end[mask])  # shape=['N', 'T']
            self.origin_goal[scene] = torch.FloatTensor(origin_goal)  # shape=['N', 'C']

            # Find the all agent's coordinates just before the frame_start
            frame_before = frame_start_end[mask, 0] - sim_fps
            trajectory = self.trajectory[scene][['frame', 'x', 'y']]
            trajectory.loc[:, 'x'] = (trajectory['x'] / (width - 1) * (image_size[1] - 1)).round()
            trajectory.loc[:, 'y'] = (trajectory['y'] / (height - 1) * (image_size[0] - 1)).round()
            trajectory = trajectory.astype({'x': 'int', 'y': 'int'})
            trajectory = trajectory.loc[trajectory['frame'].isin(frame_before)]
            coordinates = trajectory.groupby('frame').apply(lambda x: x[['x', 'y']].values)
            agent_dist = []
            for frame in frame_before:
                if frame in coordinates.index:
                    agent_dist.append(coordinates.loc[frame])
                else:
                    agent_dist.append(np.zeros((0, 2)))
            self.agent_dist_scene[scene] = agent_dist
            pbar.update()

        pbar.set_description(f'Emitter Dataset {phase}: Gather data pointers')
        self.idx2data = []
        for scene in self.scene_list:
            self.idx2data.extend([(scene, idx) for idx in range(self.num_seq_scene[scene])])
        pbar.update()
        pbar.set_description(f'Emitter Dataset {phase}: {len(self.scene_list)} Scenes, {len(self.idx2data)} Items')
        pbar.total = pbar.n = len(self.scene_list)
        pbar.close()

        # Caching the data for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate_agent_distribution(self, scene, idx):
        agent_dist_frame = self.agent_dist_scene[scene][idx]
        agent_dist = torch.zeros(self.image_size)
        agent_dist[agent_dist_frame[:, 1], agent_dist_frame[:, 0]] = 1
        return agent_dist
    
    def normalize_padding_origin_goal(self, scene, idx):
        seq_start, seq_end = self.seq_start_end[scene][idx]
        frame_start, frame_end = self.frame_start_end[scene][idx]     
        origin_goal = self.origin_goal[scene][seq_start:seq_end].clone()
        origin_goal[:, 1] = (origin_goal[:, 1] - frame_start) / (frame_end - frame_start)
        num_agents = seq_end - seq_start
        if num_agents > self.max_num_agents_pad:
            print(f'Warning: Number of agents {num_agents} in scene {scene} is larger than max_num_agents_pad {self.max_num_agents_pad}.',
                  'It will be truncated. If you want to keep all agents, please increase max_num_agents_pad.')
        num_agents = min(num_agents, self.max_num_agents_pad)
        origin_goal_pad = torch.zeros(self.max_num_agents_pad, self.output_dim)
        origin_goal_pad[:num_agents] = origin_goal[:num_agents]
        origin_goal_mask = torch.zeros(self.max_num_agents_pad, dtype=torch.bool)  # int? float?
        origin_goal_mask[:num_agents] = 1
        return origin_goal_pad, origin_goal_mask

    def __len__(self):
        return len(self.idx2data)
    
    def __getitem__(self, index):
        scene, idx = self.idx2data[index]

        # Add previous agent distribution channel to input_data
        input_data = self.input_data[scene]
        previous_agent_dist = self.generate_agent_distribution(scene, idx)
        input_data = torch.cat([input_data, previous_agent_dist.unsqueeze(dim=0)], dim=0)

        # Normalize the frame_origin and pad the output_data
        output_data_pad, output_data_mask = self.normalize_padding_origin_goal(scene, idx)

        out = {'input_data': input_data.detach(),
               'output_data': output_data_pad.detach(),
               'output_mask': output_data_mask.detach()}
        
        return out
    

if __name__ == '__main__':
    from utils.config import get_config

    config = get_config('./configs/model/CrowdES_hotel.yaml')
    dataloader = EmitterDataset(config, 'emission', 'train')
    dataloader.__len__()
    dataloader.__getitem__(0)
