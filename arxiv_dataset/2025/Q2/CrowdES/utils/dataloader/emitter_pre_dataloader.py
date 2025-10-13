import torch
import numpy as np

from utils.dataloader.base_dataloader import BaseDataset


class EmitterPreDataset(BaseDataset):
    r"""Dataloader for crowd emitter spatial map prediction"""

    def __init__(self, config, phase):
        super().__init__(config, phase)
        
        self.load_scene_image()
        self.load_scene_segmentation()
        self.load_scene_size()
        self.load_appearance_map()
        self.load_population_map()
        self.load_flow_map()

        # Resize scene image
        input_data = []
        output_data = []
        output_unit_data = []

        image_size = tuple(config.crowd_emitter.emitter_pre.image_size)
        max_population = config.crowd_emitter.emitter_pre.max_population

        for scene in self.scene_list:
            # Prepare input and output data
            input_types = config.crowd_emitter.emitter_pre.input_types
            input = [getattr(self, it)[scene] for it in input_types]
            input = [(i[:, :, None] if i.ndim == 2 else i) for i in input]
            input = np.concatenate(input, axis=2)
            input_dim = input.shape[2]
            output_types = config.crowd_emitter.emitter_pre.output_types
            output = [getattr(self, it)[scene] for it in output_types]
            output = [(o[:, :, None] if o.ndim == 2 else o) for o in output]
            output = np.concatenate(output, axis=2)
            output_dim = output.shape[2]

            # Transform to torch tensors and resize
            input = torch.FloatTensor(input).permute(2, 0, 1)
            output = torch.FloatTensor(output).permute(2, 0, 1)
            input = torch.nn.functional.interpolate(input.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
            output = torch.nn.functional.interpolate(output.unsqueeze(0), size=image_size, mode='nearest').squeeze(0)
            input_data.append(input)
            output_data.append(output)

            # Population probability estimation
            population_prob_dict = self.scene_size[scene]['population_probability']
            population_prob = torch.zeros(max_population, dtype=torch.float32)
            for k, v in population_prob_dict.items():
                k, v = int(k), float(v)
                if k < max_population:
                    population_prob[k] = v
                else:
                    print(f'Warning: Population {k} is larger than max_population {max_population}.',
                          'It will be ignored. If you want to use it, please increase max_population.')   
            population_prob = population_prob / population_prob.max()  # Normalize
            output_unit_data.append(population_prob)
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_data = input_data
        self.output_data = output_data
        self.output_unit_data = output_unit_data

    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, index):
        out = {'input_data': self.input_data[index].detach(),
               'output_data': self.output_data[index].detach(),
               'output_unit_data': self.output_unit_data[index].detach()}
        
        return out


if __name__ == '__main__':
    from utils.config import get_config

    config = get_config('./configs/model/CrowdES_hotel.yaml')
    dataloader = EmitterPreDataset(config, 'emission', 'train')
    dataloader.__len__()
    dataloader.__getitem__(0)
