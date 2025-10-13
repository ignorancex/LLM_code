import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# class SEVIRDataset(Dataset):
#     def __init__(self, h5_file_paths,mean_std_path='sevir_data/mean_std.npy'):
#         self.data = []
#         self.sample_indices = []
#         total_samples = 0

#         for file_path in h5_file_paths:
#             with h5py.File(file_path, 'r') as h5_file:
#                 dataset = h5_file['ir069'][:]  # 假设数据存储在 'ir069' 数据集中
#                 num_events, height, width, num_timesteps = dataset.shape
#                 self.data.append(dataset)
#                 # 每个事件有 num_timesteps - 1 个样本
#                 num_samples = num_events * (num_timesteps - 1)
#                 self.sample_indices.extend([(len(self.data) - 1, i) for i in range(num_samples)])
#                 total_samples += num_samples
#         self.mean, self.std = self.compute_or_load_mean_std(mean_std_path)
#         print('mean:',self.mean)
#         print('std:',self.std)


#     def compute_or_load_mean_std(self, mean_std_path):
#         """
#         计算或加载数据集的均值和标准差，忽略 NaN 值
#         """
#         if os.path.exists(mean_std_path):
#             # 如果均值和标准差已存在，则加载
#             mean_std = np.load(mean_std_path, allow_pickle=True).item()
#             mean = mean_std['mean']
#             std = mean_std['std']
#         else:
#             # 否则，计算均值和标准差
#             all_data = np.concatenate([dataset.flatten() for dataset in self.data])
#             mean = np.nanmean(all_data)
#             std = np.nanstd(all_data)
#             # 保存计算结果
#             np.save(mean_std_path, {'mean': mean, 'std': std})
    
#         return mean, std

#     def __len__(self):
#         return len(self.sample_indices)

#     def __getitem__(self, idx):
#         data_idx, sample_idx = self.sample_indices[idx]
#         dataset = self.data[data_idx]
#         num_timesteps = dataset.shape[-1]
#         event_idx = sample_idx // (num_timesteps - 1)
#         time_idx = sample_idx % (num_timesteps - 1)
#         input_image = dataset[event_idx, :, :, time_idx]
#         target_image = dataset[event_idx, :, :, time_idx + 1]


#         # 标准化处理
#         input_image = (input_image - self.mean) / self.std
#         target_image = (target_image - self.mean) / self.std
        
#         # 将数据转换为 PyTorch 张量，并添加通道维度 和 时间维度
#         input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         target_tensor = torch.tensor(target_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         return input_tensor, target_tensor



class SEVIRDataset(Dataset):
    def __init__(self, h5_file_paths, input_steps=10, output_steps=10, mean_std_path='sevir_data/mean_std.npy'):
        self.data = []
        self.sample_indices = []
        self.input_steps = input_steps
        self.output_steps = output_steps

        for file_path in h5_file_paths:
            with h5py.File(file_path, 'r') as h5_file:
                dataset = h5_file['ir069'][:]  # 假设数据存储在 'ir069' 数据集中
                num_events, height, width, num_timesteps = dataset.shape
                self.data.append(dataset)
                # 每个事件有 num_timesteps - (input_steps + output_steps - 1) 个样本
                num_samples = num_events * (num_timesteps - (input_steps + output_steps - 1))
                self.sample_indices.extend([(len(self.data) - 1, i) for i in range(num_samples)])

        self.mean, self.std = self.compute_or_load_mean_std(mean_std_path)
        print('mean:', self.mean)
        print('std:', self.std)

    def compute_or_load_mean_std(self, mean_std_path):
        """
        计算或加载数据集的均值和标准差，忽略 NaN 值
        """
        if os.path.exists(mean_std_path):
            # 如果均值和标准差已存在，则加载
            mean_std = np.load(mean_std_path, allow_pickle=True).item()
            mean = mean_std['mean']
            std = mean_std['std']
        else:
            # 否则，计算均值和标准差
            all_data = np.concatenate([dataset.flatten() for dataset in self.data])
            mean = np.nanmean(all_data)
            std = np.nanstd(all_data)
            # 保存计算结果
            np.save(mean_std_path, {'mean': mean, 'std': std})
    
        return mean, std

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        data_idx, sample_idx = self.sample_indices[idx]
        dataset = self.data[data_idx]
        num_timesteps = dataset.shape[-1]
        event_idx = sample_idx // (num_timesteps - (self.input_steps + self.output_steps - 1))
        time_idx = sample_idx % (num_timesteps - (self.input_steps + self.output_steps - 1))
        
        # 获取输入和输出序列
        input_sequence = dataset[event_idx, :, :, time_idx:time_idx + self.input_steps]
        target_sequence = dataset[event_idx, :, :, time_idx + self.input_steps:time_idx + self.input_steps + self.output_steps]

        # 标准化处理
        input_sequence = (input_sequence - self.mean) / self.std
        target_sequence = (target_sequence - self.mean) / self.std
        
        # 将数据转换为 PyTorch 张量，并添加通道维度
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).permute(3, 0, 1, 2) 
        target_tensor = torch.tensor(target_sequence, dtype=torch.float32).unsqueeze(0).permute(3, 0, 1, 2) 
        
        return input_tensor, target_tensor



class TestDataset(SEVIRDataset):
    def __init__(self, args):
        super(TestDataset, self).__init__(args)

    def __getitem__(self, idx):
        data_idx, sample_idx = self.sample_indices[idx]
        dataset = self.data[data_idx]
        num_timesteps = dataset.shape[-1]
        event_idx = sample_idx // (num_timesteps - 1)
        # time_idx = sample_idx % (num_timesteps - 1)
        input_image = dataset[event_idx, :, :, 0]
        target_image = dataset[event_idx, :, :, 1:41]


        # 标准化处理
        input_image = (input_image - self.mean) / self.std
        target_image = (target_image - self.mean) / self.std

        # print(target_image.shape)
        # 将数据转换为 PyTorch 张量，并添加通道维度 和 时间维度
        input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.tensor(target_image, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        return input_tensor, target_tensor


# h5_file_paths = [
#     'sevir_data/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5',
#     ]
# sevir_dataset = SEVIRDataset(h5_file_paths)
# dataloader = DataLoader(sevir_dataset, batch_size=32, shuffle=True, num_workers=4)

# for inputs, targets in dataloader:
#     print("Validation:", inputs.shape, targets.shape)
#     print(inputs)
#     break  

# # 创建数据集和数据加载器
# h5_file_paths = [
#     'sevir_data/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5',
#     'sevir_data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5',
#     'sevir_data/SEVIR_IR069_STORMEVENTS_2018_0701_1231.h5',
#     'sevir_data/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5',
#     ]
# sevir_dataset = SEVIRDataset(h5_file_paths)
# dataloader = DataLoader(sevir_dataset, batch_size=32, shuffle=True, num_workers=4)