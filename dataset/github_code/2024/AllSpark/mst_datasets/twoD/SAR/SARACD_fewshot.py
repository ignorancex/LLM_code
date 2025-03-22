import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import BatchSampler
from collections import defaultdict

from utils import cls_accuracy
from utils.metrics import Avg_values


class SARACDFewShotDataset(Dataset):
    def __init__(self, root_path, n_support=25):
        self.root_path = root_path
        self.n_support = n_support

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.cls_list = ["A220", "A320321", "A330", "ARJ21", "Boeing737", "Boeing787"]

        self.data = pd.DataFrame(columns=["image_path", "label_id", "label_name"])
        for idx, cls_name in enumerate(self.cls_list):
            cls_path = os.path.join(self.root_path, cls_name)
            image_list = os.listdir(cls_path)
            assert len(image_list) != 0
            for image in image_list:
                self.data.loc[len(self.data.index)] = [os.path.join(cls_path, image), 
                                                       idx, cls_name]

        self.prompts = [
            "Based on the SAR imagery feature description, please classify this object.",
            "Given the following SAR imagery characteristics, please output the most fitting scene label."
        ]

        self.results = dict()

        # Dictionary mapping class labels to sample indices.
        self.class_to_indices = defaultdict(list)
        for idx in range(len(self.data)):
            label = self.data.iloc[idx]['label_id']
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indices):

        support_indices = indices[:self.n_support]
        query_indices = indices[self.n_support:]

        support_images = []
        support_labels = []
        for idx in support_indices:
            img_path = self.data.iloc[idx, 0]
            image = Image.open(img_path).convert('RGB')
            label = self.data.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            support_images.append(image)
            support_labels.append(label)

        query_images = []
        query_labels = []
        for idx in query_indices:
            img_path = self.data.iloc[idx, 0]
            image = Image.open(img_path).convert('RGB')
            label = self.data.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            query_images.append(image)
            query_labels.append(label)

        return torch.stack(support_images), torch.tensor(support_labels), torch.stack(query_images), torch.tensor(query_labels)

    def eval(self, cls_score, target, topk=(1,)):
        batch_size = cls_score.size(0)
        res = cls_accuracy(cls_score, target, topk)
        for key in res.keys():
            if key not in self.results.keys():
                self.results[key] = Avg_values()
            self.results[key].update(res[key].item(), batch_size)

    def get_eval_res(self):
        res = dict()
        for key in self.results.keys():
            res[key] = self.results[key].avg
        self.results = dict()
        return res
    

class SARACDFewShotBatchSampler(BatchSampler):
    def __init__(self, class_to_indices, num_episodes, n_way, k_shot, q_query):
        """
        Few-Shot Sampler to sample episodes for few-shot learning.

        Args:
            class_to_indices (dict): Dictionary mapping class labels to sample indices.
            num_episodes (int): Number of episodes to sample.
            n_way (int): Number of different classes per episode (N-way).
            k_shot (int): Number of samples per class in the support set (K-shot).
            q_query (int): Number of samples per class in the query set.
        """
        self.class_to_indices = class_to_indices
        self.num_episodes = num_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def __iter__(self):
        for _ in range(self.num_episodes):  
            sampled_classes = np.random.choice(list(self.class_to_indices.keys()), self.n_way, replace=False)
            support_indices = []
            query_indices = []

            for cls in sampled_classes:
                cls_indices = np.random.permutation(self.class_to_indices[cls])
                support_indices.extend(cls_indices[:self.k_shot])
                query_indices.extend(cls_indices[self.k_shot:self.k_shot + self.q_query])

            yield [support_indices + query_indices]

    def __len__(self):
        return self.num_episodes
