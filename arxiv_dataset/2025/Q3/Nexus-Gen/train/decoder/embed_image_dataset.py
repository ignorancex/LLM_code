import json
from PIL import Image
import torch
from torchvision import transforms


def read_jsonl(file_path, num_samples=None):
    print(f"reading from {file_path}")
    data_list = []
    samples = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
            samples += 1
            if num_samples is not None and samples >= num_samples:
                break
    print(f"read {len(data_list)} samples")
    return data_list


class GenerationDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = read_jsonl(dataset_path)
        self.path = []
        self.embed_paths = []
        for data in metadata:
            self.path.append(data['images'][0])
            self.embed_paths.append(data['embed_path'])
        self.height = height
        self.width = width
        print('train dataset size:', len(self.path))
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.

        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height * scale), round(width * scale)]
        image = transforms.functional.resize(image, shape, interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)

        embed = torch.load(self.embed_paths[data_id], weights_only=True)
        if embed.ndim == 3:
            embed = embed.squeeze(0)  # Ensure embed is 2D
        return {"embed": embed, "image": image}


    def __len__(self):
        return self.steps_per_epoch


class EditingDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        metadata = read_jsonl(dataset_path)
        self.path = []
        self.embed_paths = []
        self.ref_embed_paths = []
        for data in metadata:
            self.path.append(data['images'][1])
            self.embed_paths.append(data['embed_target'])
            self.ref_embed_paths.append(data['embed_source'])

        self.height = height
        self.width = width
        print('train dataset size:', len(self.path))
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height*scale),round(width*scale)]
        image = transforms.functional.resize(image,shape,interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        return image


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)
        image = self.preprocess_image(self.path[data_id])
        embed = torch.load(self.embed_paths[data_id], weights_only=True, map_location='cpu')
        if embed.ndim == 3:
            embed = embed.squeeze(0)  # Ensure embed is 2D
        ref_embed = torch.load(self.ref_embed_paths[data_id], weights_only=True, map_location='cpu')
        if ref_embed.ndim == 3:
            ref_embed = ref_embed.squeeze(0)
        embeds_grid = torch.tensor([1, 18, 18])
        ref_embeds_grid = torch.tensor([1, 36, 36])
        return {"image": image, "embed": embed, "ref_embed": ref_embed, "embeds_grid": embeds_grid, "ref_embeds_grid": ref_embeds_grid}


    def __len__(self):
        return self.steps_per_epoch
