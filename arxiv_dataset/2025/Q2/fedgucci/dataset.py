import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import math
import random
import copy
import functools
from PIL import Image
import os
import glob
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class CustomSubset(torch.utils.data.Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        dataset.targets = torch.tensor(dataset.targets)
        self.targets = dataset.targets[indices]
        self.classes = dataset.classes
        self.indices = indices

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self):
        return len(self.indices)


EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'

class TinyImageNet(Dataset):
    """
    Ref: https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(self.root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing
        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # get targets
        self.targets = []
        for index in range(len(self.image_paths)):
            file_path = self.image_paths[index]
            label_numeral = self.labels[os.path.basename(file_path)]
            self.targets.append(label_numeral)

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img) if self.transform else img

class Glue(Dataset):
    
    def __init__(self, dst_name, tokenizer_path, split = "train"):
        self.dst = load_dataset("glue", dst_name)
        def tokenize_function(examples):
                # max_length=None => use the model max length (it's actually the default)
                column1, column2 = task_to_keys[dst_name]
                columns = (examples[column1], ) if column2 is None else (examples[column1], examples[column2])
                outputs = self.tokenizer(*columns, truncation=True, max_length=self.max_seq_len)
                return outputs
        self.max_seq_len = 512
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli_matched": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if dst_name in ['mrpc', 'rte', 'wnli', 'stsb']: 
            remove_columns = ["idx", "sentence1", "sentence2"]
        elif 'mnli' in dst_name:
            remove_columns = ["idx", "premise", "hypothesis"]
        elif dst_name == 'qnli':
            remove_columns = ["idx", "question", "sentence"]
        elif dst_name in ['sst2', 'cola']:
            remove_columns = ["idx", "sentence"]
        self.dst = self.dst.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
        )
        self.dst = self.dst.rename_column("label", "labels")
        self.dst = self.dst[split]
        self.num_classes = None
    
    def __len__(self):
        return len(self.dst)
    
    def __getitem__(self, index):
        return [self.dst[index]["input_ids"], self.dst[index]["attention_mask"]], self.dst[index]["labels"]
    
    def get_num_classes(self):
        if self.num_classes is None:
            self.num_classes = len(set(self.dst["labels"]))
        return self.num_classes
    
    def get_collator(self):
        def collator(batch):
            new_batch = []
            for i, ((input_id, attention_mask), labels) in enumerate(batch):
                new_batch.append({"input_ids": input_id, "attention_mask": attention_mask, "labels": labels})
            batch = self.tokenizer.pad(new_batch, padding="longest", max_length=self.max_seq_len, return_tensors="pt")
            return [batch["input_ids"], batch["attention_mask"]], batch["labels"]
        return collator

class Data(object):
    def __init__(self, args):
        self.args = args
        node_num = args.node_num

        if args.dataset == 'mnist':
            # Data enhancement
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

            self.train_set = torchvision.datasets.MNIST(
                root="./Dataset/mnist/", train=True, download=False, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=10, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(60000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.MNIST(
                root="./Dataset/mnist/", train=False, download=False, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'cifar10':
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            if args.local_model == 'ViT':
                tra_transformer = _transform(224)
                val_transformer = _transform(224)

            self.train_set = torchvision.datasets.CIFAR10(
                root="./Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=10, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=10, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            # not sample from every client's test dataset
            self.test_set = torchvision.datasets.CIFAR10(
                root="./Dataset/cifar/", train=False, download=True, transform=val_transformer
            )

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'cifar100':
            tra_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            val_transformer = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            if args.local_model == 'ViT':
                tra_transformer = _transform(224)
                val_transformer = _transform(224)

            self.train_set = torchvision.datasets.CIFAR100(
                root="./Dataset/cifar/", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(50000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.CIFAR100(
                root="./Dataset/cifar/", train=False, download=False, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'fmnist':
            # Data enhancement
            tra_transformer_arr = [
                transforms.ToTensor(),
            ]
            val_transformer_arr = [
                transforms.ToTensor(),
            ]
            if 'VGG' in args.local_model:
                tra_transformer_arr.append(transforms.Resize([32, 32]))
                val_transformer_arr.append(transforms.Resize([32, 32]))
            tra_transformer = transforms.Compose(
                tra_transformer_arr
            )
            val_transformer = transforms.Compose(
                val_transformer_arr
            )
            self.train_set = torchvision.datasets.FashionMNIST(
                root="./Dataset/FashionMNIST", train=True, download=True, transform=tra_transformer
            )
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=100, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=100, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(60000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = torchvision.datasets.FashionMNIST(
                root="./Dataset/FashionMNIST", train=False, download=False, transform=val_transformer
            )
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'tinyimagenet':
            # Data enhancement
            tra_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            val_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

            self.train_set = TinyImageNet("~/Dataset/tiny-imagenet-200/", 'train', tra_transformer, in_memory=False)
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)
                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(random_state=random_state, dataset=self.train_set, non_iid_alpha1=args.dirichlet_alpha,non_iid_alpha2=args.dirichlet_alpha2 ,num_classes=200, num_indices=num_indices, n_workers=node_num)  
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(random_state=random_state, dataset=self.train_set, non_iid_alpha=args.dirichlet_alpha, num_classes=200, num_indices=num_indices, n_workers=node_num)  
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion
            else:
                data_num = [int(100000/node_num) for _ in range(node_num)]
                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_set = TinyImageNet("~/Dataset/tiny-imagenet-200/", 'val', val_transformer, in_memory=False)
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])
        
        elif "glue" in args.dataset:
            dst_name = args.dataset.split(".")[1]
            self.train_set = Glue(dst_name=dst_name, tokenizer_path=args.model_path, split="train")
            data_num = [int(len(self.train_set)/node_num) for _ in range(node_num)]
            splited_set = random_split(data_num)
            self.train_loader = splited_set

            split = "test" if dst_name == "mrpc" else "validation"
            self.test_set = Glue(dst_name=dst_name, tokenizer_path=args.model_path, split=split)
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])


######## NonIID functions ###############

def build_non_iid_by_dirichlet_hybrid(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha1 = 10, non_iid_alpha2 = 1, num_classes = 10, num_indices = 60000, n_workers = 10
):
    
    #TODO
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    partition = random_state.dirichlet(np.repeat(non_iid_alpha1, n_workers), num_classes).transpose()

    partition2 = random_state.dirichlet(np.repeat(non_iid_alpha2, n_workers/2), num_classes).transpose()

    new_partition1 = copy.deepcopy(partition[:int(n_workers/2)])

    sum_distr1 = np.sum(new_partition1, axis=0)

    diag_mat = np.diag(1 - sum_distr1)

    new_partition2 = np.dot(diag_mat, partition2.T).T

    client_partition = np.vstack((new_partition1, new_partition2))

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition

def build_non_iid_by_dirichlet_new(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    
    #TODO
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition

def random_split(data_num: list):
    idxs = np.arange(sum(data_num))
    np.random.shuffle(idxs)
    dict_users = {}
    id_start = 0
    for i in range(len(data_num)):
        dict_users[i] = idxs[id_start: id_start + data_num[i]].tolist()
        id_start += data_num[i]
    return dict_users

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])