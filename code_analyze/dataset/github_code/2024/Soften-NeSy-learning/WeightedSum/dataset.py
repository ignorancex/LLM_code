from nn_utils import *
import random

import torchvision
import torchvision.transforms as transforms
import pickle 
import os.path 
import sys 
sys.path.append("..") 
from params import data_root

transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])
mnist_train_data = torchvision.datasets.MNIST(root = data_root + 'MNIST', train=True, download=True,transform=transform_mnist)
mnist_test_data = torchvision.datasets.MNIST(root = data_root + 'MNIST', train=False, download=True,transform=transform_mnist)
weights = [random.randint(10,14) for _ in range(arity)]

class MathExprDataset(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None, datasetFile=None):
        super(MathExprDataset, self).__init__()
        
        dataset = mnist_test_data
        if split == 'train':
            dataset = mnist_train_data

        if os.path.exists(datasetFile):
            filehandler = open(datasetFile, 'rb') 
            dictionary = pickle.load(filehandler)
            filehandler.close()

            self.all_image_ids = dictionary['all_image_ids']
            self.all_digits = dictionary['all_digits']
            self.all_image_seqs = dictionary['all_image_seqs']
            self.all_res = dictionary['all_res']
            self.all_labels = dictionary['all_labels']

        else:
            random.seed(randomSeed)
            all_weights_images = [torch.zeros([1, 28, 28]) + i * 0.025 for i in range(arity)]
            self.all_image_ids = [[random.randint(0,len(dataset)-500) for _ in range(arity)] for _ in range(numSamples) ]
            self.all_digits = [[dataset[image_id][1] for image_id in image_ids[0:arity]] + weights for image_ids in self.all_image_ids]
            self.all_image_seqs = [[dataset[image_id][0] for image_id in image_ids] + all_weights_images for image_ids in self.all_image_ids]
            self.all_res = [sum([digits[i] * (weights[i]-9) for i in range(arity)]) for digits in self.all_digits] 
            self.all_labels = self.all_digits
            dictionary = dict()
            dictionary['all_image_ids'] = self.all_image_ids
            dictionary['all_digits'] = self.all_digits
            dictionary['all_image_seqs'] = self.all_image_seqs
            dictionary['all_res'] = self.all_res
            dictionary['all_labels'] = self.all_labels

            filehandler = open(datasetFile, 'wb') 
            pickle.dump(dictionary, filehandler)
            filehandler.close()

        self.dataset = list()
        for index in range(numSamples):
            sample = dict()
            sample['img_seq'] = self.all_image_seqs[index]
            sample['label_seq'] = self.all_labels[index]
            sample['len'] = arity*2
            sample['res'] = self.all_res[index]
            i_pred = [p if p <= 9 else p-9 for p in sample['label_seq']]
            i_expr = '+'.join([str(i_pred[i]) + '*' + str(i_pred[i+arity]) for i in range(arity)])
            sample['expr'] = i_expr
            sample['index'] = index
            self.dataset.append(sample)

    
    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        return sample
    
    def __len__(self):
        return len(self.dataset)


def MathExpr_collate(batch):
    for sample in batch:
        sample['img_seq'] = torch.stack(sample['img_seq'])
        
        sample['label_seq'] = torch.tensor(sample['label_seq'])
        
    batch = default_collate(batch)
    return batch
