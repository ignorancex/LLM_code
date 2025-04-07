
import os
import cv2
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class Loader_train(Dataset):

    def __init__(self, path, set_list=None, preprocess=None, jump=30):

        self.preprocess = preprocess

        if self.preprocess is None:

            self.preprocess = transforms.Compose([transforms.ToTensor()])



        self.data = []

        if set_list is None: set_list = os.listdir(path)

        for subset in set_list:

            subset_path = os.path.join(path, subset)

            data = Loader_train.loadData(subset_path)

            # print(X)
            # print(y)



            self.data += data


        self.data = np.asarray(self.data)


        self.jump = jump



    def __len__(self):

        # return len(self.data)
        return int(len(self.data) / self.jump)



    def __getitem__(self, idx):

        idx = idx * self.jump

        img_path    = self.data[idx, 0]
        input_image = Image.open(img_path)

        label = self.data[idx, 1]


        # reduce mean
        input_tensor = self.preprocess(input_image)


        for k, v in label.items():

            label[k] = torch.as_tensor(v)


        return input_tensor, label
        # return {'X':input_tensor, 'Y':label}




    @staticmethod
    def loadData(path):

        images_path = os.path.join(path, 'img1')

        list_path = os.listdir(images_path)
        list_frames = Loader_train.loadGT(path)


        data = []

        for img_path in list_path:
        # for img_path in list_path[:2]:

            frame = int(img_path.split('.')[0])
            img_path = images_path + '/' + img_path

            info = {}

            boxes = list_frames[frame][:, 1:5]
            labels = list_frames[frame][:, 6]

            # print(list_frames[frame])

            # print(boxes)
            boxes_end = []

            for bx in boxes:

                x1 = int(bx[0])
                y1 = int(bx[1])

                x2 = int(bx[0] + bx[2])
                y2 = int(bx[1] + bx[3])

                boxes_end.append([x1, y1, x2, y2])

            info['boxes'] = torch.as_tensor(boxes_end)
            info['labels'] = torch.as_tensor(labels, dtype=torch.int64)

            # print(torch.as_tensor(list_frames[frame][5], dtype=torch.int64))
            # info['scores'] = 1.0

            data.append( [img_path, info] )

        # print(frames.keys())

        # list_path = [ images_path + '/' + img_path for img_path in list_path]

        return data



    @staticmethod
    def loadGT(path):

        gt_path = os.path.join(path, 'gt/gt.txt')

        file = np.loadtxt(gt_path, delimiter=',')

        unique = np.unique(file[:, 0])


        frame = {}

        for u in unique:

            a = np.where(file[:, 0] == u)
            u = int(u)

            frame[u] = file[a][:, 1:]


        return frame




if __name__ == "__main__":


    train_data = Loader_train('dataset/MOT20', set_list=['MOT20-02'])

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size(), sample['Y'].size())

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
