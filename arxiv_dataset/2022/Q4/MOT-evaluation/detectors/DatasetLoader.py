
import os
import numpy as np

from PIL import Image
from torchvision import datasets
from torchvision import transforms


IMAGES = 'img1'

TRANSLATE_str_int = {'person':1, 'bicycle':4, 'car':3, 'motorbike':5, 'bus':3, 'truck':3, 'train':3}
TRANSLATE_int = {1:1, 2:4, 3:3, 4:5, 6:3, 7:3, 8:3}

class DatasetLoader:

    def __init__(self, detectorName, path='data/images', set_data='MOT20', savePath='outputs/detections', size=None):
        '''Create a 
        '''

        self.path = os.path.join(path, set_data)
        self.setName = set_data

        self.savePath = savePath
        self.detectorName = detectorName

        self.results = []
        self.size = size


        # Create path where to save detections
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        if not os.path.exists(os.path.join(savePath, detectorName)):
            os.makedirs(os.path.join(savePath, detectorName))

        if not os.path.exists(os.path.join(savePath, detectorName, set_data)):
            os.makedirs(os.path.join(savePath, detectorName, set_data))



    def listData(self):

        list_d = os.listdir(self.path)
        list_d.sort()

        return list_d



    def size_images_ori(self):

        img_path = os.path.join(self.data_path, 'img1')
        img_list = os.listdir(img_path)
        # print (img_path)

        img_path = img_path + '/' + img_list[0]

        img = Image.open(img_path)

        width  = img.size[0]
        height = img.size[1]

        # (1920, 1080)
        return width, height



    def size_images_mod(self):

        img_path = os.path.join(self.data_path, 'img1')
        img_list = os.listdir(img_path)
        # print (img_path)

        img_path = img_path + '/' + img_list[0]

        img = Image.open(img_path)
        img = self.transform(img)

        width  = img.shape[2]
        height = img.shape[1]

        # (1920, 1080)
        return width, height


    def loadData(self, setName, transform=None):

        if transform is None:

            transform = [
                transforms.ToTensor(),
            ]


        if not self.size is None:

            transform.insert(0, transforms.Resize(self.size))



        transform = transforms.Compose(transform)

        self.transform = transform


        path = os.path.join(self.path, setName)

        self.sets = datasets.ImageFolder(path, transform=transform)
        self.data_path = path


        return self.sets


    def framesData(self, setName):

        path = os.path.join(self.path, setName, 'img1')

        return len( os.listdir(path) )




    def update(self, frame, boxes, scores, labels, label_permited=[1], preprocess=True):

        #print(boxes.shape, scores.shape, labels.shape)

        # Process bounding boxes
        if preprocess:
            boxes = DatasetLoader.processBoxes(boxes)

        self.resizeBoxes(boxes)

        # Save permited boxes in variable.
        for b, s, l in zip(boxes, scores, labels):

            # Check if label is permited
            if (label_permited is not None) and (not l in label_permited): continue

            # Translate label
            if type(l) == str:      l = TRANSLATE_str_int[l]
            elif type(l) == int:    l = TRANSLATE_int[l]

            if (b[2] == 0) or (b[3] == 0): print('continue'); continue

            # Set format and save in variable
            # detection = [frame] + [-1] + b + [s.item()] + [-1, -1, -1]
            detection = [frame] + list(b) + [s] + [l]
            self.results.append(detection)



    def detectionsData(self):

        return len( self.results )



    def save(self, setName):

        file = os.path.join(self.savePath, self.detectorName, self.setName, setName)

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det')

        if not os.path.exists(file):
            os.makedirs(file)

        file = os.path.join(file, 'det.txt')

        np.savetxt(file, self.results, delimiter=',', fmt=['%d,-1', '%.0f', '%.0f', '%.0f', '%.0f', '%.4f', '%d,-1,-1'])

        self.results = []



    @staticmethod
    def processBoxes(boxes):

        # x1, y1, x2, y2 --> x1, y1, width, height
        boxes = np.stack((boxes[:, 0],
                          boxes[:, 1],
                          boxes[:, 2] - boxes[:, 0],
                          boxes[:, 3] - boxes[:, 1]),
                          axis=1)

        return boxes


    def resizeBoxes(self, bboxes):

        # print(bboxes)
        if self.size is None: return

        width_ori, height_ori = self.size_images_ori()
        width_mod, height_mod = self.size_images_mod()

        width_rel  = width_ori / width_mod
        height_rel = height_ori / height_mod

        # print(width_rel, '=', width_ori, '/', width_mod)
        # print(height_rel, '=', height_ori, '/', height_mod)


        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width_rel
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height_rel

        bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2

        # print(bboxes)
        # print('---------------------------------------------')
        # print('---------------------------------------------')

        return bboxes
