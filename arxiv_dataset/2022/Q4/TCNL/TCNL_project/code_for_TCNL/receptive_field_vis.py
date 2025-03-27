from model.new_resnet import *
from model.new_vgg import *
import os
import sys
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
from skimage import io
from torch.autograd import Variable
import cv2
from utils.data_utils import get_train_loader
from matplotlib import pyplot as plt

def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def choose_tlayer(model_obj):
    name_to_num = {}
    sel_module = False
    name_module = None
    module_list = ['Sequential', 'Bottleneck', 'container', 'Block', 'densenet']
    while True:
        for num, module in enumerate(model_obj.named_children()):
            if any(x in torch.typename(module[1]) for x in module_list):
                print(f'[ Number: {num},  Name: {module[0]} ] -> Module: {module[1]}\n')
                name_to_num[module[0]] = num
            else:
                print(f'[ Number: {num},  Name: {module[0]} ] -> Layer: {module[1]}\n')
                name_to_num[module[0]] = num

        print('<<      You sholud not select [classifier module], [fc layer] !!      >>')
        if sel_module == False:
            a = input('Choose "Number" or "Name" of a module containing a target layer or a target layer: ')
        else:
            a = input(f'Choose "Number" or "Name" of a module containing a target layer or a target layer in {name_module} module: ')

        print('\n' * 3)
        m_val = list(model_obj._modules.values())
        m_key = list(model_obj._modules.keys())
        if isInt_str(a) == False:
            a = name_to_num[a]
        try:
            if any(x in torch.typename(m_val[int(a)]) for x in module_list):
                model_obj = m_val[int(a)]
                name_module = m_key[int(a)]
                sel_module = True
            else:
                t_layer = m_val[int(a)]
                return t_layer

        except IndexError:
            print('Selected index (number) is not allowed.')
            # raise NoIndexError('Selected index (number) is not allowed.')
        except KeyError:
            print('Selected name is not allowed.')


def isInt_str(v):
    v = str(v).strip()
    return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()


def load_model(args):
    # model = resnet_multi_concept_for_mammal(args)
    model = vgg_multi_concept_for_mammal(args)
    # model = vgg(num_classes=5)
    # 在cpu或者cuda:0上加载模型
    # ckpt = torch.load(model_name, map_location='cuda:0')
    ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
    model.load_state_dict(ckpt['net'])
    model.to(args['device'])
    model.eval()
    return model


def save_map(grad, img_path, i):
    # grad = np.abs(grad - grad.max())
    grad = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    cam = grad
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    # img[:,:,0] = np.multiply(img[:,:,0], grad)
    # img[:,:,1] = np.multiply(img[:,:,1], grad)
    # img[:,:,2] = np.multiply(img[:,:,2], grad)
    target_dir = '/home/wangzhihao/XAI/XCSGCNN/train/result/rf_vis'
    target_path = os.path.join(target_dir, str(i)+'.jpg')
    cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    cam = np.float32(cam) / 255
    img = 1.0*cam + img
    img = img / img.max()
    img = img * 255
    # cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    cv2.imwrite(target_path, img)


class RF_vis():
    def __init__(self, args:dict) -> None:
        self.img_path_list = args['img_path_list']
        self.model_path = args['ckpt_path']
        self.input_size = args['input_size']
        self.device = args['device']
        self.input = dict()
        self.activations = dict()
        self.model = load_model(args)
        self.model = self.model.cuda()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            self.input['value'] = input
            return None

        model_obj = self.model
        self.target_layer = choose_tlayer(model_obj)
        self.target_layer.register_forward_hook(forward_hook)

    def __call__(self):
        transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        heatmap = np.zeros(shape=(self.input_size, self.input_size))
        for img_path in tqdm(self.img_path_list):
            image = Image.open(img_path).convert('RGB')
            image = transformer(image)
            image = torch.unsqueeze(image, dim=0).to(self.device)
            image.requires_grad = True
            # 获取原图的最终输出
            _, pred_class, _ = self.model.inference(image)
            self.class_index = np.argmax(pred_class.cpu().data.numpy())
            # y_predict.append(self.class_index)

            #利用forward_hook抓取最后一个卷积层的输出
            feature_array = self.activations['value'].squeeze()
            for i in range(feature_array.shape[0]):
                feature_map = feature_array[i]
                feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)
                grad = torch.relu(image.grad)
                grad = grad.mean(dim=1, keepdim=False).squeeze()
                grad = grad.cpu().data.numpy()
                
                save_map(grad, img_path, i)


def my_grad():
    x = torch.tensor([[0.0, 0.0], [0.0, 4.0]], requires_grad=True)
    y = x * 2
    print(y)
    z = torch.tensor([[1.0, 1.0], [1.0, 1.0]])    #传入与y同形的权重向量
    y.backward(z)
    print(x.grad)



if __name__ == '__main__':
    root_dir = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val/object/cat'
    file_list = []
    dir_list = []
    get_file_path(root_dir, file_list, dir_list)
    file_list = ['/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val/object/cow/cow_116.jpg']
    args = {
        # 'data_loader': data_loader,
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/exp_v3/vgg_multi_concept_for_mammal/ckpt_1250_acc81.82_loss2.54.pt',
        'img_path_list': file_list,
        'input_size': 224,
        'device': 'cuda:0',
        'num_classes': 5,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }
    op = RF_vis(args)
    op()

