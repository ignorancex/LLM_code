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
plt.switch_backend('agg')


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


def save_heatmap(mask, img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    result = 1.0 * heatmap + img
    result = result / np.max(result)

    # 设置存放类激活映射图的文件夹
    # path_component = img_path.split('/')
    # new_path_component = []
    # new_path_component.extend(path_component[0:3])
    # new_path_component.extend(['code', 'interpret', 'heatmap'])
    # new_path_component.extend(path_component[5:-1])
    # cam_path = path_combination(new_path_component)
    # if not os.path.exists(cam_path):
    #     os.makedirs(cam_path)
    # cam_path = os.path.join(cam_path, path_component[-1].split('.')[0] + '.png')
    # cam_path = img_path.split('/')[0] + '/' + 'cam.jpg'
    cam_path = '/home/wangzhihao/XAI/XCSGCNN/train/result/new_vgg_cam/head/' + img_path.split('/')[-1]
    # path = r'/root/breast_duct/interpret/heatmap/'
    # if not (os.path.isdir(path)):
    #     os.makedirs(path)
    # # 设置类激活映射图的保存路径，文件名与测试数据文件名一致
    # if 'negative' in img_path:
    #     cam_path = path + 'neg_heatmap/' + img_path.split('/')[-1].split('.')[0] + '.png'
    # elif 'positive' in img_path:
    #     cam_path = path + 'pos_heatmap/' + img_path.split('/')[-1].split('.')[0] + '.png'
    # print(cam_path)
    cv2.imwrite(cam_path, np.uint8(255 * result))


class ScoreCAM():
    def __init__(self, args:dict):
        self.img_path_list = args['img_path_list']
        self.model_path = args['ckpt_path']
        self.input_size = args['input_size']
        self.device = args['device']
        self.activations = dict()
        self.model = load_model(args)
        
        self.model = self.model.cuda()
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        model_obj = self.model
        self.target_layer = choose_tlayer(model_obj)
        self.target_layer.register_forward_hook(forward_hook)

    def __call__(self):
        transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        for img_path in tqdm(self.img_path_list):
            image = Image.open(img_path).convert('RGB')
            image = transformer(image)
            image = torch.unsqueeze(image, dim=0).to(self.device)
            # 获取原图的最终输出
            _, pred_class, _ = self.model.inference(image)
            self.class_index = np.argmax(pred_class.cpu().data.numpy())
            # y_predict.append(self.class_index)

            #利用forward_hook抓取最后一个卷积层的输出
            activations = self.activations['value']

            cam = np.ones(activations[0][0].shape, dtype=np.float32)
            for i in range(len(activations[0])):
                # print('processing activations_map', i, '/', len(activations[0]))
                saliency_map = torch.unsqueeze(torch.unsqueeze(activations[0][i, :, :], 0), 0)
                # 上采样
                saliency_map = F.interpolate(saliency_map, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
                if saliency_map.max() == saliency_map.min():
                    continue
                # 归一化
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                # 获取CIC值
                w = F.softmax(self.model.inference(image * norm_saliency_map)[1][0], dim=0)[self.class_index]
                cam += w.data.cpu().numpy() * activations[0][i, :, :].data.cpu().numpy()

            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((self.input_size, self.input_size), Image.ANTIALIAS)) / 255
            save_heatmap(cam, img_path)

if __name__ == '__main__':
    # data_loader = get_train_loader('/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train_multi_concept.txt', 1)
    root_dir = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train/object'
    file_list = list()
    dir_list = list()
    get_file_path(root_dir, file_list, dir_list)
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
    scorecam = ScoreCAM(args)
    scorecam()


