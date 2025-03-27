from model.new_resnet import *
import os
import sys
import numpy as np
import torch
from torchvision.transforms import transforms as T
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
            a = input(
                f'Choose "Number" or "Name" of a module containing a target layer or a target layer in {name_module} module: ')

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
        model = resnet_multi_concept_for_mammal(args)
        # 在cpu或者cuda:0上加载模型
        # ckpt = torch.load(model_name, map_location='cuda:0')
        ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
        model.load_state_dict(ckpt['net'])
        model.to(args['device'])
        # model.eval()
        return model


def normalization(array):
    # print(array)
    # array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = torch.tensor(array, dtype=torch.float)
    array = torch.softmax(array,dim=0)
    array = array.data.numpy()
    return array


class concept_weight_calculator:
    def __init__(self, args) -> None:
        self.model_path = args['ckpt_path']
        self.cuda_device = args['device']
        self.model = self.load_model(args)
        self.data_loader = args['data_loader']
        self.concept_list = ['head', 'torso', 'leg', 'shape']


    def load_model(self, args):
        model = resnet_multi_concept_for_mammal(args)
        # 在cpu或者cuda:0上加载模型
        # ckpt = torch.load(model_name, map_location='cuda:0')
        ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
        model.load_state_dict(ckpt['net'])
        model.to(args['device'])
        model.eval()
        return model


    def normalization(self, array):
        # print(array)
        # array = (array - np.min(array)) / (np.max(array) - np.min(array))
        array = torch.tensor(array, dtype=torch.float)
        array = torch.softmax(array,dim=0)
        array = array.data.numpy()
        return array


    def __call__(self) -> list:
        wrong_cnt = 0
        weight_list = list()
        for data in tqdm(self.data_loader):
            image = data['image'].to(self.cuda_device)
            label = data['label']
            feature_dict, pred_class, reconstuct_result = self.model.inference(image)
            pred_class = torch.softmax(pred_class[0], dim=0)
            if label[0].data.numpy() != np.argmax(pred_class.cpu().data.numpy()):
                wrong_cnt += 1
                # print('wrong pred')
                continue
            prob = pred_class.cpu().data.numpy()[label]
            print(prob)
            # cure_concept_weight_list = [head, torso, leg, shape]
            curr_concept_weight_list = list()
            for concept in self.concept_list:
                zero_tensor = torch.zeros(size=feature_dict[concept].shape)
                if concept == 'head':
                    feature = self.model.feature_concat(zero_tensor, feature_dict['torso'], feature_dict['leg'], feature_dict['shape'])
                    new_pred = self.model.classifier(feature.to(self.cuda_device))
                    new_pred = torch.softmax(new_pred[0], dim=0)
                    new_prob = new_pred.cpu().data.numpy()[label]
                    print(new_prob)
                    weight = min(new_prob - prob, 0)
                    curr_concept_weight_list.append(abs(weight))
                elif concept == 'torso':
                    feature = self.model.feature_concat(feature_dict['head'], zero_tensor, feature_dict['leg'], feature_dict['shape'])
                    new_pred = self.model.classifier(feature.to(self.cuda_device))
                    new_pred = torch.softmax(new_pred[0], dim=0)
                    new_prob = new_pred.cpu().data.numpy()[label]
                    weight = min(new_prob - prob, 0)
                    curr_concept_weight_list.append(abs(weight))
                elif concept == 'leg':
                    feature = self.model.feature_concat(feature_dict['head'], feature_dict['torso'], zero_tensor, feature_dict['shape'])
                    new_pred = self.model.classifier(feature.to(self.cuda_device))
                    new_pred = torch.softmax(new_pred[0], dim=0)
                    new_prob = new_pred.cpu().data.numpy()[label]
                    weight = min(new_prob - prob, 0)
                    curr_concept_weight_list.append(abs(weight))
                elif concept == 'shape':
                    feature == self.model.feature_concat(feature_dict['head'], feature_dict['torso'], feature_dict['leg'], zero_tensor)
                    new_pred = self.model.classifier(feature.to(self.cuda_device))
                    new_pred = torch.softmax(new_pred[0], dim=0)
                    new_prob = new_pred.cpu().data.numpy()[label]
                    weight = min(new_prob - prob, 0)
                    curr_concept_weight_list.append(abs(weight))
            curr_concept_weight_list = self.normalization(np.array(curr_concept_weight_list))
            weight_list.append(curr_concept_weight_list)
        print(wrong_cnt)
        return weight_list


class concept_weight_calculator_grad:
    def __init__(self, args) -> None:
        self.model_path = args['ckpt_path']
        self.cuda_device = args['device']
        self.model = load_model(args)
        self.data_loader = args['data_loader']
        self.concept_list = ['head', 'torso', 'leg', 'shape']
        self.gradients = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
        
        model_obj = self.model
        self.target_layer = choose_tlayer(model_obj)
        self.target_layer.register_backward_hook(backward_hook)
    
    def normalization(self, array):
        # print(array)
        # array = (array - np.min(array)) / (np.max(array) - np.min(array))
        array = torch.tensor(array, dtype=torch.float)
        array = torch.softmax(array,dim=0)
        array = array.data.numpy()
        return array

    def __call__(self):
        wrong_cnt = 0
        weight_list = list()
        for data in tqdm(self.data_loader):
            image = data['image'].to(self.cuda_device)
            label = data['label']
            feature_dict, pred_class, reconstruct_result = self.model.inference(image)
            one_hot = np.zeros((1, pred_class.size()[-1]), dtype=np.float32)
            # if label[0].data.numpy() != np.argmax(pred_class.cpu().data.numpy()):
            #     wrong_cnt += 1
            #     # print('wrong pred')
            #     continue
            # for mammal it is [head, torso, leg, shape]
            current_concept_weight_list = list()
            one_hot[0][label] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot.cuda() * pred_class)
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            gradients = self.gradients['value']
            weight = torch.sum(gradients, dim=2)
            weight = torch.sum(weight, dim=2)
            weight = torch.mean(weight, dim=1)
            weight = F.relu(weight)
            weight = weight.cpu().data.numpy()
            weight_list.append(weight)
        return weight_list
            
            
        


if __name__ == '__main__':
    data_loader = get_train_loader('/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train_multi_concept.txt', 1)
    args = {
        'data_loader': data_loader,
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/ckpt_260_acc74.55_loss3.03.pt',
        'device': 'cuda:0',
        'num_classes': 5,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }
    # calculator = concept_weight_calculator_grad(args)
    # weight_list = calculator()

    # weight_list = np.array(weight_list)
    # np.save('/home/wangzhihao/XAI/XCSGCNN/train/result/leg_weight_list.npy', weight_list)
    # weight_list = np.load('/root/XCSGCNN/weight_list.npy')
    # for i in weight_list:
    #     print(i)
    # head_weight_list = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/head_weight_list.npy')
    # torso_weight_list = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/torso_weight_list.npy')
    # leg_weight_list = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/leg_weight_list.npy')
    # shape_weight_list = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/shape_weight_list.npy')
    # final_weight_list = list()
    # for i in range(976):
    #     curr_list = list()
    #     curr_list.append(head_weight_list[i][0])
    #     curr_list.append(torso_weight_list[i][0])
    #     curr_list.append(leg_weight_list[i][0])
    #     curr_list.append(shape_weight_list[i][0])
    #     curr_array = np.array(curr_list)
    #     print(curr_array)
    #     final_weight_list.append(normalization(curr_array))
    # np.save('/home/wangzhihao/XAI/XCSGCNN/train/result/final_weight_list.npy', final_weight_list)
    weight_list = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/concept_weight_grad/final_weight_list.npy')
    result = np.sum(weight_list, axis=0)
    result = normalization(result)
    print(result)


