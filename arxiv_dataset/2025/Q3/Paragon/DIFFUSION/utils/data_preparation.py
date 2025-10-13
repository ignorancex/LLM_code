import os
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

def extract_parameters(model_path):
    """提取模型的后20层参数,返回这些层的参数名称、形状和展开的一维向量"""
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    params = []
    param_info = OrderedDict()
    
    layer_names = list(state_dict.keys())
    num_layers_to_extract = min(4, len(layer_names))
    
    for name in layer_names[-num_layers_to_extract:]:
        param = state_dict[name]
        param_info[name] = param.shape
        params.append(param.cpu().numpy().flatten())
    
    return param_info, np.concatenate(params) if params else np.array([])

def process_models(base_folder='', channel=8, label_em_dim=8):
    all_params = []
    all_labels = []
    acc_list = ["0.0", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75", "0.85", "1.0"]
    res = []
    
    for folder in tqdm(os.listdir(base_folder)):
        if folder.startswith('SASRec'):
            parts = folder.split('_')
            acc = parts[1][3:]
            div = parts[2][3:]
            folder_path = os.path.join(base_folder, folder)
            for file in os.listdir(folder_path):
                if file.startswith('finetune') and file.endswith('.pt') and (acc in acc_list):
                    res.append(acc)
                    file_path = os.path.join(folder_path, file)
                    _, params = extract_parameters(file_path)
                    all_params.append(params)
                    all_labels.append([float(acc), float(div)])
    
    all_params = np.array(all_params)
    all_labels = np.array(all_labels)

    all_params = all_params.reshape((all_params.shape[0], channel, -1))
    all_labels = all_labels.reshape((all_labels.shape[0], 2))

    scale = max(1, np.max(np.abs(all_params)))
    all_params = all_params / scale

    all_params = all_params.astype(np.float32)
    all_labels = all_labels.astype(np.float32)
    return all_params, all_labels, scale

def restore_parameters(params, param_info, scale, output_path=""):
    """将展平的参数向量恢复为模型参数并保存为.pt文件"""
    params = params * scale
    flat_params = params.reshape(-1)
    
    restored_state_dict = OrderedDict()
    current_position = 0
    
    for name, shape in param_info.items():
        num_elements = np.prod(shape)
        param_flat = flat_params[current_position:current_position + num_elements]
        param_reshaped = param_flat.reshape(shape)
        param_tensor = torch.FloatTensor(param_reshaped)
        restored_state_dict[name] = param_tensor
        current_position += num_elements
    
    torch.save(restored_state_dict, output_path)

def compare_state_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        print("Keys不匹配")
        return False
    
    all_close = True
    for key in dict1.keys():
        if not torch.allclose(dict1[key], dict2[key], atol=1e-6):
            print(f"参数 {key} 不匹配")
            all_close = False
    
    return all_close