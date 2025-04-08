import os
import itertools
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import random
import copy
import constants

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

manual_seed = 1773
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

def create_directories_for_saving():
    global_saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}'
    global_saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.GLOBAL_RUN_NAME}'
    if not os.path.exists(global_saving_path):
        os.makedirs(global_saving_path)
        print("Directory created:", global_saving_path)
    else:
        print("Directory already exists:", global_saving_path)

    saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}'
    saving_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print("Directory created:", saving_path)
    else:
        print("Directory already exists:", saving_path)

    saving_weights_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights'
    saving_weights_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights'
    if not os.path.exists(saving_weights_path):
        os.makedirs(saving_weights_path)
        print("Directory created:", saving_weights_path)
    else:
        print("Directory already exists:", saving_weights_path)

    saving_results_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results'
    saving_results_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results'
    if not os.path.exists(saving_results_path):
        os.makedirs(saving_results_path)
        print("Directory created:", saving_results_path)
    else:
        print("Directory already exists:", saving_results_path)

def initialize_losses_arrays():
    losses = []
    for fold in range(constants.NUMBER_FOLDS):
        losses.append([])
        for client in range(constants.NUMBER_CLIENTS):
            losses_current = {'l1_test1': [], 'pcc_test1': [], 'topological_test1': [], 'global_test1': [], 
                    'l1_test2': [], 'pcc_test2': [], 'topological_test2': [], 'global_test2': [], 
                    'global': []}
            losses[fold].append(losses_current)

    return losses

def initialize_losses_arrays_simple():
    losses = []
    for fold in range(constants.NUMBER_FOLDS):
        losses.append([])
        for client in range(constants.NUMBER_CLIENTS):
            losses_current = {'g': [], 'd': [], 'a': [], 'l1_loss': [], 'pcc_loss': [], 'eigen_loss': [], 'global_test1': []}
            losses[fold].append(losses_current)

    return losses

def generate_fold_configurations(n):
    fold_configs = []
    one_config = [i for i in range(n)]
    fold_configs.append(one_config)

    for i in range(n - 1):
        one_config = one_config[1:] + [one_config[0]]
        fold_configs.append(one_config)
    
    return fold_configs

def generate_fold_indices(source_data):
    kf = KFold(n_splits=constants.NUMBER_FOLDS, shuffle=True, random_state=1773)
    kf = KFold(n_splits=constants.NUMBER_FOLDS, shuffle=True, random_state=1773)
    
    fold_indices = []
    for train_index, test_index in kf.split(source_data):
        fold_indices.append(test_index)

    return fold_indices

def generate_fold_indices_dataset(dataset):
    kf = StratifiedKFold(n_splits=constants.NUMBER_FOLDS, shuffle=True, random_state=1773)
    input_list = []
    target_list = []
    for input, target in dataset:
        input_list.append(input)
        target_list.append(target)

    # print(target_list)
    fold_indices = []
    for train_index, test_index in kf.split(input_list, target_list):
        fold_indices.append(test_index)

    return fold_indices

def concatenate_tensors(tensors, dim=0):
    if dim < 0:
        dim += tensors[0].dim()

    sizes = [tensor.size(dim) for tensor in tensors]
    cat_size = sum(sizes)
    cat_shape = list(tensors[0].shape)
    cat_shape[dim] = cat_size

    result = torch.empty(cat_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    offset = 0
    for tensor in tensors:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(offset, offset + tensor.size(dim))
        result[tuple(slices)] = tensor
        offset += tensor.size(dim)

    return result

def save_losses_to_file(losses, fold, loss_name):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results/losses_{loss_name}_fold_{fold}.txt', "a")
    f = open(f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/results/losses_{loss_name}_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        losses_current_client = ""
        for epoch_loss in losses[fold][client][loss_name]:
            losses_current_client += str(epoch_loss) + ","
        
        f.write(losses_current_client)
        f.write('\n')

    f.close()

def save_mae_to_file(models, fold, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        f.write(str(models[fold][client].get_best_mae()))
def save_mae_to_file(models, fold, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        f.write(str(models[fold][client].get_mae()))
        f.write('\n')
        
    f.close()

def save_average_mae_to_file(models, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_average_folds.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        mae_folds = []
        for fold in range(constants.NUMBER_FOLDS):
            mae_folds.append(models[fold][client].get_mae())
        mae_folds = np.array(mae_folds)
        mae_average = np.mean(mae_folds)

        f.write(str(mae_average))
        f.write('\n')
        
    f.close()

def save_metric_to_file(models, fold, current_run_name = constants.RUN_NAME, metric='acc'):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/{metric}_fold_{fold}.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        if metric == 'acc':
            metric_to_write = models[fold][client].get_accuracy()
        elif metric == 'sens':
            metric_to_write = models[fold][client].get_sensitivity()
        elif metric == 'spec':
            metric_to_write = models[fold][client].get_specificity()
        elif metric == 'f1':
            metric_to_write = models[fold][client].get_f1()

        f.write(str(metric_to_write))
        f.write('\n')
        
    f.close()

def save_average_metric_to_file(models, current_run_name = constants.RUN_NAME, metric='acc'):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/{metric}_average_folds.txt', "a")

    for client in range(constants.NUMBER_CLIENTS):
        mae_folds = []
        for fold in range(constants.NUMBER_FOLDS):
            if metric == 'acc':
                metric_to_write = models[fold][client].get_accuracy()
            elif metric == 'sens':
                metric_to_write = models[fold][client].get_sensitivity()
            elif metric == 'spec':
                metric_to_write = models[fold][client].get_specificity()
            elif metric == 'f1':
                metric_to_write = models[fold][client].get_f1()
                
            mae_folds.append(metric_to_write)
        mae_folds = np.array(mae_folds)
        mae_average = np.mean(mae_folds)

        f.write(str(mae_average))
        f.write('\n')
        
    f.close()

def read_losses_from_file(fold, current_run_name = constants.RUN_NAME, loss_name=None):
    if loss_name != None:
        f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/losses_{loss_name}_fold_{fold}.txt', "r")
    else:
        f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/losses_fold_{fold}.txt', "r")

    losses_fold = []
    for client in range(constants.NUMBER_CLIENTS):
        losses_current_client_as_string = f.readline()
        losses_current_client = losses_current_client_as_string.split(',')
        losses_current_client = losses_current_client[:-1]
        # losses_current_client = np.array(losses_current_client)
        # losses_current_client_float = losses_current_client.astype(np.float)
        losses_current_client_float = np.asarray(losses_current_client, dtype=float)
        
        losses_fold.append(losses_current_client_float)
    
    f.close()

    return losses_fold

def read_mae_from_file(fold, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/mae_fold_{fold}.txt', "r")

    mae_fold = []
    for client in range(constants.NUMBER_CLIENTS):
        mae_current_client_as_string = f.readline()

        if len(mae_current_client_as_string) > 0:
            mae_current_client_float = np.asarray(mae_current_client_as_string, dtype=float)
            mae_fold.append(mae_current_client_float)
        
    
    f.close()

    return mae_fold

def read_acc_from_file(fold, current_run_name = constants.RUN_NAME, metric='acc'):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/{metric}_fold_{fold}.txt', "r")

    mae_fold = []
    # for client in range(constants.NUMBER_CLIENTS):
    #     mae_current_client_as_string = f.readline()
    for client in range(constants.NUMBER_CLIENTS):
        mae_current_client_as_string = f.readline()
        mae_current_client_as_string = mae_current_client_as_string.strip()

        if len(mae_current_client_as_string) > 0:
            mae_current_client_float = np.asarray(mae_current_client_as_string, dtype=float)
            mae_fold.append(mae_current_client_float)
    
    f.close()

    return mae_fold

def read_acc_from_file(fold, current_run_name = constants.RUN_NAME, metric='acc'):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/results/{metric}_fold_{fold}.txt', "r")

    mae_fold = []
    # for client in range(constants.NUMBER_CLIENTS):
    #     mae_current_client_as_string = f.readline()
    for client in range(constants.NUMBER_CLIENTS):
        mae_current_client_as_string = f.readline()
        mae_current_client_as_string = mae_current_client_as_string.strip()

        if len(mae_current_client_as_string) > 0:
            mae_current_client_float = np.asarray(mae_current_client_as_string, dtype=float)
            mae_fold.append(mae_current_client_float)
    
    f.close()

    return mae_fold

def get_mdl_params(model_list, n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            if name in constants.aggregating_layers:
                n_par += len(param.data.reshape(-1))
        
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            if name in constants.aggregating_layers:
                temp = param.data.cpu().numpy().reshape(-1)
                param_mat[i, idx:idx + len(temp)] = temp
                idx += len(temp)
    return np.copy(param_mat)

def get_mdl_params_resnet(model_list, n_par=None):
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            if "conv" in name:
                n_par += len(param.data.reshape(-1))
        
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            if name in constants.aggregating_layers:
                temp = param.data.cpu().numpy().reshape(-1)
                param_mat[i, idx:idx + len(temp)] = temp
                idx += len(temp)
    return np.copy(param_mat)

def set_model_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        if name in constants.aggregating_layers:
            weights = param.data
            length = len(weights.reshape(-1))
            dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
            idx += length
        
    mdl.load_state_dict(dict_param, strict=False)    
    return mdl

def set_model_from_params_resnet(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        # if name in constants.aggregating_layers:
        if "conv" in name:
            weights = param.data
            length = len(weights.reshape(-1))
            dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
            idx += length
        
    mdl.load_state_dict(dict_param, strict=False)    
    return mdl

def l2_norm_parameters(parameter_vector1, parameter_vector2):
    # Convert the input lists or arrays to numpy arrays for ease of computation
    theta = parameter_vector1.cpu().numpy()
    phi = parameter_vector2.cpu().numpy()

    # print('\n')
    # print(theta)
    # print(phi)
    # print('\n')

    # Calculate the squared differences between corresponding elements
    squared_diff = (theta - phi) ** 2

    # Sum up the squared differences
    sum_squared_diff = np.sum(squared_diff)

    # Compute the L2 norm (Euclidean distance)
    l2_distance = np.sqrt(sum_squared_diff)

    return l2_distance

def l2_norm_models(model1, model2):
    no_layers_conv = 0
    l2_norm_total = 0

    for param_name in model1.state_dict():
        if "conv" in param_name:
            l2_norm = l2_norm_parameters(model1.state_dict()[param_name].data, model2.state_dict()[param_name].data)
            l2_norm_total = l2_norm_total + l2_norm
            no_layers_conv = no_layers_conv + 1
    
    # for name, param in model1.named_parameters():
    #     if "conv" in name:
    #         l2_norm = l2_norm_parameters(model1.state_dict()[name].data, model2.state_dict()[name].data)
    #         l2_norm_total = l2_norm_total + l2_norm
    #         no_layers_conv = no_layers_conv + 1

    l2_norm_total = l2_norm_total / no_layers_conv

    return l2_norm_total

def compute_weight_from_l2_norm(l2_norm_values):
    # Normalize weights to sum up to 1
    total_weight = sum(l2_norm_values)
    normalized_weights = [w / total_weight for w in l2_norm_values]

    return normalized_weights

def kl_divergence_parameters(p_parameters, q_parameters):
    # Convert raw scores to probability distributions using softmax
    p_probabilities = F.softmax(p_parameters, dim=0)
    q_probabilities = F.softmax(q_parameters, dim=0)
    
    # Calculate KL divergence between the two probability distributions
    kl_divergence = torch.sum(p_probabilities * (torch.log(p_probabilities) - torch.log(q_probabilities)))
    
    return kl_divergence

def kl_divergence_models(model1, model2):
    no_layers_conv = 0
    kl_divergence_total = 0

    for param_name in model1.state_dict():
        if "conv" in param_name:
            kl_divergence = kl_divergence_parameters(model1.state_dict()[param_name].data, model2.state_dict()[param_name].data)
            kl_divergence_total = kl_divergence_total + kl_divergence
            no_layers_conv = no_layers_conv + 1

    kl_divergence_total = kl_divergence_total / no_layers_conv

    return kl_divergence_total

def compute_weights_from_kl(kl_divergence_values, temperature=1.0):
    # Transform KL divergence values into weights using the exponential of the negative value
    weights = [torch.exp(-kl_div / temperature) for kl_div in kl_divergence_values]
    
    # Normalize the weights to sum up to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    return normalized_weights

def write_details_to_file(args, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/run_details.txt', "a")

    # Iterate through all parsed arguments
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        f.write(f"Argument '{arg_name}': {arg_value}\n")

    # f.write(f"Argument 'LR': {constants.LR}\n")
        
    f.close()

def write_time_to_file(time_cost, current_run_name = constants.RUN_NAME):
    f = open(f'{constants.SAVING_FOLDER_PATH}/{current_run_name}/run_details.txt', "a")

    f.write(f"Time cost: {time_cost}\n")
        
    f.close()