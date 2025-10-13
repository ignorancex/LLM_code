import random
import pickle
import numpy as np

model_path = "/homes/sayehs/clado_mpqco_results/"
model_name = "bert-base-uncased-squad-v1"

def load_gradients(file_name):
    with open(file_name,'rb') as f:
        hm = pickle.load(f)

    L = hm['Ltilde'].shape[0]
    index2layerscheme = [None for i in range(L)]

    for name in hm['layer_index']:
        index = hm['layer_index'][name]
        if model_name == "bert-tiny":
            layer_name = name[:-10]
            scheme = name[-10:]
        elif model_name == "bert-base" or model_name == "bert-base-uncased-squad-v1":
            layer_name = name[:-11]
            scheme = name[-11:]
        else:
            assert "Model choices are: bert-base and bert-tiny."     
        index2layerscheme[index] = (layer_name, scheme)

    return index2layerscheme


random.seed(3)

num_batches = [2, 4, 8, 16, 32, 64, 128, 256, 512]
n_runs = 24
selected_samples = [0] * len(num_batches)

for i, n_batch in enumerate(num_batches):
    selected_samples[i] = []
    for run in range(n_runs):
        batch_indices = []
        while len(batch_indices) != n_batch:
            rand_index = random.randint(0, 767)
            if rand_index not in batch_indices:
                batch_indices.append(rand_index)
        selected_samples[i].append(batch_indices)

file_name = model_path + model_name + '/Ltilde_' + model_name + f'/Ltilde_calib128_batch0(size8).pkl'
with open(file_name, 'rb') as f:
    hm = pickle.load(f)
ref_Ltilde_clado = hm['Ltilde']
ref_layer_index = hm['layer_index']
index2layerscheme = load_gradients(file_name)

# #naive
# file_path = model_path + model_name + '/Ltilde_' + model_name + "_naive-only"
# deltal_file_path = model_path + model_name + '/sensitivity/Ltilde_naive'
# for i, n_batch in enumerate(num_batches):
#     # print(f'selected indices for batch num: {n_batch}\n{selected_samples[i]}')
#     for run in range(n_runs):
#         deltal_multi_batches = []
#         n_samples = 8 * n_batch
#         for j in range(n_batch):
#             batch_id = selected_samples[i][run][j]
#             file_name = file_path + f'/Ltilde_calib128_batch{batch_id}(size8).pkl'
#             with open(file_name, 'rb') as f:
#                 hm = pickle.load(f)
#             deltal_multi_batches.append(hm['Ltilde'])
#         file_name = deltal_file_path + f'/sample_size{n_samples}/' + f'Ltilde_calib128_multi-batches_run{run+1}_bs8.pkl'
#         with open(file_name, 'wb') as f:
#             pickle.dump({'Ltilde': np.mean(deltal_multi_batches, axis=0), 'layer_index': ref_layer_index}, f)

# mpqco
file_path = model_path + model_name + '/DeltaL_MPQCO_' + model_name
deltal_file_path = model_path + model_name + '/sensitivity/DeltaL_mpqco'
for i, n_batch in enumerate(num_batches):
    # print(f'selected indices for batch num: {n_batch}\n{selected_samples[i]}')
    for run in range(n_runs):
        n_samples = 8 * n_batch
        batch_deltaLs_mpqco = []
        for batch_id in range(n_batch):
            b = selected_samples[i][run][batch_id]
            file_name = model_path + model_name + '/DeltaL_MPQCO_' + model_name + f'/MPQCO_DELTAL_batch{b}(size8).pkl'
            with open(file_name,'rb') as f:
                hm_mpqco = pickle.load(f)

            deltal = np.zeros(ref_Ltilde_clado.shape)

            for layer_id in range(len(index2layerscheme)):
                layer_name, scheme = index2layerscheme[layer_id]
                wbit = eval(scheme[:-4])[1]
                deltal[layer_id,layer_id] = hm_mpqco[layer_name][wbit]
            batch_deltaLs_mpqco.append(deltal) 
        file_name = deltal_file_path + f'/sample_size{n_samples}/' + f'DeltaL_calib128_multi-batches_run{run+1}_bs8.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump({'DeltaL': np.mean(batch_deltaLs_mpqco, axis=0), 'layer_index': ref_layer_index}, f)