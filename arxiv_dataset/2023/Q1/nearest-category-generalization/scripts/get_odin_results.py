import os

from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import ParameterGrid
import numpy as np
import torch
from torch.autograd import Variable
import joblib
from tqdm import tqdm

from lolip.variables import auto_var
from lolip.models.torch_utils import archs

def get_score(net1, device, loader, noiseMagnitude1=0.0014, temper=1000):
    criterion = torch.nn.CrossEntropyLoss()
    ret = []
########################################In-distribution###########################################
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        images  = data[0]

        inputs = Variable(images.to(device), requires_grad=True)
        outputs = net1(inputs)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        #nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        #f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs, 1)
        labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Adding small perturbations to images
        #tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        tempInputs = inputs.data - (noiseMagnitude1 * gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        #nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        #in_ret.append((temper, noiseMagnitude1, np.max(nnOutputs, 1)))
        ret.append(np.max(nnOutputs, 1))
    return np.concatenate(ret)

batch_size = 32
in_dataset = "cifar10wo0"
#in_dataset = "cifar10wo4"
n_classes = 9
#in_dataset = "cifar100coarsewo0"
#in_dataset = "cifar100coarsewo4"
#n_classes = 19
model_path = f"./models/out_of_sample/pgd-64-{in_dataset}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"
device = "cuda"
architecture = "WRN_40_10"
model = getattr(archs, architecture)(n_classes=n_classes, n_channels=3)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.to(device)
model.eval()

X, y, tstX, _, rest = auto_var.get_var_with_argument("dataset", in_dataset)
oodX = np.concatenate((rest[0], rest[1]), axis=0)

print(in_dataset)
#for out_dataset in ["mnist", "svhn", "tinyimgnet"]:
for out_dataset in ["svhn", "tinyimgnet"]:
    print(out_dataset)

    #temp_candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    #noise_candidates = [i for i in np.arange(0, 0.004, 0.004/21)]
    temp_candidates = [10, 100, 1000]
    noise_candidates = [i for i in np.arange(0, 0.004, 0.004/11)]
    params = ParameterGrid({
        'temp': temp_candidates,
        'noise': noise_candidates,
    })

    output_path = f"./results/notebooks_detection/ODIN_{in_dataset}_{out_dataset}.pkl"
    #if os.path.exists(output_path):
    #    temp = joblib.load(output_path)
    #    if len(temp) == len(params):
    #        continue
    ood2X, ood2y, ood3X, _ = auto_var.get_var_with_argument("dataset", out_dataset)
    if out_dataset == "mnist":
        ood2X = np.concatenate([ood2X] * 3, axis=3)
        ood3X = np.concatenate([ood3X] * 3, axis=3)
    ood2X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood2X], axis=0)
    ood3X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood3X], axis=0)

    dset = torch.utils.data.TensorDataset(torch.from_numpy(X.transpose(0, 3, 1, 2)).float())
    tr_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dset = torch.utils.data.TensorDataset(torch.from_numpy(tstX.transpose(0, 3, 1, 2)).float())
    ts_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dset = torch.utils.data.TensorDataset(torch.from_numpy(oodX.transpose(0, 3, 1, 2)).float())
    ood_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dset = torch.utils.data.TensorDataset(torch.from_numpy(ood2X.transpose(0, 3, 1, 2)).float())
    ood2_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dset = torch.utils.data.TensorDataset(torch.from_numpy(ood3X.transpose(0, 3, 1, 2)).float())
    ood3_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

    all_results = []
    for param in tqdm(params):
        temp = (
            get_score(model, device, tr_loader, noiseMagnitude1=param['noise'], temper=param['temp']),
            get_score(model, device, ts_loader, noiseMagnitude1=param['noise'], temper=param['temp']),
            get_score(model, device, ood_loader, noiseMagnitude1=param['noise'], temper=param['temp']),
            get_score(model, device, ood2_loader, noiseMagnitude1=param['noise'], temper=param['temp']),
            get_score(model, device, ood3_loader, noiseMagnitude1=param['noise'], temper=param['temp']),
            param,
        )
        all_results.append(temp)

        if len(all_results) % 10 == 0:
            joblib.dump(all_results, output_path)

    joblib.dump(all_results, output_path)
