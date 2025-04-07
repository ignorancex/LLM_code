import numpy as np
import matplotlib.pyplot as plt
import json, sys
import math
import copy

import graph_generator
import trackerapi
import math
import pickle

import torch
from torch.utils.data import DataLoader
from nets.mlpnet import MLPNet
from sklearn.metrics import mean_squared_error

def xxx_rads(a):
    def xxx_rads_item(a):
        while a > math.pi:
            a -= 2.*math.pi
        while a < -math.pi:
            a += 2.*math.pi
        return a
    if isinstance(a, list):
        return [xxx_rads_item(x) for x in a]
    return xxx_rads_item(a)

def xxx_degrees(a):
    def xxx_degrees_item(a):
        while a > 180:
            a -= 360
        while a < -180:
            a += 360
        return a
    if isinstance(a, list):
        return [xxx_degrees_item(x) for x in a]
    return xxx_degrees_item(a)

def rads2degrees(a):
    if a < 0:
        a += 2.*math.pi
    return a*180/math.pi


def to_error(got, ground_truth):
    e = copy.deepcopy(got)
    for i in range(len(e)):
        e[i] = xxx_degrees(e[i]-ground_truth[i])
    return e


def processData(filename):
    test_dataset = graph_generator.CalibrationDataset(filename, 'run','1')

    with open(filename, 'r') as f:
        raw = f.read()
    raw = list(raw)

    raws = ''.join(raw)
    data = json.loads(raws)['data_set'][:]

    USING = 'MLPNET'

    model_gnn = trackerapi.TrackerAPI('.', test_dataset)
    results_gnn = [x for x in model_gnn.predict()]
    train_2D = False # False->3D, True->2D
    model_mlp = MLPNet(train_2D)
    if train_2D:
        test_data = MLPNet.load_data_2D(filename)
    else:
        test_data = MLPNet.load_data(filename)
    X = torch.tensor(test_data['data'])
    Y = torch.tensor(test_data['groundtruth'])
    test_dataset = torch.utils.data.TensorDataset(X,Y)
    test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    print("mlp dataset len", len(test_loader.dataset))
    results_mlp = model_mlp.predict(test_loader)

    x_gt = []
    x_cc = []
    x_nn = []
    x_mlp = []
    z_gt = []
    z_cc = []
    z_nn = []
    z_mlp = []
    a_gt = []
    a_cc = []
    a_nn = []
    a_mlp = []

    xyz_gtMSE = []
    sc_gtMSE = []
    xyz_gnnMSE = []
    sc_gnnMSE = []
    xyz_mlpMSE = []
    sc_mlpMSE = []





    s = 0
    ang_prev = 0
    for n in range(int(len(results_gnn))):   
        xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][0]/4000)
        xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][1]/4000)
        xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][2]/4000)
        sc_gtMSE.append(math.sin(data[n]['superbody'][0]['ground_truth'][3])*0.7)
        sc_gtMSE.append(math.cos(data[n]['superbody'][0]['ground_truth'][3])*0.7)

        xyz_gnnMSE.append(results_gnn[n][0])
        xyz_gnnMSE.append(results_gnn[n][1])
        xyz_gnnMSE.append(results_gnn[n][2])
        sc_gnnMSE.append(results_gnn[n][3])
        sc_gnnMSE.append(results_gnn[n][4])

        xyz_mlpMSE.append(results_mlp[n][0])
        xyz_mlpMSE.append(results_mlp[n][1])
        xyz_mlpMSE.append(results_mlp[n][2])
        sc_mlpMSE.append(results_mlp[n][3])
        sc_mlpMSE.append(results_mlp[n][4])


        n_joints = 0
        for cam in range(len(data[n]['superbody'])):
           n_joints += len(data[n]['superbody'][cam]['joints'])

        if n_joints < 3:
            continue

        s += 1
 
        if s>=2 and s%1 == 0:
            i = n
            x_gt.append(data[i]['superbody'][0]['ground_truth'][0])
            z_gt.append(data[i]['superbody'][0]['ground_truth'][2])
            a_gt.append(rads2degrees(data[i]['superbody'][0]['ground_truth'][3]))

            val_x_cc = 0
            val_z_cc = 0
            ncams = 0
            for cam in range(0, len(data[i]['superbody'])):
                val_x_cc += data[i]['superbody'][cam]['world'][0]
                val_z_cc += data[i]['superbody'][cam]['world'][2]
                ncams+=1

            x_cc.append(val_x_cc/ncams)
            z_cc.append(val_z_cc/ncams)

            val_sin_cc = 0
            val_cos_cc = 0
            ncams = 0
            for cam in range(0, len(data[i]['superbody'])):
                joints = data[i]['superbody'][cam]["joints"]
                if ("right_shoulder" in joints and "left_shoulder" in joints) or ("right_hip" in joints and "left_hip" in joints):
                    val_sin_cc += math.sin(data[i]['superbody'][cam]['world'][3])
                    val_cos_cc += math.cos(data[i]['superbody'][cam]['world'][3])
                    ncams+=1

            if ncams>0:
                val_a_cc = math.atan2(val_sin_cc/ncams, val_cos_cc/ncams)
                ang_prev = val_a_cc
            else:
                val_a_cc = ang_prev


            a_cc.append(rads2degrees(val_a_cc))

            x_nn.append(results_gnn[i][0]*4000)
            z_nn.append(results_gnn[i][2]*4000)
            a_nn.append(rads2degrees(math.atan2(results_gnn[i][3]/0.7, results_gnn[i][4]/0.7)))
            x_mlp.append(results_mlp[i][0]*4000)
            z_mlp.append(results_mlp[i][2]*4000)
            a_mlp.append(rads2degrees(math.atan2(results_mlp[i][3]/0.7, results_mlp[i][4]/0.7)))
        
    error_dict = dict()

    print("dataset len", len(x_gt))

    err_x_cc = np.array([abs(x_cc[i]-x_gt[i]) for i in range(len(x_gt))]).mean()
    error_dict['err_x_cc'] = err_x_cc
    err_z_cc = np.array([abs(z_cc[i]-z_gt[i]) for i in range(len(z_gt))]).mean()
    error_dict['err_z_cc'] = err_z_cc
    list_err_a_cc = to_error(a_cc, a_gt)
    err_a_cc = np.array([abs(list_err_a_cc[i]) for i in range(len(a_gt))]).mean()
    error_dict['err_a_cc'] = err_a_cc

    err_x_nn = np.array([abs(x_nn[i].item()-x_gt[i]) for i in range(len(x_gt))]).mean()
    error_dict['err_x_nn'] = err_x_nn
    err_z_nn = np.array([abs(z_nn[i].item()-z_gt[i]) for i in range(len(z_gt))]).mean()
    error_dict['err_z_nn'] = err_z_nn
    list_err_a_nn = to_error(a_nn, a_gt)
    err_a_nn = np.array([abs(list_err_a_nn[i]) for i in range(len(a_gt))]).mean()
    error_dict['err_a_nn'] = err_a_nn

    err_x_mlp = np.array([abs(x_mlp[i].item()-x_gt[i]) for i in range(len(x_gt))]).mean()
    error_dict['err_x_mlp'] = err_x_mlp
    err_z_mlp = np.array([abs(z_mlp[i].item()-z_gt[i]) for i in range(len(z_gt))]).mean()
    error_dict['err_z_mlp'] = err_z_mlp
    list_err_a_mlp = to_error(a_mlp, a_gt)
    err_a_mlp = np.array([abs(list_err_a_mlp[i]) for i in range(len(a_gt))]).mean()
    error_dict['err_a_mlp'] = err_a_mlp

    print("---------------------------------------------")
    print("GNN MSE GLOBAL:", mean_squared_error(xyz_gtMSE + sc_gtMSE, xyz_gnnMSE + sc_gnnMSE))
    print("GNN MSE sin cos:", mean_squared_error(sc_gtMSE, sc_gnnMSE))
    print("GNN MSE xyz:", mean_squared_error(xyz_gtMSE, xyz_gnnMSE))

    print("---------------------------------------------")
    print("MLP MSE GLOBAL:", mean_squared_error(xyz_gtMSE + sc_gtMSE, xyz_mlpMSE + sc_mlpMSE))
    print("MLP MSE sin cos:", mean_squared_error(sc_gtMSE, sc_mlpMSE))
    print("MLP MSE xyz:", mean_squared_error(xyz_gtMSE, xyz_mlpMSE))

   
    return error_dict


if len(sys.argv)<2:
    print("Please, specify a json file")
    exit(0)

errorsList = []
labels = []

for i in range(1, len(sys.argv)):
    errors = processData(sys.argv[i])
    errorsList.append(errors)
    labels.append("S"+str(i))
#    print(i, "---CC---", "mean_err_x", errors['err_x_cc'], "mean_err_z", errors['err_z_cc'], "mean_err_a", errors['err_a_cc'])
#    print(i, "---NN---", "mean_err_x", errors['err_x_mlp'], "mean_err_z", errors['err_z_mlp'], "mean_err_a", errors['err_a_mlp'])


x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

plt.figure()

xError_cc = [errorsList[i]['err_x_cc'] for i in range(len(labels))]
xError_nn = [errorsList[i]['err_x_nn'] for i in range(len(labels))]
xError_mlp = [errorsList[i]['err_x_mlp'] for i in range(len(labels))]

ax = plt.subplot(131)
ax.set_ylim([0,200])
ax.bar(x - width/2, xError_cc, width, label='baseline')
ax.bar(x + width/2, xError_nn, width, label='GNN')
ax.bar(x + 1.5*width, xError_mlp, width, label='MLP')
ax.set_ylabel('Error (mm)')
ax.set_title('Error in position (X)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


zError_cc = [errorsList[i]['err_z_cc'] for i in range(len(labels))]
zError_nn = [errorsList[i]['err_z_nn'] for i in range(len(labels))]
zError_mlp = [errorsList[i]['err_z_mlp'] for i in range(len(labels))]

az = plt.subplot(132)
az.set_ylim([0,200])
az.bar(x - width/2, zError_cc, width, label='baseline')
az.bar(x + width/2, zError_nn, width, label='GNN')
az.bar(x + 1.5*width, zError_mlp, width, label='MLP')
az.set_ylabel('Error (mm)')
az.set_title('Error in position (Z)')
az.set_xticks(x)
az.set_xticklabels(labels)
az.legend()

aError_cc = [errorsList[i]['err_a_cc'] for i in range(len(labels))]
aError_nn = [errorsList[i]['err_a_nn'] for i in range(len(labels))]
aError_mlp = [errorsList[i]['err_a_mlp'] for i in range(len(labels))]

aa = plt.subplot(133)
aa.bar(x - width/2, aError_cc, width, label='baseline')
aa.bar(x + width/2, aError_nn, width, label='GNN')
aa.bar(x + 1.5*width, aError_mlp, width, label='MLP')
aa.set_ylabel('Error (degrees)')
aa.set_title('Error in orientation')
aa.set_xticks(x)
aa.set_xticklabels(labels)
aa.legend()

#with open('final_SimOnly_error_results_GAT2D', 'wb') as f:
#    pickle.dump(xError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(xError_nn, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(zError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(zError_nn, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(aError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(aError_nn, f, pickle.HIGHEST_PROTOCOL)

#with open('final_SimOnly_error_results__MLP2D', 'wb') as f:
#    pickle.dump(xError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(xError_mlp, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(zError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(zError_mlp, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(aError_cc, f, pickle.HIGHEST_PROTOCOL)
#    pickle.dump(aError_mlp, f, pickle.HIGHEST_PROTOCOL)


plt.show()


