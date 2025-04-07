import numpy as np
import matplotlib.pyplot as plt
import json, sys
import math
import copy

import graph_generator
import trackerapi
import math
import pickle
from sklearn.metrics import mean_squared_error

params = pickle.load(open('calibration.prms', 'rb'), fix_imports=True)
test_dataset = graph_generator.CalibrationDataset(sys.argv[1], params['net'], '1')



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
        while a > 180+100:
            a -= 360
        while a < -180+70:
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

with open(sys.argv[1], 'r') as f:
    raw = f.read()
raw = list(raw)

raws = ''.join(raw)
data = json.loads(raws)['data_set']


model = trackerapi.TrackerAPI('.', test_dataset)


x_gt = []
x_cc = []
x_nn = []
z_gt = []
z_cc = []
z_nn = []
a_gt = []
a_cc = []
a_nn = []
s_gt = []
s_cc = []
s_nn = []
c_gt = []
c_cc = []
c_nn = []


try:
    with open('kk', 'rb') as f:
        results = pickle.load(f)
except:
    results = [x for x in model.predict()]
    with open('kk', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
print(len(results), "-", len(data))
print(len(results[0]))
#print(results[0])
#print(data[0]['superbody'][0]['ground_truth'])

s = 0
ang_prev = 0

xyz_gtMSE = []
sc_gtMSE = []
xyz_nnMSE = []
sc_nnMSE = []


for n in range(int(len(results))):
    xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][0]/4000)
    xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][1]/4000)
    xyz_gtMSE.append(data[n]['superbody'][0]['ground_truth'][2]/4000)
    sc_gtMSE.append(math.sin(data[n]['superbody'][0]['ground_truth'][3])*0.7)
    sc_gtMSE.append(math.cos(data[n]['superbody'][0]['ground_truth'][3])*0.7)

    xyz_nnMSE.append(results[n][0])
    xyz_nnMSE.append(results[n][1])
    xyz_nnMSE.append(results[n][2])
    sc_nnMSE.append(results[n][3])
    sc_nnMSE.append(results[n][4])


    n_joints = 0
    for cam in range(len(data[n]['superbody'])):
       n_joints += len(data[n]['superbody'][cam]['joints'])

    if n_joints < 3:
        continue


    s += 1
 
    if s%1 == 0:
        i = n
        x_gt.append(data[i]['superbody'][0]['ground_truth'][0])
        z_gt.append(data[i]['superbody'][0]['ground_truth'][2])
        a_gt.append(rads2degrees(data[i]['superbody'][0]['ground_truth'][3]))
        s_gt.append(math.sin(data[i]['superbody'][0]['ground_truth'][3]))
        c_gt.append(math.cos(data[i]['superbody'][0]['ground_truth'][3]))


        val_x_cc = 0
        val_z_cc = 0
        ncams = 0
        for cam in range(0, len(data[i]['superbody'])):
            val_x_cc += data[i]['superbody'][cam]['world'][0]
            val_z_cc += data[i]['superbody'][cam]['world'][2]
            ncams += 1

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
                ncams += 1

        if ncams>0:
            val_a_cc = math.atan2(val_sin_cc/ncams, val_cos_cc/ncams)
            ang_prev = val_a_cc
        else:
            val_a_cc = ang_prev

        a_cc.append(rads2degrees(val_a_cc))

        s_cc.append(math.sin(data[i]['superbody'][0]['world'][3]))
        c_cc.append(math.cos(data[i]['superbody'][0]['world'][3]))
        x_nn.append(results[i][0]*4000)
        z_nn.append(results[i][2]*4000)
        a_nn.append(rads2degrees(math.atan2(results[i][3]/0.7, results[i][4]/0.7)))
        s_nn.append(results[i][3])
        c_nn.append(results[i][4])

#        if len(x_gt) > 500:
#            break

print("MSE xyz:", mean_squared_error(xyz_gtMSE, xyz_nnMSE))
print("MSE sin cos:", mean_squared_error(sc_gtMSE, sc_nnMSE))
print("MSE GLOBAL:", mean_squared_error(xyz_gtMSE + sc_gtMSE, xyz_nnMSE + sc_nnMSE))

xx = [x for x in range(len(x_gt))]
global_fig = plt.figure()

fig1 = plt.subplot(311)
plt.plot(xx, x_gt, 'g-', label='ground truth')
plt.plot(xx, x_cc, 'r-', label='analytical estimation', linestyle=(0, (1, 1)))
plt.plot(xx, x_nn, 'b-', label='GNN', linestyle='dashed')
plt.legend(prop={'size': 11})
fig1.set_ylabel('x (mm)')

fig2 = plt.subplot(312)
plt.plot(xx, z_gt, 'g-', label='ground truth')
plt.plot(xx, z_cc, 'r-', label='analytical estimation', linestyle=(0, (1, 1)))
plt.plot(xx, z_nn, 'b-', label='GNN', linestyle='dashed')
#plt.legend(prop={'size': 11})
fig2.set_ylabel('y (mm)')

fig3 = plt.subplot(313)
plt.plot(xx, xxx_degrees(a_gt), 'g-', label='ground truth')
plt.plot(xx, xxx_degrees(a_cc), 'r-', label='analytical estimation', linestyle=(0, (1, 1)))
plt.plot(xx, xxx_degrees(a_nn), 'b-', label='GNN', linestyle='dashed')
plt.gca().set_ylim([-90,280])
#plt.legend(prop={'size': 11})
fig3.set_ylabel('angle (degrees)')

#plt.subplot(414)
#plt.plot(xx, np.zeros((len(a_gt),1)), 'k-')
#plt.plot(xx, xxx_degrees(a_gt), 'g-', label='angle (ground truth)')
#plt.plot(xx, to_error(a_nn, a_cc), 'r-', label='error (analytical estimation)')
#plt.plot(xx, to_error(a_nn, a_gt), 'y-', label='error (GNN)')
#plt.legend()

#plt.subplot(413)
#plt.plot(xx, s_gt, 'g-', label='sin (ground truth)')
#plt.plot(xx, s_cc, 'r-', label='sin (manual)')
#plt.plot(xx, s_nn, 'b-', label='sin (GNN)')
#plt.legend()

#plt.subplot(414)
#plt.plot(xx, c_gt, 'g-', label='cos (ground truth)')
#plt.plot(xx, c_cc, 'r-', label='cos (manual)')
#plt.plot(xx, c_nn, 'b-', label='cos (GNN)')
#plt.legend()

global_fig.align_ylabels([fig1, fig2, fig3])
plt.tight_layout()
plt.show()
