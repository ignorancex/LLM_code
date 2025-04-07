import numpy as np
import json, sys
import math
import copy

import graph_generator
import trackerapi
import math
from getkey import getkey, keys
import pickle


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


with open(sys.argv[1], 'r') as f:
    raw = f.read()
raw = list(raw)

raws = ''.join(raw)
data = json.loads(raws)['data_set']

params = pickle.load(open('calibration.prms', 'rb'), fix_imports=True)
model = trackerapi.TrackerAPI('.', None)

stop = False
i = 0
while not stop:
    print("**************************************************")
    print("*************   PRESS q TO FINISH   **************")
    print("**************************************************")

    test_dataset = graph_generator.CalibrationDataset(data[i], 'run', '1')
    results = model.predictOneGraph(test_dataset)[0]

    x_gt = data[i]['superbody'][0]['ground_truth'][0]
    z_gt = data[i]['superbody'][0]['ground_truth'][2]
    a_gt = rads2degrees(data[i]['superbody'][0]['ground_truth'][3])
    
    x_nn = results[0].item()*4000
    z_nn = results[2].item()*4000
    a_nn = rads2degrees(math.atan2(results[3].item()/0.7, results[4].item()/0.7))

    eX = abs(x_gt - x_nn)
    eZ = abs(z_gt - z_nn)
    eA = xxx_degrees(a_gt - a_nn)

    print("Ground-truth:", "X", x_gt, "Z", z_gt, "Angle", a_gt)
    print("Net output  :", "X", x_nn, "Z", z_nn, "Angle", a_nn)
    print("Error       :", "X", eX, "Z", eZ, "Angle", eA)

    i += 1

    k = getkey()
    if k == 'q' or i == len(data):
        stop = True


