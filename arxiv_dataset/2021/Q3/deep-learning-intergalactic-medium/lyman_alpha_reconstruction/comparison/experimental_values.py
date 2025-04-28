from ..reconstruction_methods import *
from ..data_preprocessing import fluxTest, tauTest

import torch
import numpy as np
import os
import csv
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def experimentalRMSEs(instance, noise):
    noiseGen =  torch.distributions.normal.Normal(0.0, signalRMS*noiseDict[noise], validate_args=None)
    fluxsource = fluxTest
    tausource = tauTest
    predictionList = np.zeros((512*12996))
    with torch.no_grad():
        for midpoint in range(512):
            fluxs = torch.cat((fluxsource[:,:,midpoint-256:],fluxsource[:,:,:midpoint-256]),2)
            fluxs += noiseGen.sample(fluxs.shape)
            prediction = instance.net(fluxs)
            predictionList[midpoint::512] = prediction.flatten().data.cpu().numpy()
    instance.net.train()
    a = torch.flatten(tausource[:,0,:]).cpu().numpy()
    b = predictionList
    
    naive = -torch.log(fluxsource)
    
    logp = np.zeros((512*12996))
    
    for i in range(12996):
        if i % 100 == 99:
            print(i)
        logp[i*512:(i+1)*512] = log_predictor(fluxsource[i,0] + noiseGen.sample((512,)))
        
    smoothlogp = np.zeros((512*12996))
    fluxs = gaussian_filter1d(fluxsource.cpu().numpy() + noiseGen.sample(fluxsource.shape).cpu().numpy(),
                              6, axis=-1, mode="wrap")
    for i in range(12996):
        if i % 100 == 99:
            print(i)
        smoothlogp[i*512:(i+1)*512] = log_predictor(fluxs[i,0])
        
    c = a/b
        
    def cubicFit(x, a, b, c, d):
        return a*(x**3) + b*(x**2) + c*x + d

    cubicopt, cubiccov = curve_fit(cubicFit, a[a > 2], c[a > 2], bounds=([-100,-100,-100,-100], [100,100,100,100]))
    print(cubicopt)
    plt.figure()
    plt.scatter(a[a > 2],c[a > 2])
    xs = np.linspace(2,a.max(),100)
    plt.plot(xs, cubicFit(xs,*cubicopt), "r-")
    plt.show()
        
    curvedNN = np.copy(b)
    curvedNN[a>2] = b[a>2] * cubicFit(a[a>2], *cubicopt)
    
    print("Total")
    print("SmoothedLog: " + str(np.sqrt(np.mean((smoothlogp - a)**2))))
    smoothed_tot = np.sqrt(np.mean((smoothlogp - a)**2))
    print("Log: " + str(np.sqrt(np.mean((logp - a)**2))))
    log_tot = np.sqrt(np.mean((logp - a)**2))
    print("NeuralNetwork: " + str(np.sqrt(np.mean((b - a)**2))))
    nn_tot = np.sqrt(np.mean((b - a)**2))
    print("CurvedNN: " + str(np.sqrt(np.mean((curvedNN - a)**2))))
    curvednn_tot = np.sqrt(np.mean((curvedNN - a)**2))
    print("Low")
    print("SmoothedLog: " + str(np.sqrt(np.mean((smoothlogp[a<2] - a[a<2])**2))))
    smoothed_low = np.sqrt(np.mean((smoothlogp[a<2] - a[a<2])**2))
    print("Log: " + str(np.sqrt(np.mean((logp[a<2] - a[a<2])**2))))
    log_low = np.sqrt(np.mean((logp[a<2] - a[a<2])**2))
    print("NeuralNetwork: " + str(np.sqrt(np.mean((b[a<2] - a[a<2])**2))))
    nn_low = np.sqrt(np.mean((b[a<2] - a[a<2])**2))
    print("CurvedNN: " + str(np.sqrt(np.mean((curvedNN[a<2] - a[a<2])**2))))
    curvednn_low = np.sqrt(np.mean((curvedNN[a<2] - a[a<2])**2))
    print("High")
    print("SmoothedLog: " + str(np.sqrt(np.mean((smoothlogp[a>2] - a[a>2])**2))))
    smoothed_high = np.sqrt(np.mean((smoothlogp[a>2] - a[a>2])**2))
    print("Log: " + str(np.sqrt(np.mean((logp[a>2] - a[a>2])**2))))
    log_high = np.sqrt(np.mean((logp[a>2] - a[a>2])**2))
    print("NeuralNetwork: " + str(np.sqrt(np.mean((b[a>2] - a[a>2])**2))))
    nn_high = np.sqrt(np.mean((b[a>2] - a[a>2])**2))
    print("CurvedNN: " + str(np.sqrt(np.mean((curvedNN[a>2] - a[a>2])**2))))
    curvednn_high = np.sqrt(np.mean((curvedNN[a>2] - a[a>2])**2))
    print("Naive: "+ str(np.sqrt(np.mean((naive.cpu().numpy().flatten() - a)**2))))
    
    if not (os.path.isfile('statistics.csv')):
        with open('statistics.csv', 'x') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Noise','Smoothed_Tot','Log_Tot','NN_Tot','CurvedNN_Tot','Smoothed_Low','Log_Low','NN_Low','CurvedNN_Low','Smoothed_High','Log_High','NN_High','CurvedNN_High'])
    
    with open('statistics.csv','a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([noise, smoothed_tot, log_tot, nn_tot, curvednn_tot, smoothed_low, log_low, nn_low, curvednn_low, smoothed_high, log_high, nn_high, curvednn_high])