import numpy as np
import matplotlib.pyplot as plt

def MaxMove(vect):
    imxm = np.argmax(np.abs(vect))
    val = vect[imxm]
    return val

    #return (val<0)*(-1) + (val>=0)*1.0    
    #return vect[0]

#transition time in ms
#0...H, ..., 5 ... L
TT = np.array([
[0  , 95.5  , 303.4  , 783.8  , 335.8  , 399.9],
[191.7  , 0  , 271.9  , 831.9  , 383.9  , 463.7],
[399.8  , 304.0  , 0  , 223.9  , 128.0  , 320.0],
[543.9  , 479.8  , 223.8  , 0  , 192.0  , 255.9],
[671.8  , 207.9  , 495.8  , 560.0  , 0  , 127.9],
[719.9  , 568.0  , 368.0  , 224.0  , 431.8  , 0],
])

#Voltage configuration
VT = np.array([
    [4.629, 6.594, 4.6785],
    [4.85175, 7.25325, 10.134],
    [5.3535, 7.42275, 5.022],
    [3.94725, 5.43225, 4.869],
    [4.66725, 8.09175, 2.613],
    [2.88075, 5.07, 7.59075],    
])

projList = ['H', 'V', 'D', 'A', 'R', 'L']

print("Stepsize-transition-time plot")
VdiffTra = np.zeros((6,6,3))
xy = []
legend = []
for i in range(6):
    for j in range(6):
        VdiffTra[i][j] = VT[j]-VT[i]
        if i != j:
            step = MaxMove(VT[j]-VT[i])
            xy.append((step, TT[i,j]))
            legend.append(f'{projList[i]}{projList[j]}')
xy = np.array(xy)

plt.plot(xy[:,0], xy[:,1],"o")
for i in range(len(xy)):
    plt.annotate(legend[i], xy[i])
plt.ylim(ymin=20)
plt.xlabel("Largest step [Vpp]")
plt.ylabel("Transition time [ms]")
plt.savefig('transitionanalysis.png')
plt.show()


print("Directionality analysis")
for i in range(6):
    for j in range(i+1,6):
        direct_V = VT[j] - VT[i]
        reverse_V = -direct_V
        key_dir = f'{projList[i]}>{projList[j]}'
        key_rev = f'{projList[j]}>{projList[i]}'
        direct_T = TT[i,j]
        reverse_T = TT[j,i]
        print("///")
        print(i,j)
        print(key_dir, direct_T, np.round(direct_V,1), MaxMove(direct_V))
        print(key_rev, reverse_T, np.round(reverse_V,1), MaxMove(reverse_V))



