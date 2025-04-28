import numpy as np
from sklearn.linear_model import LinearRegression
m=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
P=[0.9115734772261084,1.1856678944832488,1.5491163790750622,1.9092077283425182,2.225657720143466,2.486014696215505,2.691608897953176,2.849567869062738,2.9685893044529106,3.056992807612127,3.1219472776204498,3.1692778108239446,3.2035453214237313,3.2282313318498783,3.245945506007917,3.2586179365262704]
Pmod=[0.756250751206084,0.8473268459667701,1.1020104192452824,1.4268817888961787,1.7579906842023618,2.0611275640807323,2.322018221539106,2.537890612722478,2.7117735579825766,2.849154301869047,2.956145152622651,3.0385542280630222,3.1014814070385825,3.1491992476648254,3.1851786785514062,3.212179421839488]
lim=np.pi*np.pi/6
lim2=np.pi*np.sqrt(2+np.sqrt(2))

logs=[m[i]*(np.log(2)) for i in range(len(m))]
ps=[2**m[i]/10000 for i in range(len(m))]
ps2=[np.sqrt(ps[i]) for i in range(len(m))]
ps3=[2**(m[i]/5) for i in range(len(m))]

P1=[np.log(2)*m[i]+np.log(P[i]/2) for i in range(len(m))]
P2=[2**(m[i]-1)*P[i]/10000 for i in range(len(m))]
P3=[np.log(2)*m[i]+np.log(lim-P[i]/2) for i in range(len(m))]
P4=[np.exp(P3[i])/100 for i in range(len(m))]
P5=[np.log(2)*m[i]+np.log(P[i]/2 - lim +lim2/(2**(m[i]/2))) for i in range(len(m))]
P6=[np.exp(P5[i]) for i in range(len(m))]
P4mod=[(2**(m[i]/2))*(lim-Pmod[i]/2) for i in range(len(m))]
#x=np.array(logs[-3:]).reshape((-1, 1)) #use this one for log plots
#y=np.array(P1[-3:])
#x=np.array(ps[-3:]).reshape((-1, 1)) #use this one for P2
#y=np.array(P2[-3:])
#x=np.array(logs[-3:]).reshape((-1, 1)) #use this one for log plots
#y=np.array(P3[-3:])
#x=np.array(ps2[-3:]).reshape((-1, 1)) #use this one for P4
#y=np.array(P4[-3:])
#x=np.array(logs[-3:]).reshape((-1, 1)) #use this one for log plots
#y=np.array(P5[-3:])
#x=np.array(ps3[-3:]).reshape((-1, 1)) #use this one for P6
#y=np.array(P6[-3:])
x=np.array(logs[-3:]).reshape((-1, 1)) #use this one for log plots
y=np.array(P4mod[-3:])
reg = LinearRegression().fit(x,y)
#ax+b
a=reg.coef_[0]
b=reg.intercept_
print(a,b)
def line(x):
    return a*x+b
