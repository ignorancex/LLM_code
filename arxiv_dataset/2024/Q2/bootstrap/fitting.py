import numpy as np
from scipy.optimize import curve_fit
m=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
logs=[m[i]*(np.log(2)) for i in range(len(m))]
P=[0.9115734772261084,1.1856678944832488,1.5491163790750622,1.9092077283425182,2.225657720143466,2.486014696215505,2.691608897953176,2.849567869062738,2.9685893044529106,3.056992807612127,3.1219472776204498,3.1692778108239446,3.2035453214237313,3.2282313318498783,3.245945506007917,3.2586179365262704]
Pmod=[0.756250751206084,0.8473268459667701,1.1020104192452824,1.4268817888961787,1.7579906842023618,2.0611275640807323,2.322018221539106,2.537890612722478,2.7117735579825766,2.849154301869047,2.956145152622651,3.0385542280630222,3.1014814070385825,3.1491992476648254,3.1851786785514062,3.212179421839488]
MCmod=[0.4444444444444444,0.398406374501992,0.31645569620253167,0.25510204081632654,0.20040080160320642,0.15408320493066255,0.06402048655569782,0.023573785950023574,0.006145148405333989,0.0008026004253782255,0.00004145576048519822,0.000003090020962702211]
n_MC=len(MCmod)
logs_MC=[np.log(2)*(2+i/5) for i in range(n_MC)]

lim=np.pi*np.pi/6
lim2=np.pi*np.sqrt(2+np.sqrt(2))
P1=[np.log(2)*m[i]+np.log(P[i]/2) for i in range(len(m))]
P1mod=[np.log(2)*m[i]+np.log(Pmod[i]/2) for i in range(len(m))]
P3=[np.log(2)*m[i]+np.log(lim-P[i]/2) for i in range(len(m))]
P3mod=[np.log(2)*m[i]+np.log(lim-Pmod[i]/2) for i in range(len(m))]
P4=[np.exp(P3[i])/100 for i in range(len(m))]
P4mod=[(2**(m[i]/2))*(lim-Pmod[i]/2) for i in range(len(m))]
P5=[np.log(2)*m[i]+np.log(P[i]/2 - lim +lim2/(2**(m[i]/2))) for i in range(len(m))]
P5mod=[np.log(2)*m[i]+np.log(-Pmod[i]/2 + lim -logs[i]*np.sqrt(2+np.sqrt(2))/(2*(2**(m[i]/2)))) for i in range(len(m))]
P6=[np.exp(P5[i]) for i in range(len(m))]
P6mod=[np.exp(P5mod[i]-logs[i]/2) for i in range(len(m))]
P7=[2**(m[i])*(P[i]/2 - lim +lim2/(2**(m[i]/2))-5.0765/(2**(4*m[i]/5))) for i in range(len(m))]
Pmod_MC=-np.log(np.array(MCmod))
P4mod_MC=[(2**((2+i/5)/2))*(2*lim+np.log(MCmod[i])/(2**(2+i/5))) for i in range(n_MC)]

def summed(x,ll,fit):
    out=0
    for i in range(n):
        out+=(fun(x,ll[i],fit[i]))**2
    return out

##1-scaling
def fun4(x, t, y):
    return x[0]*t+x[1]+x[2]*np.exp(-x[3]*t) - y
def jac4(ll,a,c,d,e):
    return np.transpose((ll,np.array([1]*len(ll)),np.exp(-ll*e),-d*ll*np.exp(-ll*e)))
def func4(ll, a,c,d,e):
    return a*ll+c+d*np.exp(-ll*e)
fun=fun4
jac=jac4
func=func4
n=5
bounds=([0,-3,-5, 0.1],[2,3,5,1])
FIT=P1
#[ 0.99989769  0.4993439  -2.9494712   0.48275621] 1.504895126503702e-13
FIT=P1mod
#[ 0.99973411  0.50241917 -3.61907411  0.42058282] 6.825757727439676e-14

##1-const
def fun31(x, t, y):
    return t+x[0]+x[1]*np.exp(-x[2]*t) - y
def jac31(ll,c,d,e):
    return np.transpose((np.array([1]*len(ll)),np.exp(-ll*e),-d*ll*np.exp(-ll*e)))
def func31(ll, c,d,e):
    return ll+c+d*np.exp(-ll*e)
fun=fun31
jac=jac31
func=func31
n=4
bounds=([-3,-5, 0.1],[3,5,1])
FIT=P1
#[ 0.49781852 -3.07098771  0.48894981] 1.93044063021315e-12
FIT=P1mod
#[ 0.49829817 -3.76690503  0.42733905] 1.0556612809584475e-11

##2-scaling
def fun5(x, t, y):
    return x[0]*t+x[1]*np.log(t)+x[2]+x[3]*np.exp(-x[4]*t) - y
def jac5(ll,a,b,c,d,e):
    return np.transpose((ll,np.log(ll),np.array([1]*len(ll)),np.exp(-ll*e),-d*ll*np.exp(-ll*e)))
def func5(ll, a, b,c,d,e):
    return a*ll+b*np.log(ll)+c+d*np.exp(-ll*e)
fun=fun5
jac=jac5
func=func5
n=6
FIT=P3
bounds=([0.4,-0.2,1,-3, 0.1],[0.6,0.2,2,0,1])
#[ 0.49693271  0.05935735  1.63900364 -1.18496349  0.36321974] 1.8104827063192395e-12
FIT=P3mod
bounds=([0.45,0.9,-0.1,0.4, 0.1],[0.55,1.1,0.1,0.7,0.2])
#There is a very annoying minimum at [ 0.50750008  0.70720451  0.81111754 -0.55427983  0.61428907] 1.7141173478306365e-12. Still, the actual one is a bit lower: [ 0.49609068  1.04569387 -0.01624505  0.62020804  0.13494828] 4.3138423966229647e-13

##2-constant
def fun32(x, t, y):
    return t/2+x[0]+x[1]*np.exp(-x[2]*t) - y
def jac32(ll,c,d,e):
    return np.transpose((np.array([1]*len(ll)),np.exp(-ll*e),-d*ll*np.exp(-ll*e)))
def func32(ll, c,d,e):
    return ll/2+c+d*np.exp(-ll*e)
FIT=P3
fun=fun32
jac=jac32
func=func32
n=4
bounds=([1.5,-5, 0.1],[2,1,5])
#[ 1.75690954 -1.16656434  0.32946557] 1.0522963029604813e-11

def fun42(x, t, y):
    return x[0]*t+x[1]+x[2]*np.exp(-x[3]*t) - y
def jac42(ll,a,b,c,d):
    return np.transpose((ll,np.array([1]*len(ll)),np.exp(-ll*d),-c*ll*np.exp(-ll*d)))
def func42(ll, a,b,c,d):
    return a*ll+b+c*np.exp(-ll*d)
FIT=P4mod
fun=fun42
jac=jac42
func=func42
n=5
bounds=([0,-5, -5,0.1],[3,5,5,1])
#[ 0.9241962   3.26415026 -3.75474489  0.31544702] 3.530640782239311e-12


##3-const
print(P6[-1]/(2**(m[-1]/5)))
#5.0764803221695125
def fun33(x, t, y):
    return x[0]+x[1]*np.exp(-x[2]*t) - y
def jac33(ll,b,c,d):
    return np.transpose((np.array([1]*len(ll)),np.exp(-ll*d),-c*ll*np.exp(-ll*d)))
def func33(ll, b,c,d):
    return b+c*np.exp(-ll*d)
FIT=P6mod
fun=fun33
jac=jac33
func=func33
n=4
bounds=([-5,-5,0],[5,5,1])
#[ 3.26956388 -3.71608336  0.31301783] 1.649437541613924e-11


xdata = np.array(logs[-n:])
ydata = np.array(FIT[-n:])
popt, _ = curve_fit(func, xdata, ydata,jac=jac,bounds=bounds)
print(popt,summed(popt,xdata,ydata))
