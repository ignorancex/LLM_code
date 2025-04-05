
import numpy as np 
from scipy import optimize , stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')


path=input('enter file path ')
#path='fig2_16.dat'
t=open(path, "r")
data1 = np.loadtxt(t)                    
t.close()
data1=np.transpose(data1)

path=input('enter file path ')
#path='fig2_26.dat'
t=open(path, "r")
data2 = np.loadtxt(t)                    
t.close()
data2=np.transpose(data2)


#-----------------MODULATION-------------------

def fit_cosine(x,S,w,t_0):
    	return S*np.cos(w*(x+t_0))

def log_likelihood_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_cosine(P,DATA):
        sigma=DATA[2]
        y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#degrees of freedom
def dof_val(P,DATA):
  return len(DATA[0]) - len(P)

#chi squared likelihood function
def chi2L_cosine(P,DATA):
  chi2 = chi2_val_cosine(P,DATA)
  dof = dof_val(P,DATA)
  return stats.chi2(dof).pdf(chi2)


#=========================constant background function==========================
def fit_const(x,k):
    	return k[0]*x**0

def log_likelihood_const(k,DATA):
    y_fit=fit_const(DATA[0],k)
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_const(k,DATA):
        sigma=DATA[2]
        y_fit=fit_const(DATA[0],k)
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#chi squared likelihood function
def chi2L_const(k,DATA):
  chi2 = chi2_val_const(k,DATA)
  dof = dof_val(k,DATA)
  return stats.chi2(dof).pdf(chi2)

def frequentist(P,k,DATA):
    cc=chi2_val_cosine(P,DATA)
    ck=chi2_val_const(k,DATA)
    dofc=dof_val(P,DATA)
    dofk=dof_val(k,DATA)
    
    print("\nCosine : Amplitude=",'%.4f'%P[0],";  w=",'%.4f'%P[1],"/days ","; time period",'%.2f'%(2.0*np.pi/P[1]),"days",";  initial phase= ",'%.2f'%P[2]," days")
    print("Constant k= ",'%.4f'%k[0])
    print("\nCosine :  Chi-Square likelihood:" , '%.4f'%chi2L_cosine(P,DATA)," ; Chi square value=",'%.4f'%cc,   " ;  Chi2/dof=",'%.4f'%cc,'/',dofc,
          "\nConstant :  Chi-Square likelihood:" , '%.4f'%chi2L_const(k,DATA)," ; Chi square value=",'%.4f'%ck ,"  ;  chi2/dof=",'%.4f'%ck,'/',dofk)
    d=np.abs(cc-ck)
    print("difference in chi square values = ",'%.4f'%d)
    p=stats.chi2(dofk-dofc).sf(d)
    print ("p value=",'%.4f'%p)
    print("Confidence level : ",'%.4f'%stats.norm.isf(p),'\u03C3','\n')
    
def AIC(P,k,DATA):
    aic_const=chi2_val_const(k,DATA) + 2
    aic_cosine=chi2_val_cosine(P,DATA) + 2*3
    del_aic= np.abs(aic_const-aic_cosine)
    print("AIC cosine=",'%.2f'%aic_cosine,", AIC const=",'%.2f'%aic_const)
    print ("diff in AIC values = ",'%.2f'%del_aic)

    
def BIC(P,k,DATA):
    bic_const=chi2_val_const(k,DATA) + np.log(len(DATA[0]))
    bic_cosine=chi2_val_cosine(P,DATA)  + 3*np.log(len(DATA[0]))
    del_bic= np.abs(bic_const-bic_cosine)
    print("BIC cosine=",'%.2f'%bic_cosine,", BIC const=",'%.2f'%bic_const)
    print ("diff in BIC values = ",'%.2f'%del_bic,'\n')

    
def plot(P1,P2,k1,k2):
       
    fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,4))
    
    
    plt.subplot(211)
    plt.xlim(-20,650)
    p1 = np.linspace(data1[0].min(),data1[0].max(),10000)   
    plt.plot(p1, fit_cosine(p1,P1[0],P1[1],P1[2]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data1[0],data1[1],c='black',s=20)
    plt.plot(p1, np.ones(10000)*k1[0],color='dodgerblue',lw=1,linestyle='--',label='$H_0$' ,linewidth=1.75)
    plt.grid(color='w')
    plt.errorbar(data1[0],data1[1],yerr = data1[2],fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='1-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.xlabel('Days after Aug 3, 2017',fontsize=14)
    plt.ylabel('Rate\n(cpd/kg/keV)',fontsize=14)    
    
    plt.subplot(212)
    plt.xlim(-20,650)
    plt.ylim(-0.1,0.1)
    p2 = np.linspace(data2[0].min(),data2[0].max(),10000)
    plt.plot(p2, fit_cosine(p2,P2[0],P2[1],P2[2]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data2[0],data2[1],c='black',s=20)
    plt.plot(p2, np.ones(10000)*k2[0], color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data2[0],data2[1],yerr = data2[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='2-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.xlabel('Days after Aug 3, 2017',fontsize=14)
    plt.ylabel('Rate\n(cpd/kg/keV)',fontsize=14)    
    #plt.savefig('fig.png')


#==============================================================================

g=[0]

k1 = optimize.minimize(chi2_val_const, g,args=data1,method='SLSQP')
k2 = optimize.minimize(chi2_val_const, g,args=data2,method='SLSQP')

guess=[0,0.0172,0]

bounds=([-np.inf,2.*np.pi/545,-np.inf], [np.inf, 2.*np.pi/10.0, np.inf])
popt1, pcov1 =optimize.curve_fit(fit_cosine, data1[0], data1[1], p0=guess, sigma=data1[2],bounds=bounds)

popt2, pcov2 = optimize.curve_fit(fit_cosine, data2[0], data2[1], p0=guess, sigma=data2[2],bounds=bounds)


print("H_0 : No modulation, constant fit")
print("H_1 : Cosine modulation")

print("\n1-6keV")
frequentist(popt1, k1.x, data1)
AIC(popt1, k1.x, data1)
BIC(popt1, k1.x, data1)

print("\n2-6keV")
frequentist(popt2, k2.x, data2)
AIC(popt2, k2.x, data2)
BIC(popt2, k2.x, data2)


#plot(popt1, popt2, k1.x, k2.x)
