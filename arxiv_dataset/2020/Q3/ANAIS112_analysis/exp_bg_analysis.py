import numpy as np 
from scipy import optimize , stats
import nestle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#path=input('enter file path ')
path='data16.dat'
t=open(path, "r")
data1 = np.loadtxt(t)                    
t.close()
data1=np.transpose(data1)

#path=input('enter file path ')
path='data26.dat'
t=open(path, "r")
data2 = np.loadtxt(t)                    
t.close()
data2=np.transpose(data2)


def fit_bg(x,R0,R1,tau):
    	return R0 + R1*np.exp(-x/tau)

#background-only best fit (for each data set separately)
guess=[2,1,1000]

def chi2_bg(P,DATA):
        sigma=DATA[2]
        y_fit=fit_bg(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

bnd=((None,None),(None,None),(1,None))
bg1 = optimize.minimize(chi2_bg, guess,args=data1,method='SLSQP',bounds=bnd)
bg2= optimize.minimize(chi2_bg, guess,args=data2,method='SLSQP',bounds=bnd)


def log_likelihood_bg(P,DATA):
        y_fit=fit_bg(DATA[0],P[0],P[1],P[2])
        return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


#-----------------MODULATION-------------------


def fit_cosine(x,S,w,t_0,R0,R1,tau):
    	return R0 + R1*np.exp(-x/tau) + S*np.cos(w*(x+t_0))

def chi2_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2],P[3],P[4],P[5])
    sigma=DATA[2]
    r = (DATA[1] - y_fit)/sigma
    return np.sum(r**2)

def log_likelihood_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2],P[3],P[4],P[5])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))



guess1=[0,0.025,0,bg1.x[0],bg1.x[1],bg1.x[2]]
guess2=[0,0.025,0,bg2.x[0],bg2.x[1],bg2.x[2]]

#cos1 = optimize.minimize(chi2_cosine, guess1,args=data1,method='BFGS') 
#cos2 = optimize.minimize(chi2_cosine, guess2,args=data2,method='BFGS')


popt1, pcov1 =optimize.curve_fit(fit_cosine, data1[0], data1[1], p0=guess1, sigma=data1[2])

popt2, pcov2 = optimize.curve_fit(fit_cosine, data2[0], data2[1], p0=guess2, sigma=data2[2])



def dof_val(P,DATA):
  return len(DATA[0]) - len(P)

#chi squared likelihood function
def chi2L_cosine(P,DATA):
  chi2 = chi2_cosine(P,DATA)
  dof = dof_val(P,DATA)
  return stats.chi2(dof).pdf(chi2)

def chi2L_bg(k,DATA):
  chi2 = chi2_bg(k,DATA)
  dof = dof_val(k,DATA)
  return stats.chi2(dof).pdf(chi2)

def frequentist(P,k,DATA):
    cc=chi2_cosine(P,DATA)
    ck=chi2_bg(k,DATA)
    dofc=dof_val(P,DATA)
    dofk=dof_val(k,DATA)
    
    print("\nCosine : Amplitude=",'%.4f'%P[0],";  w=",'%.4f'%P[1],"/days ","; time period",'%.2f'%(2.0*np.pi/P[1]),"days",";  initial phase= ",'%.2f'%P[2]," days")
    print("bg= ",k)
    print("\nCosine :  Chi-Square likelihood:" , '%.4f'%chi2L_cosine(P,DATA)," ; Chi square value=",'%.4f'%cc,   " ;  Chi2/dof=",'%.4f'%cc,'/',dofc,
          "\nConstant :  Chi-Square likelihood:" , '%.4f'%chi2L_bg(k,DATA)," ; Chi square value=",'%.4f'%ck ,"  ;  chi2/dof=",'%.4f'%ck,'/',dofk)
    d=np.abs(cc-ck)
    print("difference in chi square values = ",'%.4f'%d)
    p=stats.chi2(dofk-dofc).sf(d)
    print ("p value=",'%.4f'%p)
    print("Confidence level : ",'%.4f'%stats.norm.isf(p),'\u03C3','\n')
    
def AIC(P,k,DATA):
    aic_const=chi2_bg(k,DATA) + 2*3
    aic_cosine=chi2_cosine(P,DATA) + 2*6
    del_aic= np.abs(aic_const-aic_cosine)
    print("AIC cosine=",'%.2f'%aic_cosine,", AIC const=",'%.2f'%aic_const)
    print ("diff in AIC values = ",'%.2f'%del_aic)

    
def BIC(P,k,DATA):
    bic_const=chi2_bg(k,DATA) + 3*np.log(len(DATA[0]))
    bic_cosine=chi2_cosine(P,DATA)  + 6*np.log(len(DATA[0]))
    del_bic= np.abs(bic_const-bic_cosine)
    print("BIC cosine=",'%.2f'%bic_cosine,", BIC const=",'%.2f'%bic_const)
    print ("diff in BIC values = ",'%.2f'%del_bic,'\n')


def plot(P1,P2,k1,k2):
       
    fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,4))
    
    plt.subplot(211)
    plt.xlim(-20,650)
    p1 = np.linspace(data1[0].min(),data1[0].max(),10000)   
    plt.plot(p1, fit_cosine(p1,P1[0],P1[1],P1[2],P1[3],P1[4],P1[5]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data1[0],data1[1],c='black',s=20)
    plt.plot(p1, fit_bg(p1,k1[0],k1[1],k1[2]),color='dodgerblue',lw=1,linestyle='--',label='$H_0$' ,linewidth=1.75)
    plt.grid(color='w')
    plt.errorbar(data1[0],data1[1],yerr = data1[2],fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='1-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.xlabel('Days after Aug 3, 2017',fontsize=14)
    plt.ylabel('Rate\n(cpd/kg/keV)',fontsize=14)
    
    
    plt.subplot(212)
    plt.xlim(-20,650)
    p2 = np.linspace(data2[0].min(),data2[0].max(),10000)
    plt.plot(p2, fit_cosine(p2,P2[0],P2[1],P2[2],P2[3],P2[4],P2[5]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data2[0],data2[1],c='black',s=20)
    plt.plot(p2, fit_bg(p2,k2[0],k2[1],k2[2]), color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data2[0],data2[1],yerr = data2[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='2-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.xlabel('Days after Aug 3, 2017',fontsize=14)
    plt.ylabel('Rate\n(cpd/kg/keV)',fontsize=14)
    plt.show()

    #plt.savefig('fig2.png')


print("\n1-6keV")
frequentist(popt1, bg1.x, data1)
AIC(popt1, bg1.x, data1)
BIC(popt1, bg1.x, data1)

print("\n2-6keV")
frequentist(popt2, bg2.x, data2)
AIC(popt2, bg2.x, data2)
BIC(popt2, bg2.x, data2)

#plot(popt1,popt2, bg1.x, bg2.x)

#==============================================================================

