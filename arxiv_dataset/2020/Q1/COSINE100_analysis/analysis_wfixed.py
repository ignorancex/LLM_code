# TIME PERIOD FIXED
# MODEL COMPARISON : Frequentist , AIC , BIC


import numpy as np 
from scipy import optimize , stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#f=input('enter path to data file : ')
data1 = np.loadtxt('crystal2.txt',delimiter=',')                    

#g=input('enter path to data file : ')
data2 = np.loadtxt('crystal3.txt',delimiter=',')                    

#h=input('enter path to data file : ')
data3 = np.loadtxt('crystal4.txt',delimiter=',')                    

#k=input('enter path to data file : ')
data4 = np.loadtxt('crystal6.txt',delimiter=',')                    

#l=input('enter path to data file : ')
data5 = np.loadtxt('crystal7.txt',delimiter=',')        

data=np.hstack((data1,data2,data3,data4,data5))


def fit_bg(x,c,p0,p1):
    	return c + p0*np.exp(-np.log(2)*x/p1)

#background-only best fit (for each data set separately)
guess=[0,1,1000]
def chi2(P,DATA):
        sigma1=DATA[3]
        y_fit1=fit_bg(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit1)/sigma1
        return np.sum(r**2)
#background fits
        
bnd=((1e-10,None),(1e-10,None),(1e-10,None))
cos1 = optimize.minimize(chi2, guess,args=data1,method='SLSQP',bounds=bnd)
cos2 = optimize.minimize(chi2, guess,args=data2,method='SLSQP',bounds=bnd)
cos3 = optimize.minimize(chi2, guess,args=data3,method='SLSQP',bounds=bnd)
cos4 = optimize.minimize(chi2, guess,args=data4,method='SLSQP',bounds=bnd)
cos5 = optimize.minimize(chi2, guess,args=data5,method='SLSQP',bounds=bnd)
        
    
def chi2_bg(P):
    
        y_fit1=fit_bg(data1[0],P[0],P[1],P[2])
        
        y_fit2=fit_bg(data2[0],P[3],P[4],P[5])
        
        y_fit3=fit_bg(data3[0],P[6],P[7],P[8])
        
        y_fit4=fit_bg(data4[0],P[9],P[10],P[11])
        
        y_fit5=fit_bg(data5[0],P[12],P[13],P[14])   
        
        sigma1,sigma2,sigma3,sigma4,sigma5= data1[3],data2[3],data3[3],data4[3],data5[3]

        y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
        sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))        

        r=(data[1]-y_fit)/sigma
        return np.sum(r**2)

def log_likelihood_bg(P):
    
    
        y_fit1=fit_bg(data1[0],P[0],P[1],P[2])
        
        y_fit2=fit_bg(data2[0],P[3],P[4],P[5])
        
        y_fit3=fit_bg(data3[0],P[6],P[7],P[8])
        
        y_fit4=fit_bg(data4[0],P[9],P[10],P[11])
        
        y_fit5=fit_bg(data5[0],P[12],P[13],P[14])   
        
        sigma1,sigma2,sigma3,sigma4,sigma5= data1[3],data2[3],data3[3],data4[3],data5[3]

        y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
        sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))

        return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))


#----------------------------------------------------------------------------

w=0.0172

def fit_cosine(x,c,p0,p1,A,t_0):
    	return c + p0*np.exp(-np.log(2)*x/p1) + A*np.cos(w*(x-t_0))

def chi2_cosine(P):
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[15],P[16])
    
    y_fit2=fit_cosine(data2[0],P[3],P[4],P[5],P[15],P[16])
    
    y_fit3=fit_cosine(data3[0],P[6],P[7],P[8],P[15],P[16])
    
    y_fit4=fit_cosine(data4[0],P[9],P[10],P[11],P[15],P[16])
    
    y_fit5=fit_cosine(data5[0],P[12],P[13],P[14],P[15],P[16])
    
    sigma1,sigma2,sigma3,sigma4,sigma5= data1[3],data2[3],data3[3],data4[3],data5[3]

    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    r = (data[1] - y_fit)/sigma
    return np.sum(r**2)

def log_likelihood_cosine(P):
    y_fit1=fit_cosine(data1[0],P[0],P[1],P[2],P[15],P[16])
    
    y_fit2=fit_cosine(data2[0],P[3],P[4],P[5],P[15],P[16])
    
    y_fit3=fit_cosine(data3[0],P[6],P[7],P[8],P[15],P[16])
    
    y_fit4=fit_cosine(data4[0],P[9],P[10],P[11],P[15],P[16])
    
    y_fit5=fit_cosine(data5[0],P[12],P[13],P[14],P[15],P[16])
    
    sigma1,sigma2,sigma3,sigma4,sigma5= data1[3],data2[3],data3[3],data4[3],data5[3]

    sigma=np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))            
    y_fit=np.hstack((y_fit1,y_fit2,y_fit3,y_fit4,y_fit5))
    
    return sum(stats.norm.logpdf(*args) for args in zip(data[1],y_fit,sigma))


#-----------------------------------------------------------------------------

def frequentist(cos_fin,k_fin):
    c1=chi2_cosine(cos_fin)
    c2=chi2_bg(k_fin)
    d=np.abs(c1-c2)
    print("chi2 value background=",c2, "    log likelihood bg=",log_likelihood_bg(k_fin))
    print("chi2 value bg+cosine modulation=",c1,"    log likelihood bg+cosine=",log_likelihood_cosine(cos_fin))
    print("difference in chi square values = ",d)
    p=stats.chi2(2).sf(d)
    print ("p value=",p)
    print("Confidence level : ",stats.norm.isf(p),'\u03C3','\n')
    print('bg chi2 pdf',  stats.chi2(180).pdf(c2))
    print('modulation chi2 pdf',  stats.chi2(178).pdf(c1))
    
def AIC(cos_fin,k_fin):
    aic_bg=chi2_bg(k_fin) + 2*15
    aic_cosine=chi2_cosine(cos_fin) +2*17
    del_aic= np.abs(aic_bg-aic_cosine)
    print("AIC cosine=",'%.2f'%aic_cosine,"AIC bg=",'%.2f'%aic_bg)
    print ("diff in AIC values = ",'%.2f'%del_aic)
    
def BIC(cos_fin,k_fin):
    bic_bg=chi2_bg(k_fin) + 15*np.log(len(data[0]))
    bic_cosine=chi2_cosine(cos_fin) +17*np.log(len(data[0]))
    del_bic= np.abs(bic_bg-bic_cosine)
    print("BIC cosine=",'%.2f'%bic_cosine,"BIC bg=",'%.2f'%bic_bg)
    print ("diff in BIC values = ",'%.2f'%del_bic)

#-----------------------------------------------------------------------------

def plot(bg_fit, cosine_fit):
       
    fig,ax=plt.subplots(nrows=5,ncols=1,figsize=(15,15))
    
    
    plt.subplot(511)
    p1 = np.linspace(data1[0].min(),data1[0].max(),10000)   
    plt.plot(p1, fit_cosine(p1,cosine_fit[0],cosine_fit[1],cosine_fit[2],
                            cosine_fit[15],cosine_fit[16]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data1[0],data1[1],c='black',s=20)
    plt.plot(p1, np.ones(10000)*fit_bg(p1,bg_fit[0],bg_fit[1],bg_fit[2]),
                color='dodgerblue',lw=1,linestyle='--',label='$H_0$' ,linewidth=1.75)
    plt.grid(color='w')
    plt.errorbar(data1[0],data1[1],yerr = data1[3],xerr=data1[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.25,"Crystal 2",fontsize=15)
    
    
    plt.subplot(512)
    p2 = np.linspace(data2[0].min(),data2[0].max(),10000)
    plt.plot(p2, fit_cosine(p2,cosine_fit[3],cosine_fit[4],cosine_fit[5],
                            cosine_fit[15],cosine_fit[16]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data2[0],data2[1],c='black',s=20)
    plt.plot(p2, np.ones(10000)*fit_bg(p2,bg_fit[3],bg_fit[4],bg_fit[5]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data2[0],data2[1],yerr = data2[3],xerr=data2[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.55,"Crystal 3",fontsize=15)
    
        
    plt.subplot(513)
    p3 = np.linspace(data3[0].min(),data3[0].max(),10000)
    plt.plot(p3, fit_cosine(p3,cosine_fit[6],cosine_fit[7],cosine_fit[8],
                            cosine_fit[15],cosine_fit[16]), color = 'red' ,label='$H_1$',linewidth=1.6)
    plt.scatter(data3[0],data3[1],c='black',s=20)
    plt.plot(p3, np.ones(10000)*fit_bg(p3,bg_fit[6],bg_fit[7],bg_fit[8]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data3[0],data3[1],yerr = data3[3],xerr=data3[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,3.5,"Crystal 4",fontsize=15)
    
        
    plt.subplot(514)
    p4 = np.linspace(data4[0].min(),data4[0].max(),10000)
    plt.plot(p4, fit_cosine(p4,cosine_fit[9],cosine_fit[10],cosine_fit[11],
                            cosine_fit[15],cosine_fit[16]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data4[0],data4[1],c='black',s=20)
    plt.plot(p4, np.ones(10000)*fit_bg(p4,bg_fit[9],bg_fit[10],bg_fit[11]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data4[0],data4[1],yerr = data4[3],xerr=data4[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,2.7,"Crystal 6",fontsize=15)
    
    
    plt.subplot(515)
    p5 = np.linspace(data5[0].min(),data5[0].max(),10000)
    plt.plot(p5, fit_cosine(p5,cosine_fit[12],cosine_fit[13],cosine_fit[14],
                            cosine_fit[15],cosine_fit[16]), color = 'red',label='$H_1$',linewidth=1.6 )
    plt.scatter(data5[0],data5[1],c='black',s=20)
    plt.plot(p5, np.ones(10000)*fit_bg(p5,bg_fit[12],bg_fit[13],bg_fit[14]),
                 color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data5[0],data5[1],yerr = data5[3],xerr=data5[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    plt.text(830,2.75,"Crystal 7",fontsize=15)
    
    
    fig.text(0.5, 0.085, '\nTime (days)', ha='center',fontweight='bold',color='dimgrey',fontsize=17)
    fig.text(0.075, 0.5, '2-6 keV event rate(cpd/kg/keV)',va='center',rotation='vertical',fontweight='bold',fontsize=17,color='dimgrey')
        
   # plt.savefig("__fig2__.png")
#============================================================================

guess_w=[0.1,100]
guess=np.hstack((cos1.x,cos2.x,cos3.x,cos4.x,cos5.x,guess_w))
bnd=((1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(1e-10,None),(None,None))
res = optimize.minimize(chi2_cosine, guess,method='SLSQP',bounds=bnd)

cosine_fit=res.x

#background-only fit by chi-sq minimization
bg_est=np.hstack((cos1.x,cos2.x,cos3.x,cos4.x,cos5.x))

# Model Comparison
frequentist(cosine_fit,bg_est)

AIC(cosine_fit,bg_est)

BIC(cosine_fit,bg_est)
