import numpy as np 
from matplotlib import pyplot as plt 
plt.rc('lines', linewidth=3, color='r')
plt.rcParams.update({'font.size': 14})

UNIT_DENSITY=1.67262171e-27
Om=0.27
Ol=0.73
H0=70.e5/3.086e24
h=0.7
G=6.67e-8
dcr0=3.*H0*H0/(8.*np.pi*G)
Msun=2.e33

def Ezsqr(z):
    return Om*(1.+z)**3+(1.-Om)

def main():
    data2=np.loadtxt('12_0.txt')
    data2=np.transpose(data2)
    data3=np.loadtxt('12_1.txt')
    data3=np.transpose(data3)
    data4=np.loadtxt('15_0.txt')
    data4=np.transpose(data4)
    data5=np.loadtxt('15_1.txt')
    data5=np.transpose(data5)
    data6=np.loadtxt('14_0.txt')
    data6=np.transpose(data6)
    data7=np.loadtxt('14_1.txt')
    data7=np.transpose(data7)

    dcr_0=dcr0*Ezsqr(0)/UNIT_DENSITY
    dcr_1=dcr0*Ezsqr(1)/UNIT_DENSITY

    plt.loglog(data2[0],data2[1]/dcr_0,'b',label="$M_0=10^{12} M_\odot$")
    plt.loglog(data6[0],data6[1]/dcr_0,'k',label="$M_0=10^{14} M_\odot$")
    plt.loglog(data4[0],data4[1]/dcr_0,'r',label="$M_0=10^{15} M_\odot$")
    plt.loglog(data7[0],data7[1]/dcr_1,'k--')
    plt.loglog(data3[0],data3[1]/dcr_1,'b--')
    plt.loglog(data5[0],data5[1]/dcr_1,'r--')
    plt.xlabel("$r/R_{200c}$",fontsize=16)
    plt.ylabel(r"$\rho/\rho_{cr}$",fontsize=16)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()