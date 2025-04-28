import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from foreground_map_visual import plot_healpy

def beta_noreg(map_nu1, map_nu2, T_cmb, nu1, nu2):
    power_nu1 = np.load(map_nu1)
    power_nu1 = hp.ud_grade(power_nu1, 128)
    power_nu2 = np.load(map_nu2)
    power_nu2 = hp.ud_grade(power_nu2, 128)
    tcmb = T_cmb
    numr = (power_nu1 - tcmb)/(power_nu2-tcmb)
    denr = nu2/nu1
    beta = np.log10(numr)/np.log10(denr)
    return beta

def beta_reg(beta, N, min_val, max_val, show=False):
    plt.clf()
    plot_healpy(data=beta, min_val=min_val, max_val=max_val, cbar=None, cmap=plt.cm.get_cmap('inferno', N))
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    # fig.colorbar(image, ax=ax, orientation='horizontal', aspect=50, extend='both', pad=0.03)
    if show==True:
        plt.show()

if __name__=='__main__':
    beta = beta_noreg(map_nu1='./../data/baseMaps/base_map_230.npy',
                      map_nu2='./../data/baseMaps/gsm_base_map_408.npy',
                      T_cmb=2.725, nu1=230., nu2=408.)
    plot_healpy(data=beta, show=False, cmap='inferno', cbar=False)
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, ax=ax, orientation='horizontal', aspect=60,
                 pad=0.07, label=r'$\beta\ (\theta, \phi)$')
    plt.savefig('noregion.pdf', bbox_inches="tight")
    
    beta_reg(beta=beta, N=30, min_val=2.45, max_val=3.15)
    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    fig.colorbar(image, ax=ax, orientation='horizontal', aspect=60, extend="both",
                 pad=0.07, label=r'$\beta\ (\theta, \phi)$')
    plt.savefig('region30.pdf', bbox_inches="tight")
