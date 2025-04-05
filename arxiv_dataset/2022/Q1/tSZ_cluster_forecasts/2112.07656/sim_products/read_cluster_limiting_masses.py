import numpy as np
fname = 'cluster_limiting_masses_M500c_vs_z.npy'
cluster_lmz_dic = np.load(fname, allow_pickle = True).item()
for expname in sorted(cluster_lmz_dic):
    current_lmz_dic = cluster_lmz_dic[expname]
    print(expname)
    z_arr, M500c_arr = current_lmz_dic['redshift'], current_lmz_dic['M500c']
    for (z, M500c) in zip(z_arr, M500c_arr):
        print('\t%s = %s [1e14 Msol]' %(z, M500c))