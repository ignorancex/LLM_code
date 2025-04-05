import sys
import numpy as np

njk=461
parnames=['M1', 'Mmin', 'b_hydro', 'r_corr', 'width', 'b_g', 'b_y', 'chi2']
for exp in ['2mpz']+['wisc%d'%(n+1) for n in range(5)]:
    prefix="output_default/sampler_lmin10_kmax1_tinker08_ymilca_wnarrow_"+exp
#    prefix="output_default/sampler_lmin10_kmax0p5_tinker08_ymilca_wnarrow_"+exp
    par_arrs={p:np.zeros(njk) for p in parnames}
    for jk in range(njk):
        print(exp,jk)
        d=np.load(prefix+'_jk%d_vals.npz'%jk)
        for p in parnames:
            par_arrs[p][jk]=d[p]
    fname_save=prefix+"_jkall"
    np.savez(fname_save,
             M1=par_arrs['M1'],
             Mmin=par_arrs['Mmin'],
             b_hydro=par_arrs['b_hydro'],
             r_corr=par_arrs['r_corr'],
             width=par_arrs['width'],
             b_g=par_arrs['b_g'],
             b_y=par_arrs['b_y'],
             chi2=par_arrs['chi2'])
