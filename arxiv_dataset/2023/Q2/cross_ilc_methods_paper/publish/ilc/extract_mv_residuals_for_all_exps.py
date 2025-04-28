import numpy as np, os, sys

fname = 'ilc_weights_residuals_agora_fg_model.npy'
ilc_type = 'mv'
opfd = 'mv_ilc_residuals/'

if not os.path.exists(opfd): os.system('mkdir -p %s' %(opfd))

ilc_dict = np.load(fname, allow_pickle = True).item()['total_ilc_residuals']

for exp in ilc_dict:
    print(exp)
    els, cl_ilc = ilc_dict[exp][ilc_type]
    op_dic = {}
    op_dic['el'], op_dic['cl_residual'] = els, cl_ilc
    opfname = '%s/%s.npy' %(opfd, exp)
    np.save(opfname, op_dic)
