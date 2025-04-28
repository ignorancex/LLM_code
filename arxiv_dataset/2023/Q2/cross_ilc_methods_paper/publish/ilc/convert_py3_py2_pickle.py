import numpy as np, pickle, gzip

fname = 'ilc_weights_residuals_agora_fg_model_forlensing.npy'
res_dic = np.load(fname, allow_pickle=True).item()

opfname = fname.replace('.npy', '.pkl.gz')
pickle.dump(res_dic, gzip.open(opfname, 'wb'), protocol =2)
