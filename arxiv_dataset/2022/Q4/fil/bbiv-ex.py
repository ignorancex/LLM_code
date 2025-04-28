import dataclasses
import os
import jax
from jax import numpy as np
import numpy as onp

import exputils
from data import DataConfig
from kernels import *
from utils import *
from regressors import *
from bbiv import parser, batched_apply
from hllt_data import gen_hllt_data


parser.add_argument('--use_im', action='store_true', default=False)


def main(args):
    dc: DataConfig = args.d
    model_name = 'hllt' if not args.use_im else 'hllt-im'
    rc: RegressorConfig = dataclasses.replace(
        args.r, inp_dims=dc.obs_z_dims, out_dims=args.n_gps,
        nc=dataclasses.replace(args.r.nc, model=model_name))
    np_rng = onp.random.RandomState(rc.seed)
    rng = PRNGKeyHolder(jax.random.PRNGKey(rc.seed))
    gp_rkey, _ = jax.random.split(rng.get())

    dtrain, dtest, f0, true_ysd = gen_hllt_data(dc.n_train*2, 0.5, dc.data_seed, args.use_im)
    Ztr, Wtr, Xtr, Ytr = dtrain
    Zb_tr = np.concatenate([Ztr, Wtr], 1)  # in the order of (S, E, T)
    dtrain = (Zb_tr, Wtr, Xtr, Ytr)
    dtrain, dval = data_split(*dtrain, split_ratio=0.5, rng=np_rng)
    f_test = f0(dtest[1], dtest[2])

    # Stage 1.  As we will only use the learned I to model f as opposed to y - f, we don't
    # (and cannot) include y as the regression target.
    kx_spec = args.kx.instantiate(x_train=dval[2])
    s1_dtrain_, s1_dval_ = map(lambda t: (t[0], t[2]), (dval, dtrain))  # (Zb, X)
    gp_sampler = GPSampler(Xtr.shape[1], args.n_gps, args.s1_rf, kx_spec, gp_rkey)
    s1_dtrain, s1_dval = map(
        lambda dtup: (dtup[0], batched_apply(gp_sampler, dtup[1])),
        (s1_dtrain_, s1_dval_))

    rc_s1 = dataclasses.replace(rc, nc=dataclasses.replace(rc.nc, model=rc.nc.model+'-s1'))
    s1_val_loss_inl, iv_featurizer = get_nn_predictor(rc_s1, s1_dtrain, s1_dval)
    print('S1 val loss =', s1_val_loss_inl) #, s1_val_loss_gen)

    # Stage 2
    s2_val_loss, f_hat = get_nn_exo_predictor(rc, iv_featurizer, gp_sampler, dtrain, dval)

    print('y_nmse_val =', mse(f_hat(dval[1], dval[2]), dval[3]))
    if not args.use_im:
        print('cf_nmse_val =', mse(f_hat(dval[1], dval[2]), f0(dval[1], dval[2])))

    cf_mse = mse(f_hat(dtest[1], dtest[2]), f_test)
    print(to_py_dict({
        's2_val_loss': s2_val_loss,
        'cf_mse': cf_mse,
        'log_cf_mse': onp.log10(cf_mse * true_ysd**2)
    }))


if __name__ == '__main__':
    args = parser.parse_args()
    exputils.preflight(args)
    main(args)
    with open(os.path.join(args.dir, 'COMPLETED'), 'w') as fout:
        fout.write('.')

