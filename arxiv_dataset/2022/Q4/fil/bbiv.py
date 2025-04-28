import dataclasses
import os
import pickle
import jax
from jax.config import config
from jax import numpy as np
import numpy as onp
import scipy.stats
from scipy.special import softmax
import matplotlib; matplotlib.use('svg')
from matplotlib import pyplot as plt

import exputils
from data import DataConfig, gen_data
import iv
from kernels import *
from utils import *
from regressors import *


parser = exputils.s_parser('bbiv')
parser.add_arguments(RegressorConfig, dest='r', prefix='r.')
parser.add_arguments(DataConfig, dest='d', prefix='d.')
parser.add_argument('--n_gps', type=int, default=60)
parser.add_arguments(KernelSpec, dest='kx', prefix='kx.',
                     default=dataclasses.replace(KernelSpec(), bw_med_multiplier=1))
parser.add_argument('--s1_rf', type=int, default=3000)
parser.add_argument('--s2_lam', type=str, default='fixed', choices=['fixed', 'grid'])
parser.add_argument('--lam_s', type=float, default=0.25)
parser.add_argument('--lam_e', type=float, default=10)
parser.add_argument('--n_lams', type=int, default=10)
parser.add_argument('--learn_s1', action='store_true', default=True)
parser.add_argument('--no_learn_s1', action='store_false', dest='learn_s1')
# for the fixed-form kernel baseline
parser.add_arguments(KernelSpec, dest='kz', prefix='kz.',
                     default=dataclasses.replace(KernelSpec(), bw_med_multiplier=2))
parser.add_argument('--nu_s', type=float, default=0.2)
parser.add_argument('--nu_e', type=float, default=10)
parser.add_argument('--n_nus', type=int, default=10)
parser.add_argument('--n_nys', type=int, default=100)
parser.add_argument('--jitter', type=float, default=1e-12)
parser.add_argument('--bma', action='store_true', default=False)
parser.add_argument('--no_bma', action='store_false', dest='bma')
parser.add_argument('--bma_bw_a', type=float, default=2.)
parser.add_argument('--bma_sd_a', type=float, default=2.)


def generate_gp_datasets(n_gps, n_rfs, k_spec, rng, *dsets
                         ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    Xtr = dsets[0][1]
    assert len(Xtr.shape) == 2
    eval_gp_train_samples = GPSampler(Xtr.shape[1], n_gps, n_rfs, k_spec, rng)
    return tuple(map(
        lambda dtup: (dtup[0], batched_apply(eval_gp_train_samples, dtup[1], B=200)),
        dsets))


def evaluate_krr(dtrain, dval, kern, nu, nys_sampler=None, linear=False, jitter=1e-12
                 ) -> np.ndarray:
    Xtr, Ytr = dtrain
    # transform output to compute NMSE. for radial kernels no need to transform input
    ymean, ysd = Ytr.mean(), Ytr.std()
    Xva, Yva = dval[0], (dval[1]-ymean)/ysd
    pred_fn = iv.krr(Xtr, (Ytr-ymean)/ysd, kern, nu, nystrom=nys_sampler, linear=linear,
                     jitter=jitter)
    return mse(pred_fn(Xva), Yva)


def get_cb_samples(cov: np.ndarray, pkey: np.ndarray, n_samples: int, return_f=False
                   ) -> Union[List[float], List[Tuple[float, onp.ndarray]]]:
    """
    the credible ball radius is defined as the percentile of \|Δf\|_2^2 where Δf~N(0, cov)
    we return samples of this rv for potential downstream processing, and work with the 
    eigenbasis for fast sampling
    """
    n, jtr = cov.shape[0], 1 / (1e3*cov.shape[0])
    evals, evecs = np.linalg.eigh(cov + jtr*np.eye(n))  # evecs[:, i] crsp to evals[:, i]
    evals = np.clip(evals, 0)
    samples = []
    for _ in range(n_samples):
        pkey, ckey = jax.random.split(pkey)
        eps = jax.random.normal(ckey, shape=evals.shape) * evals**0.5
        sqnorm = (eps**2).mean()
        if not return_f:
            samples.append(sqnorm)
        else:
            samples.append((sqnorm, onp.asarray(evecs @ eps[:, None])))
    return samples


def get_test_stats(pred_mean, pred_cov, f_test, rng):
    if isinstance(pred_cov, np.ndarray):
        pred_cov_diag = np.diag(pred_cov)
        cb_samples = get_cb_samples(pred_cov, rng.get(), 1000)
    else:
        pred_cov_diag, cb_samples = pred_cov
    ret = {'cf_mse': mse(pred_mean, f_test)}
    for pct in [80, 90, 95]:
        ciw = pred_cov_diag[:,None]**0.5 * scipy.stats.norm.ppf(1 - ((100-pct)/200))
        ret.update({
            f'cbr_p{pct}': onp.percentile(cb_samples, pct),
            f'ciw_p{pct}': ciw.mean(),
            f'cic_p{pct}': (np.abs(pred_mean - f_test) <= ciw).mean()
        })
    return ret


def do_plot(pred_tup, name, Xvis, f_vis, path_base, title=''):
    ypred_mean, ypred_cov_diag = pred_tup
    plt.plot(Xvis, ypred_mean, label='pred')
    ymn = ypred_mean.squeeze()
    ysd = ypred_cov_diag**0.5
    t = scipy.stats.norm.ppf(0.95)  # 90% CI
    plt.fill_between(Xvis.squeeze(), ymn-ysd*t, ymn+ysd*t, alpha=0.2)
    plt.plot(Xvis, f_vis, linestyle=':', label='actual')
    plt.title(title) #f'lam={lam:.2f}, stats={s2_proj_stats:.3f}, MSE={cf_mse:.5f}')
    plt.savefig(os.path.join(path_base, f'pred-{name}.svg'))
    with open(os.path.join(path_base, f'pred-{name}.pkl'), 'wb') as fout:
        pickle.dump(tuple(map(onp.asarray, pred_tup)), fout)


def main(args, override_dat_tuple=None):
    dc: DataConfig = args.d
    rc: RegressorConfig = dataclasses.replace(
        args.r, inp_dims=dc.obs_z_dims, out_dims=args.n_gps)
    np_rng = onp.random.RandomState(rc.seed)
    rng = PRNGKeyHolder(jax.random.PRNGKey(rc.seed))
    gp_rkey, gp_v_rkey = jax.random.split(rng.get())

    config.update('jax_enable_x64', False)
    if override_dat_tuple is None:
        d_all, f0 = gen_data(dc)
        dtrain, dtest = data_split(*d_all, split_ratio=2/3, rng=np_rng)
        dtrain, dval = data_split(*dtrain, split_ratio=0.5, rng=np_rng)
        # save evaluation of f0. We need this because the PRNGKey in the GPSamplers will be 
        # interpreted differently once we enable x64. 
        Xtest = dtest[1]
        Xvis = np.linspace(*onp.percentile(dtest[1], [2.5, 97.5]), 100)[:, None]
        f_test, f_vis = map(f0, (Xtest, Xvis))
        del f0
        override_dat_tuple = (dtrain, dval, Xtest, Xvis, f_test, f_vis)

    dtrain, dval, Xtest, Xvis, f_test, f_vis = override_dat_tuple

    # Stage 1
    kx_spec = args.kx.instantiate(x_train=dval[1])
    s1_dtrain, s1_dval = generate_gp_datasets(
        args.n_gps, args.s1_rf, kx_spec, gp_rkey, dval, dtrain)
    s1_y_dtrain, s1_y_dval = (dval[0], dval[2]), (dtrain[0], dtrain[2])
    s1_th_dtrain, s1_th_dval = generate_gp_datasets(
        max(args.n_gps, 80), args.s1_rf, kx_spec, gp_v_rkey, dval, dtrain)

    # - train
    if args.learn_s1:
        train_fn = {
            'nn': get_nn_predictor
        }[rc.method]
        s1_val_loss_inl, fx_pred_fn = train_fn(rc, s1_dtrain, s1_dval)
        fea_extractor_x = compose(lambda i: i/args.n_gps**0.5, fx_pred_fn)
        # regress y
        rc_y = dataclasses.replace(rc, out_dims=1)
        s1_val_loss_y, fea_extractor_y = train_fn(rc_y, s1_y_dtrain, s1_y_dval)
        # set nu
        nu = args.nu_s
        # compute task generalization loss
        _dtrain, _dval = map(
            lambda dtup: (batched_apply(fea_extractor_x, dtup[0]), dtup[1]),
            (s1_th_dval, s1_th_dtrain))
        config.update('jax_enable_x64', True)
        s1_val_loss_gen = evaluate_krr(
            _dtrain, _dval, LinearKernel(), nu=nu, linear=True, jitter=args.jitter)
        #
        fea_extractor = lambda t: np.concatenate([fea_extractor_x(t), fea_extractor_y(t)], 1)
        nys_sampler = None
    else:
        config.update('jax_enable_x64', True)
        s1_kz, s1_kx = args.kz.create(x_train=dval[0]), kx_spec.create()
        nys_sampler = UniformNystromSampler(rng.get(), mn=args.n_nys)
        # determine nu. inl and taskgen are both equiv. to the closed-form val stats
        nu_space = log_linspace(args.nu_s, args.nu_e, args.n_nus)
        _, nmse_gp = iv.kiv_hps_selection(
            dval, dtrain, s1_kz, s1_kx, nu_space, z_nystrom=nys_sampler, jitter=args.jitter,
            return_all_stats=True)
        print(nmse_gp)
        nmse_gp = np.array(nmse_gp) / s1_kx.var
        kwargs = {'kern': s1_kz, 'nys_sampler': nys_sampler, 'jitter': args.jitter}
        nmse_y = np.array(
            [evaluate_krr(s1_y_dtrain, s1_y_dval, nu=nu_, **kwargs) for nu_ in nu_space])
        nu = nu_space[onp.nanargmin(onp.array(nmse_gp + nmse_y))]
        # for comparison, compute val statistics the same way
        s1_val_loss_inl = evaluate_krr(s1_dtrain, s1_dval, nu=nu, **kwargs)
        s1_val_loss_y   = evaluate_krr(s1_y_dtrain, s1_y_dval, nu=nu, **kwargs)
        s1_val_loss_gen = evaluate_krr(s1_th_dtrain, s1_th_dval, nu=nu, **kwargs)
    print('S1 val loss =', s1_val_loss_inl, s1_val_loss_gen, s1_val_loss_y, 'nu =', nu)

    # Stage 2
    if args.learn_s1:
        s2_dtrain, s2_dval = map(
            lambda dtup: (batched_apply(fea_extractor, dtup[0]), dtup[1], dtup[2]),
            (dtrain, dval))
        Kz = LinearKernel()
    else:
        s2_dtrain, s2_dval = (dtrain, dval)
        Kz = args.kz.create(x_train=s2_dtrain[0])
    # - krr statistics
    Y_Z_pred = iv.krr(
        s2_dtrain[0], s2_dtrain[2], Kz, nu, nystrom=nys_sampler, linear=args.learn_s1,
        jitter=args.jitter)
    combined_krr_stats = mse(Y_Z_pred(s2_dval[0]), s2_dval[2])
    # - point estimate (default)
    assert args.s2_lam == 'fixed', NotImplementedError(args.s2_lam)
    # set lam to be E(y-\hat{E}(y|z))^2, which ~corresponds to the DGP Y = (Ef)(Z) + N(0, lam)
    lam = combined_krr_stats
    # reuse the code below to compute the projection-based statistics
    lam_space = [lam]
    Kx = kx_spec.create(x_train=s2_dtrain[1])
    _, lam, stats = iv.kiv_hps_selection(
        s2_dtrain, s2_dval, Kz, Kx, [nu], lam_space, jitter=args.jitter,
        s2_criterion='proj', z_linear=args.learn_s1, z_nystrom=nys_sampler,
        return_all_stats=True)
    for i, lam_i in enumerate(lam_space):
        if lam == lam_i:
            s2_proj_stats = stats[0][i]

    if not args.bma:
        kiv_pred = iv.kiv(
            *s2_dtrain, Kz, Kx, lam, nu, z_linear=args.learn_s1, z_nystrom=nys_sampler,
            jitter=args.jitter)
        pred_mean, (cov_subtracted_part, _) = kiv_pred(Xtest, full_cov=True)
        pred_cov = Kx(Xtest, Xtest) - cov_subtracted_part.A @ cov_subtracted_part.B
        test_stats = get_test_stats(pred_mean, pred_cov, f_test, rng) | {
            's2_neg_logqlh': -kiv_pred.log_qlh()
        }
        logs = {
            's1_val_stats_inl': s1_val_loss_inl,
            's1_val_stats_gen': s1_val_loss_gen,
            's1_val_stats_y': s1_val_loss_y,
            's2_proj_stats': s2_proj_stats,
            'combined_krr_stats': combined_krr_stats,
        } | test_stats
        print(to_py_dict(logs))
        do_plot(kiv_pred(Xvis), 'point', Xvis, f_vis, args.dir)
    else:
        # compute predictions for all `kx_var`. Note the learned instruments are still optimal
        # after scaling. 
        nu_bak, Z_s2_bak = nu, (s2_dtrain[0], s2_dval[0])
        pred_ss = {}
        for kx_sd in [0.5, 1, 1.5, 2, 2.5, 3]:
            kx_var = kx_sd**2
            Kx = dataclasses.replace(kx_spec, var=kx_var).create(x_train=s2_dtrain[1])
            nu = nu_bak * max(1, kx_var)
            if not args.learn_s1:
                kz_spec = dataclasses.replace(args.kz, var=args.kz.var*kx_var)
                Kz = kz_spec.create(x_train=s2_dtrain[0])
            else:
                # scale the learned instruments. The kernel should scale accordingly, but as
                # Kz.var is fixed at 1, we don't need to do anything
                def rescale(inp): return np.concatenate([inp[:, :-1]*kx_sd, inp[:, -1:]], 1)
                # def rescale(inp): return inp * kx_sd
                s2_dtrain = (rescale(Z_s2_bak[0]), s2_dtrain[1], s2_dtrain[2])
                s2_dval = (rescale(Z_s2_bak[1]), s2_dval[1], s2_dval[2])
            kiv_pred = iv.kiv(
                *s2_dtrain, Kz, Kx, lam, nu, z_linear=args.learn_s1, z_nystrom=nys_sampler,
                jitter=args.jitter)
            pred_mean, (cov_subtracted_part, _) = kiv_pred(Xtest, full_cov=True)
            pred_cov = Kx(Xtest, Xtest) - cov_subtracted_part.A @ cov_subtracted_part.B
            pred_cov_ss = (get_cb_samples(pred_cov, rng.get(), 1000, True), np.diag(pred_cov))
            pred_ss[kx_var] = jax.tree_map(
                onp.asarray, (pred_mean, pred_cov_ss, kiv_pred(Xvis), kiv_pred.log_qlh()))
            del kiv_pred, pred_cov

        return override_dat_tuple, pred_ss


def bma_get_pred_tup(p_marg, pred_means, pred_cov_diags):
    pred_mean = sum(p_i * mean_i for p_i, mean_i in zip(p_marg, pred_means))
    pred_cov = sum(p_i * (cov_i + ((mean_i-pred_mean)**2).squeeze())
                   for p_i, mean_i, cov_i in zip(p_marg, pred_means, pred_cov_diags))
    return pred_mean, pred_cov


def bma_main(args):
    kz_spec, kx_spec = args.kz, args.kx
    dtup = None
    preds_dct = {}
    for kx_bw in [0.5, 1, 1.5]:
        args.kx = dataclasses.replace(kx_spec, bw_med_multiplier=kx_bw)
        kz_bw = kz_spec.bw_med_multiplier * kx_bw / kx_spec.bw_med_multiplier  # for baseline
        args.kz = dataclasses.replace(kz_spec, bw_med_multiplier=kz_bw)
        dtup, pred_ss = main(args, override_dat_tuple=dtup)
        preds_dct[kx_bw] = pred_ss
        dtup = jax.tree_map(onp.asarray, dtup)
        reset_xla_memory()
        dtup = jax.tree_map(np.asarray, dtup)

    # with open(os.path.join(args.dir, 'pred-all.pkl'), 'wb') as fout:
    #     pickle.dump(preds_dct, fout)

    Xvis, f_test, f_vis = dtup[-3:]
    
    rng = onp.random.RandomState(args.r.seed)
    pred_ss = []
    log_joints = []
    log_qlhs = []
    for kx_bw, preds_bw in preds_dct.items():
        log_p_bw = scipy.stats.gamma.logpdf(kx_bw, a=args.bma_bw_a)
        for kx_var, ss in preds_bw.items():
            log_qlh = float(ss[-1])
            log_qlhs.append(log_qlh)
            log_p_sd = scipy.stats.invgamma.logpdf(
                kx_var**0.5, a=args.bma_sd_a, scale=args.bma_sd_a)
            log_joints.append(log_p_bw + log_p_sd + log_qlh)
            pred_ss.append(ss[:-1])
    p_marg = softmax(onp.asarray(log_joints))
    print(np.log(p_marg))
    print(log_qlhs)
    # EB
    eb_pred_mean, (_, eb_pred_cov_diag) = pred_ss[onp.argmax(log_qlhs)][:2]
    assert len(eb_pred_cov_diag.shape) == 1
    eb_stats_ = get_test_stats(eb_pred_mean, np.diag(eb_pred_cov_diag), f_test,
                               rng=PRNGKeyHolder(jax.random.PRNGKey(args.r.seed)))
    eb_stats = dict(('eb_'+k, v) for k, v in eb_stats_.items())
    
    # BMA
    K = p_marg.shape[0]
    test_pred_mean, test_pred_cov_diag = bma_get_pred_tup(
        p_marg, [pred_ss[i][0] for i in range(K)], [pred_ss[i][1][1] for i in range(K)])
    vis_pred_mean, vis_pred_cov_diag = bma_get_pred_tup(
        p_marg, [pred_ss[i][2][0] for i in range(K)], [pred_ss[i][2][1] for i in range(K)])
    test_all_cb_samples = [list(pred_ss[i][1][0]) for i in range(K)]
    cb_samples = []
    for _ in range(1000):
        i = rng.choice(K, p=p_marg)
        pmean_i = pred_ss[i][0]
        _, pf_centered_sample = test_all_cb_samples[i].pop()
        cb_samples.append(mse(test_pred_mean, pf_centered_sample + pmean_i))
    bma_stats = get_test_stats(
        test_pred_mean, (test_pred_cov_diag, cb_samples), f_test, rng=None)
    print(to_py_dict(eb_stats | bma_stats))
    do_plot((vis_pred_mean, vis_pred_cov_diag), 'bma', Xvis, f_vis, args.dir)


if __name__ == '__main__':
    args = parser.parse_args()
    exputils.preflight(args)
    (bma_main if args.bma else main)(args)
    with open(os.path.join(args.dir, 'COMPLETED'), 'w') as fout:
        fout.write('.')
