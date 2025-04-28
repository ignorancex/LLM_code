from typing import Callable
import gzip, os, pickle
import dataclasses
import jax.numpy as np
import jax

from layers import WAE, sample_from_prior
from kernels import *
from layers import sample_from_prior
from utils import *


@dataclasses.dataclass
class DataConfig(object):
    n_train: int = 2500
    dist_seed: int = 23  # for linear+randgp
    data_seed: int = 34
    x_dims: int = 1
    mode: str = 'linear'  # / image-linear / hllt
    image_dset: str = 'mnist'  # / cifar
    # AE
    true_z_dims: int = 10
    obs_z_dims: int = 20
    used_z_dims: int = 2
    decoder_act: str = 'swish'
    prior_type: str = 'cube'
    # for linear S1 / AGMM design
    lin_xz_corr: float = 0.5**0.5
    agmm_f0: str = 'abs'
    # for default design
    var_ux: float = 0.1
    var_uy: float = 0.1
    uxz_corr: float = 0.6
    n_rf: int = 1600  
    kx_bw: float = 0.2


def gen_dgmm_data(
        Z: np.ndarray, iv_strength: float, f0_type: str, data_rng: np.ndarray,
        dist_rng: np.ndarray):
    """
    See CausalML/DeepGMM, scenarios/toy_scenarios.py#L101, generate_zoo_data.py#L17
    """
    N, D = Z.shape
    assert D == 1
    data_rng = PRNGKeyHolder(data_rng)
    U = jax.random.normal(data_rng.get(), (N, 1))
    X = Z * iv_strength*2 + U * (1-iv_strength)*2 +\
        jax.random.normal(data_rng.get(), (N, 1)) * 0.1
    if f0_type == 'randgp':
        kspec = KernelSpec(name='rbf', bw_med_multiplier=1).instantiate(x_train=X)
        f0 = GPSampler(None, 1, 3000, kspec, dist_rng)
    else:
        f0 = {
            'sin': lambda x: onp.sin(x),
            'sin_m': lambda x: onp.sin(x+0.3) * (2**0.5),
            'abs': lambda x: onp.abs(x),
            'step': lambda x: 1*(x<0) + 2.5*(x>=0),
            'sigmoid': lambda x: 2/(1+np.exp(-2*x)),
            'linear': lambda x: x,
            '2dpoly': lambda x: -1.5 * x + .9 * (x**2),
            '3dpoly': lambda x: -1.5 * x + .9 * (x**2) + x**3,
        }[f0_type]
    Y = f0(X) + U * 2 + jax.random.normal(data_rng.get(), (N, 1)) * 0.1
    s, b = Y.std(), Y.mean()
    if f0_type == 'randgp':
        s, b = 1, 0  # normalize randgp leads to misspecified variance
    true_f = lambda x: (f0(x) - b) / s
    Y = (Y-b) / s
    return X, Y, true_f


def load_image_data(dset: str, base_path='~/run/liv-im-data'):
    with gzip.open(os.path.expanduser(f'{base_path}/{dset}.pkl.gz'), 'rb') as fin:
        Xall, Yall = pickle.load(fin)
    assert Xall.dtype == onp.uint8
    if dset != 'mnist':
        Xall = Xall.transpose([0, 2, 3, 1])
    assert Xall.shape[-1] <= 3, Xall.shape
    return Xall.astype('f') / 256, Yall


def gen_data(dc: DataConfig, return_z_true=False) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # (Z, X, Y)
    Callable[[np.ndarray], np.ndarray]  # f0, expects 2d input
]:
    assert dc.x_dims == 1, NotImplementedError(dc.x_dims)
    assert dc.mode in ['linear', 'image-linear'], NotImplementedError(dc.mode)

    data_rng = PRNGKeyHolder(jax.random.PRNGKey(dc.data_seed))
    dist_rng = PRNGKeyHolder(jax.random.PRNGKey(dc.dist_seed))
    onp_rng = onp.random.RandomState(dc.data_seed)

    Z_true = sample_from_prior(
        dc.prior_type, data_rng.get(), (dc.n_train*3, dc.true_z_dims))
    if dc.obs_z_dims == dc.true_z_dims:
        Z_obs = Z_true.copy()
    elif dc.mode != 'image-linear':
        # generate Z by sampling from a randomly initialized AE decoder
        dec_layers = [dc.obs_z_dims*3//2, dc.obs_z_dims*3//2]
        wae_model = WAE(
            dc.true_z_dims, dc.obs_z_dims, dec_layers, [20],  # disc unused
            dc.decoder_act, dc.prior_type, 1e-3)
        wae_vars = wae_model.init(
            {'params': dist_rng.get(), 'z': dist_rng.get()},
            np.ones((3, dc.obs_z_dims)), True, method=wae_model.get_loss)
        Z_obs = wae_model.apply(
            wae_vars, Z_true, method=lambda self, inp: self.decoder(inp, False))
    else:
        images, labels = load_image_data(dc.image_dset)
        Z_true = jax.random.uniform(data_rng.get(), (dc.n_train*3, 1), minval=-3, maxval=3)
        Z_im_label = onp.floor(1.5 * Z_true + 5).astype('i')  # result uniform in {0,...,9}
        Z_obs = onp.zeros(tuple([Z_true.shape[0]] + list(images.shape[1:])), dtype=onp.float32)
        idcs = onp.arange(images.shape[0])
        for i in range(10):
            mask = (Z_im_label==i).squeeze()
            idcs_i = onp_rng.choice(idcs[labels==i], (mask.sum(),), replace=False)
            Z_obs[mask] = images[idcs_i]
        Z_obs = np.asarray(Z_obs)

    # DGMM expects Z~U[-3, 3], while our Z_true is from a scaled Uniform distribution
    Z = Z_true[:, :1] / Z_true.max() * 3
    X_obs, Y_obs, f0 = gen_dgmm_data(
        Z, dc.lin_xz_corr**2, dc.agmm_f0, data_rng.get(), dist_rng.get())

    if return_z_true:
        return (Z_obs, X_obs, Y_obs), f0, Z_true
    else:
        return (Z_obs, X_obs, Y_obs), f0


if __name__ == '__main__':
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(DataConfig, dest='d')
    parser.add_argument('--dump_dir', type=str, default=os.path.expanduser('~/run/liv-data'))
    parser.add_argument('--seed_s', type=int, default=20)
    parser.add_argument('--seed_e', type=int, default=30)
    args = parser.parse_args()
    dc: DataConfig = args.d

    os.makedirs(args.dump_dir, exist_ok=True)
    for seed in range(args.seed_s, args.seed_e):
        path = os.path.join(args.dump_dir,
                            f'{dc.agmm_f0}_{dc.true_z_dims}_{dc.n_train}_{seed}.pkl')
        print(path)
        dc = dataclasses.replace(dc, data_seed=seed, dist_seed=seed)
        np_rng = onp.random.RandomState(seed)
        d_all, f0 = gen_data(dc)
        dtrain, dtest = data_split(*d_all, split_ratio=2/3, rng=np_rng)
        dtrain, dval = data_split(*dtrain, split_ratio=0.5, rng=np_rng)
        f_test = f0(dtest[1])
        lcs = locals()
        to_dump = dict((k, lcs[k]) for k in ['dtrain', 'dval', 'dtest', 'f_test'])
        with open(path, 'wb') as fout:
            pickle.dump(to_dump, fout)
