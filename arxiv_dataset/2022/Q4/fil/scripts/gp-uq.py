"""
Reproduces the uncertainty quantification experiment for f0~GP (Table 4 in appendix),
for our method (`-learn_s1`) or the AGMM-RBF baseline (`-no_learn_s1`).

The relevant metrics are `cbr_p90` (credible ball radius), `cic_p90` (CI coverage) and
`cf_mse` (test MSE), and can be read out like

    python filter.py <base_log_dir> '{"d.n_train": 500}' cf_mse

Note that `n_train` corresponds to n_1 / n_2 (half the combined training set size) in paper. To 
compute the credible ball coverage, you'll need to write your own script.
"""

import monad_do as M

try:
    from .utils import *
except:
    from utils import *


def add_args(parser):
    parser.add_argument('-Ns', type=int, nargs='+', default=[500, 2500, 5000])
    parser.add_argument('-learn_s1', default=True, action='store_true')
    parser.add_argument('-no_learn_s1', action='store_false', dest='learn_s1',
                        help='run the kernelized baseline')
    parser.set_defaults(seed_s=0, seed_e=300)
    return parser


@M.do(M.List)
def list_hps(args):
    n_train = yield PF('d', *args.Ns)
    true_z_dims = yield PF('d', 2, 20, 50)
    obs_z_dims = yield PF('d', 2 if true_z_dims.v == 2 else true_z_dims.v*2)
    agmm_f0 = yield PF('d', 'randgp')
    n_gps = 60 if n_train.v > 1000 else 30
    kz_bw_mmp = yield F('kz.bw_med_multiplier', 4)  # for baseline
    lr = 1e-3
    r_seed = yield F('r.seed', *range(args.seed_s, args.seed_e))
    dist_seed = PF('d', r_seed.v)[0]
    data_seed = PF('d', r_seed.v)[0]
    return [locals()]


def base_cmd(args):
    ret = 'python bbiv.py -production '
    if not args.learn_s1:
        ret += '--no_learn_s1'
    return ret
