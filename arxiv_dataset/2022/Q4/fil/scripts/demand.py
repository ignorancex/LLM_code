"""
Reproduces the demand experiment in Appendix G.4.  By default runs the low-dimensional setup;
the `-im` flag activates the image setup.  Results can be read out using commands of the form

    python filter.py <base_log_dir> '{"d.n_train": 500}' log_cf_mse

Note that `n_train` corresponds to n_1 / n_2 (half of the combined training set size) in paper.
"""
import os
import monad_do as M

try:
    from .utils import *
except:
    from utils import *


def add_args(parser):
    parser.add_argument('-Ns', type=int, nargs='+', default=[500, 2500])
    parser.add_argument('-im', action='store_true', default=False)
    return parser


@M.do(M.List)
def list_hps(args):
    n_train = yield PF('d', *args.Ns)
    r_seed = yield F('r.seed', *range(args.seed_s, args.seed_e))
    dist_seed = PF('d', r_seed.v)[0]
    data_seed = PF('d', r_seed.v)[0]
    return [locals()]


def base_cmd(args):
    ret = 'python bbiv-ex.py -production --dropout_rate 0.05 --lr 5e-4 '
    if args.im:
        assert os.path.exists(os.path.expanduser('~/run/mnist.pkl.gz'))
        ret += '--use_im '
    return ret

