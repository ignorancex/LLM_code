def add_default_args(parser):
    
    # Arguments for script
    parser.add_argument('--exp', type=str, help='which experiment')
    parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
    parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

    # Arguments parsed by HMC
    parser.add_argument('--seed', type=int, default=999, help='seed')
    parser.add_argument('--n_leapfrog', type=int, default=20, help='number of leapfrog steps')
    parser.add_argument('--n_samples', type=int, default=1001, help='number of samples')
    parser.add_argument('--n_burnin', type=int, default=100, help='number of iterations for burn-in')
    parser.add_argument('--target_accept', type=float, default=0.80, help='target acceptance')
    parser.add_argument('--step_size', type=float, default=0.02, help='initial step size')
    parser.add_argument('--n_stepsize_adapt', type=int, default=100, help='number of iterations for step size adaptation')

    # Argument parsed by uturn sampler
    parser.add_argument('--offset', type=float, default=1.0, help='offset for uturn sampler')
    parser.add_argument('--min_nleapfrog', type=int, default=3, help='minimum number of leapfrog steps')
    parser.add_argument('--max_nleapfrog', type=int, default=1024, help='maximum number of leapfrog steps')

    # Argument parsed by jitter-uturn
    parser.add_argument('--n_leapfrog_adapt', type=int, default=100, help='number of iterations for trajectory length adaptation')
    parser.add_argument('--low_nleap_percentile', type=int, default=10, help='lower percentile of trajectory distribution')
    parser.add_argument('--high_nleap_percentile', type=int, default=90, help='higher percentile of trajectory distribution')
    parser.add_argument('--nleap_distribution', type=str, default='empirical', help='higher percentile of trajectory distribution')
    
    # Argument parsed by stepsize adaptation
    parser.add_argument('--constant_trajectory', type=int, default=1, help='trajectory length of delayed stage, deault=1') 
    parser.add_argument('--probabilistic', type=int, default=1, help='probabilistic atlas, default=1')
    parser.add_argument('--n_hessian_samples', type=int, default=10, help='number of points for lbfgs')
    parser.add_argument('--n_hessian_attempts', type=int, default=10, help='number of points for lbfgs')
    parser.add_argument('--hessian_mode', type=str, default='bfgs', help='method to approximate hessian')
    parser.add_argument('--stepsize_distribution', type=str, default='beta', help='distribution for stepsize')
    parser.add_argument('--stepsize_sigma', type=float, default=2., help='width of lognormal distribution')
    parser.add_argument('--max_stepsize_reduction', type=float, default=1000., help='maximum reduction in stepsize')

    # Argument parsed by Atlas
    parser.add_argument('--delayed_proposals', type=int, default=1, help='make delayed proposal when stepsize is okay')

    
    # Arguments for NUTS
    parser.add_argument('--metric', type=str, default="unit_e", help='metric for NUTS')
    parser.add_argument('--n_metric_adapt', type=int, default=0, help='number of iterations for NUTS metric adaptation')
    parser.add_argument('--nuts', type=int, default=0, help='run nuts')

    return parser
