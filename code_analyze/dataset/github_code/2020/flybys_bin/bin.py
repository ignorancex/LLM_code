import os,argparse,pickle

import numpy as np
from scipy.integrate import odeint

import bin_tools,bin_plot

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    
def mkdir_p(path):
    import os,errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
def add_bool_arg(parser, name, default=False,help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',help="Enable %s"%help)
    group.add_argument('--no-' + name, dest=name, action='store_false',help="Disable %s"%help)
    parser.set_defaults(**{name:default})

def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mode",                           type=float,     dest="mode",                        default=1,              help="mode -- 1: single integration -- 2-5: series of integrations (2: changing a2; 3: changing q2; 4: changing i2; 5: changing a1 -- -1: make overview plot")
    parser.add_argument("--name",                           type=str,       dest="name",                        default="test01",       help="name used in filenames of data files and figures")
    parser.add_argument("--m1",                             type=float,     dest="m1",                          default=10.0,           help="Primary mass")
    parser.add_argument("--m2",                             type=float,     dest="m2",                          default=10.0,           help="Secondary mass")
    parser.add_argument("--m3",                             type=float,     dest="m3",                          default=5.0,            help="Tertiary mass")
    parser.add_argument("--m4",                             type=float,     dest="m4",                          default=5.0,            help="Quartery mass")
    parser.add_argument("--q2_min",                         type=float,     dest="q2_min",                      default=0.1,            help="Minimum q2 value in mode 3")
    parser.add_argument("--q2_max",                         type=float,     dest="q2_max",                      default=1.0,            help="Maximum q2 value in mode 3")
    parser.add_argument("--Q",                              type=float,     dest="Q",                           default=20.0,           help="Outer orbit periapsis distance")
    parser.add_argument("--N_a1",                           type=int,       dest="N_a1",                        default=10,             help="Number of systems in series (mode 5)")
    parser.add_argument("--N_a2",                           type=int,       dest="N_a2",                        default=10,             help="Number of systems in series (mode 2)")
    parser.add_argument("--N_q2",                           type=int,       dest="N_q2",                        default=10,             help="Number of systems in series (mode 3)")
    parser.add_argument("--N_i2",                           type=int,       dest="N_i2",                        default=10,             help="Number of systems in series (mode 4)")
    parser.add_argument("--a1",                             type=float,     dest="a1",                          default=1.0,            help="Orbit 1 semimajor axis")    
    parser.add_argument("--a1_min",                         type=float,     dest="a1_min",                      default=1.0,            help="Minimum a1 value in mode 5")    
    parser.add_argument("--a1_max",                         type=float,     dest="a1_max",                      default=3.0,            help="Maximum a1 value in mode 5")
    parser.add_argument("--a2",                             type=float,     dest="a2",                          default=1.0,            help="Orbit 2 semimajor axis")    
    parser.add_argument("--a2_min",                         type=float,     dest="a2_min",                      default=0.5,            help="Minimum a2 value in mode 2")    
    parser.add_argument("--a2_max",                         type=float,     dest="a2_max",                      default=3.0,            help="Maximum a2 value in mode 2")    
    parser.add_argument("--e1",                             type=float,     dest="e1",                          default=0.01,           help="Orbit 1 eccentricity")    
    parser.add_argument("--e2",                             type=float,     dest="e2",                          default=0.01,           help="Orbit 2 eccentricity")    
    parser.add_argument("--e3",                             type=float,     dest="e3",                          default=1.5,            help="Outer orbit eccentricity (>=1.0)")    
    parser.add_argument("--i1",                             type=float,     dest="i1",                          default=np.pi/2.0,      help="Orbit 1 inclination (rad)")    
    parser.add_argument("--i2",                             type=float,     dest="i2",                          default=0.01*np.pi/180.0,help="Orbit 2 inclination (rad)")    
    parser.add_argument("--i2_min",                         type=float,     dest="i2_min",                      default=0.01*np.pi/180.0,help="Minimum i2 value (rad; mode 4)")    
    parser.add_argument("--i2_max",                         type=float,     dest="i2_max",                      default=70.0*np.pi/180.0,help="Maximum i2 value (rad; mode 4)")    
    parser.add_argument("--AP1",                            type=float,     dest="AP1",                         default=np.pi/4.0,      help="Orbit 1 argument of periapsis (rad)")
    parser.add_argument("--AP2",                            type=float,     dest="AP2",                         default=0.01*np.pi/180.0,help="Orbit 2 argument of periapsis (rad)")
    parser.add_argument("--LAN1",                           type=float,     dest="LAN1",                        default=0.01*np.pi/180.0,help="Orbit 1 longitude of the ascending node (rad)")
    parser.add_argument("--LAN2",                           type=float,     dest="LAN2",                        default=0.01*np.pi/180.0,help="Orbit 2 longitude of the ascending node (rad)")
    parser.add_argument("--TA1",                            type=float,     dest="TA1",                         default=0.01*np.pi/180.0,help="Orbit 1 true anomaly (rad)")
    parser.add_argument("--TA2",                            type=float,     dest="TA2",                         default=0.01*np.pi/180.0,help="Orbit 2 true anomaly (rad)")
    parser.add_argument("--TA3_fraction",                   type=float,     dest="TA3_fraction",                default=0.98,           help="Initial perturber true anomaly, expressed as a fraction of -\arccos(-1/e3). Increase if the  numerical integrations do not seem converged. ")
    parser.add_argument("--N_steps",                        type=int,       dest="N_steps",                     default=3000,           help="Number of external (print) output steps")
    parser.add_argument("--mxstep",                         type=int,       dest="mxstep",                      default=1000000,        help="Maximum number of internal steps taken in the ODE integration. Increase if ODE integrator give mstep errors. ")    
    parser.add_argument("--G",                              type=float,     dest="CONST_G",                     default=4.0*np.pi**2,   help="Gravitational constant used in N-body integrations. Should not affect Newtonian results. ")
#    parser.add_argument("--c",                              type=float,     dest="c",                           default=63239.72638679138, help="Speed of light (PN terms only). ")
    parser.add_argument("--fontsize",                       type=float,     dest="fontsize",                    default=22,             help="Fontsize for plots")
    parser.add_argument("--labelsize",                      type=float,     dest="labelsize",                   default=18,             help="Labelsize for plots")
    parser.add_argument("--code",                           type=str,       dest="code",                        default="rebound",      help="N-body code used: rebound or abie")
    parser.add_argument("--annotation",                     type=str,       dest="annotation",                  default="",             help="Annotation string in plot")

    
    ### boolean arguments ###
    add_bool_arg(parser, 'verbose',                         default=False,         help="verbose terminal output")
    add_bool_arg(parser, 'calc',                            default=True,          help="do calculation (and save results). If False, will try to load previous results")
    add_bool_arg(parser, 'plot',                            default=True,          help="make plots")
    add_bool_arg(parser, 'plot_fancy',                      default=False,         help="use LaTeX for plot labels (slower but nicer fonts)")
    add_bool_arg(parser, 'show',                            default=True,          help="show plots")
    add_bool_arg(parser, 'quad',                            default=True,          help="include quadrupole-order terms")
    add_bool_arg(parser, 'oct',                             default=True,          help="include octupole-order terms")
    add_bool_arg(parser, 'hex',                             default=True,          help="include hexadecupole-order terms")
    add_bool_arg(parser, 'hex_cross',                       default=True,          help="include hexadecupole-order cross term")
    add_bool_arg(parser, 'do_nbody',                        default=True,          help="do N-body integrations as well as partially averaged")
    add_bool_arg(parser, 'include_backreaction',            default=True,          help="include backreaction of outer orbit on inner orbits in inner-averaged integrations")
    add_bool_arg(parser, 'compare_backreaction',            default=False,         help="special mode comparing cases with backreaction included and not included")
    
    args = parser.parse_args()

    ### Processed arguments ###
    args.M1 = args.m1 + args.m2
    args.M2 = args.m3 + args.m4
    args.M = args.M1 + args.M2
    args.a3 = args.Q/(1.0 - args.e3)
    args.TA3 = -args.TA3_fraction*np.arccos(-1.0/args.e3)

    args.n3 = np.sqrt(args.CONST_G*args.M/(np.fabs(args.a3)**3))

    a = -args.TA3
    n3 = args.n3
    e3 = args.e3
    args.tend = (1.0/n3)*( -4*np.arctanh(((-1 + e3)*np.tan(a/2.))/np.sqrt(-1 + e3**2)) + (2*e3*np.sqrt(-1 + e3**2)*np.sin(a))/(1 + e3*np.cos(a)) )

    if args.verbose==True:
        print('tend',args.tend,'1/n3',1.0/n3,'P_bin1',2.0*np.pi*np.sqrt(args.a1**3/(args.CONST_G*args.M1)))
    
    args.times = np.linspace(0.0,args.tend,args.N_steps)

    args.N_bodies = 4
    args.N_orbits = args.N_bodies - 1

    args.series_description = ""
    if args.mode==2:
        args.series_description = "a2"
    elif args.mode==3:
        args.series_description = "q2"
    elif args.mode==4:
        args.series_description = "i2"
    elif args.mode==5:
        args.series_description = "a1"

    args.data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/data_' + args.name + ".pkl"
    args.series_data_filename = os.path.dirname(os.path.realpath(__file__)) + '/data/series_data_' + str(args.series_description) + "_" + args.name + ".pkl"
    
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/data')
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs')

    args.fig_dir = os.path.dirname(os.path.realpath(__file__)) + '/figs/'
    args.fig_filename = args.fig_dir + args.name + ".pdf"
    args.series_fig_filename = os.path.dirname(os.path.realpath(__file__)) + '/figs/series_' + str(args.series_description) + "_" + args.name + ".pdf"
    
    return args

def integrate(args):

    data_av = bin_tools.integrate_averaged(args)

    data_nbody = None
    if args.do_nbody==True:
        data_nbody = bin_tools.integrate_nbody(args)

    data = args,data_av,data_nbody
    
    filename = args.data_filename

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def integrate_series(args):
    
    if args.mode == 2:
        values = np.linspace(args.a2_min,args.a2_max,args.N_a2)
    elif args.mode == 3:
        ### Will assume M2 = m3 + m4 is constant
        values = np.linspace(args.q2_min,args.q2_max,args.N_q2)
    elif args.mode == 4:
        values = np.linspace(args.i2_min,args.i2_max,args.N_i2)
    if args.mode == 5:
        values = np.linspace(args.a1_min,args.a1_max,args.N_a1)
    
    data_series_av = {'series_description':args.series_description,'values':values,'Delta_a1s':[],'Delta_e1s':[],'Delta_i1s':[], \
        'Delta_a2s':[],'Delta_e2s':[],'Delta_i2s':[]}
    data_series_nbody = {'series_description':args.series_description,'values':values,'Delta_a1s':[],'Delta_e1s':[],'Delta_i1s':[], \
        'Delta_a2s':[],'Delta_e2s':[],'Delta_i2s':[]}

    for index,value in enumerate(values):
        if args.mode==2:
            args.a2 = value
        elif args.mode==3:
            q2 = value
            args.m3 = args.M2/(1.0+q2)
            args.m4 = args.m3*q2
        elif args.mode==4:
            args.i2 = value
        elif args.mode==5:
            args.a1 = value

        print("index",index,"a1",args.a1,"a2",args.a2,"m3",args.m3,"m4",args.m4,"i2",args.i2)
        data_av = bin_tools.integrate_averaged(args)
        
        data_series_av["Delta_a1s"].append( data_av["Delta_a1"] )
        data_series_av["Delta_a2s"].append( data_av["Delta_a2"] )
        data_series_av["Delta_e1s"].append( data_av["Delta_e1"] )
        data_series_av["Delta_e2s"].append( data_av["Delta_e2"] )
        data_series_av["Delta_i1s"].append( data_av["Delta_i1"] )
        data_series_av["Delta_i2s"].append( data_av["Delta_i2"] )
        
        data_nbody = None
        if args.do_nbody==True:
            data_nbody = bin_tools.integrate_nbody(args)

            data_series_nbody["Delta_a1s"].append( data_nbody["Delta_a1"] )
            data_series_nbody["Delta_a2s"].append( data_nbody["Delta_a2"] )
            data_series_nbody["Delta_e1s"].append( data_nbody["Delta_e1"] )
            data_series_nbody["Delta_e2s"].append( data_nbody["Delta_e2"] )
            data_series_nbody["Delta_i1s"].append( data_nbody["Delta_i1"] )
            data_series_nbody["Delta_i2s"].append( data_nbody["Delta_i2"] )
        elif args.compare_backreaction == True:
            args.include_backreaction = False
            data_av = bin_tools.integrate_averaged(args)
            
            if args.verbose==True:
                print("Carying out inner-averaged integration without backreaction")
            data_series_nbody["Delta_a1s"].append( data_av["Delta_a1"] )
            data_series_nbody["Delta_a2s"].append( data_av["Delta_a2"] )
            data_series_nbody["Delta_e1s"].append( data_av["Delta_e1"] )
            data_series_nbody["Delta_e2s"].append( data_av["Delta_e2"] )
            data_series_nbody["Delta_i1s"].append( data_av["Delta_i1"] )
            data_series_nbody["Delta_i2s"].append( data_av["Delta_i2"] )
            args.include_backreaction = True
        
    filename = args.series_data_filename
    
    data_series = args,data_series_av,data_series_nbody
    with open(filename, 'wb') as handle:
        pickle.dump(data_series, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_arguments()

    if args.verbose==True:
        print('arguments:')
        from pprint import pprint
        pprint(vars(args))

    if args.plot>0:
        if HAS_MATPLOTLIB == False:
            print( 'Error importing Matplotlib -- choose --plot 0')
            exit(-1)

        if args.plot_fancy == True:
            pyplot.rc('text',usetex=True)
            pyplot.rc('legend',fancybox=True)


    if args.calc == True:
        if args.mode in [1]:
            data = integrate(args)
        elif args.mode in [2,3,4,5]:
            data = integrate_series(args)

    if args.plot == True:
        if args.mode in [1]:
            filename = args.data_filename
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            bin_plot.plot_function_single(args,data)
        elif args.mode in [2,3,4,5]:
            filename = args.series_data_filename
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            bin_plot.plot_function_series(args,data)
        elif args.mode in [-1]:
            bin_plot.plot_function_overview(args)

