import numpy as np

from matplotlib import pyplot

import numpy.random as randomf

from scipy import integrate

import pickle
import numpy.random as randomf
import os,argparse,copy
from os import path

import time

import math

try:
    from secularmultiple.secularmultiple import SecularMultiple,Particle,Tools
except ImportError:
    print("Unable to import SecularMultiple")

CONST_R_SUN = 0.004649130343817401
CONST_G = 4.0*np.pi**2


def add_bool_arg(parser, name, default=False,help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true',help="Enable %s"%help)
    group.add_argument('--no-' + name, dest=name, action='store_false',help="Disable %s"%help)
    parser.set_defaults(**{name:default})

def parse_arguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mode",                   type=int,       dest="mode",                default=0,                help="Mode -- 0: analytic calculation only; 1: analytic calculation and numerical integration; 2: doing multiple calculations and making plots for a series of parameters; 3: making some illustrative contour plots; 4: making plots illustrating the allowed ranges of gamma; 5: make plots showing parameter space for orbital flips ")

    parser.add_argument("--e0",                     type=float,     dest="e0",                  default=0.01,             help="Initial inner orbit eccentricity")    
    parser.add_argument("--g0",                     type=float,     dest="g0",                  default=0.01,             help="Initial inner orbit argument of periapsis")    
    parser.add_argument("--theta0",                 type=float,     dest="theta0",              default=0.1,              help="Cosine of the initial relative inclination")    
    parser.add_argument("--gamma",                  type=float,     dest="gamma",               default=0.1,              help="Parameter quantifying the angular momentum of the inner orbit relative to the outer; gamma = (1/2) L_1/G_2")    

    parser.add_argument("--m1",                     type=float,     dest="m1",                  default=1.0,              help="Inner binary primary mass (numerical integrations only; advised not to change)")
    parser.add_argument("--m2",                     type=float,     dest="m2",                  default=1.0,              help="Inner binary secondary mass (numerical integrations only; advised not to change)")    
    parser.add_argument("--m3",                     type=float,     dest="m3",                  default=1.0,              help="Tertiary mass (numerical integrations only; advised not to change)")    
    parser.add_argument("--a1",                     type=float,     dest="a1",                  default=1.0,              help="Inner binary semimajor axis (numerical integrations only; advised not to change)")
    parser.add_argument("--e2",                     type=float,     dest="e2",                  default=0.8,              help="Outer binary eccentricity (numerical integrations only; advised not to change)")        
    parser.add_argument("--i1",                     type=float,     dest="i1",                  default=0.0,              help="Inner binary inclination (numerical integrations only; advised not to change)")        
    parser.add_argument("--LAN1",                   type=float,     dest="LAN1",                default=0.0,              help="Inner binary longitude of the ascending node (numerical integrations only; advised not to change)")        
    parser.add_argument("--LAN2",                   type=float,     dest="LAN2",                default=np.pi,            help="Outer binary longitude of the ascending node (numerical integrations only; advised not to change)")        
    parser.add_argument("--AP2",                    type=float,     dest="AP2",                 default=0.0,              help="Outer binary argument of periapsis (numerical integrations only; advised not to change)")        
    parser.add_argument("--Nsteps",                 type=int,       dest="Nsteps",              default=1000,             help="Number of snapshot output steps for numerical integration (does not affect root finding). Advised not to change. ")        

    ### boolean arguments ###
    add_bool_arg(parser, 'diag',                            default=False,         help="Show diagnostic plots (numerical integrations)")
    add_bool_arg(parser, 'num',                             default=True,          help="For plot series (mode = 2): do new calculations including direct integration")
    add_bool_arg(parser, 'calc',                            default=True,          help="For plot series (mode = 2): do new calculations")
    add_bool_arg(parser, 'show',                            default=False,         help="Show plots")
    add_bool_arg(parser, 'plot_fancy',                      default=True,          help="Use TeX when making plots")
    add_bool_arg(parser, 'verbose',                         default=False,         help="Verbose terminal output")
    
    args = parser.parse_args()

    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs')
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/data')

    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs/no_num/')
    mkdir_p(os.path.dirname(os.path.realpath(__file__)) + '/figs/diag/')


    eps = 1.0e-2
    if args.theta0 <= -1.0:
        args.theta0 = -1.0 + eps
    if args.theta0 >= 1.0:
        args.theta0 = 1.0 - eps
    if args.e0 <= 0.0:
        args.e0 = eps
    if args.e0 >= 1.0:
        args.e0 = 1.0 - eps

    return args

def mkdir_p(path):
    import os,errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def save(data,filename):
    with open(filename,'wb') as file:
        pickle.dump(data,file)

def load(filename):
    with open(filename,'rb') as file:
        data = pickle.load(file)
    return data

try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_function_series(args):
    filename3 = "data/data3_an_only_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pkl"
    filename4 = "data/data4_an_only_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pkl"

    if args.num == True:
        filename1 = "data/data1_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pkl"
        filename2 = "data/data2_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pkl"
#        filename3 = "data/data3_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pkl"
#        filename4 = "data/data4_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pkl"
    else:
        filename1 = "data/data1_an_only_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pkl"
        filename2 = "data/data2_an_only_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pkl"
    
    if args.calc==True:

        print("As a function of theta0 for select gamma")
        e0 = args.e0
        g0 = args.g0
        theta0_points_num = np.linspace(-0.99,0.99,20)
        theta0_points_an = np.linspace(-0.99,0.99,200)
        gamma_points = np.array([0.05,0.1,0.2])

        e_max_an = [[] for x in range(len(gamma_points))]
        e_max_num_mean = [[] for x in range(len(gamma_points))]
        e_max_num_std = [[] for x in range(len(gamma_points))]
        
        T_ZLK_an = [[] for x in range(len(gamma_points))]
        T_ZLK_num_mean = [[] for x in range(len(gamma_points))]
        T_ZLK_num_std = [[] for x in range(len(gamma_points))]
        
        for index_gamma,gamma in enumerate(gamma_points):
            if args.num == True:
                for index_theta0,theta0 in enumerate(theta0_points_num):
                    print("index_gamma",index_gamma,"gamma",gamma,"theta0",theta0,"index_theta0",index_theta0)
        
                    e_min_mean,e_min_std,e_max_mean,e_max_std,T_ZLK_mean,T_ZLK_std,L1_div_C2 = numerical_integration(args,e0,theta0,gamma,g0)
                    #print("e_min_mean",e_min_mean,"e_min_std",e_min_std,"e_max_mean",e_max_mean,"e_max_std",e_max_std,"T_ZLK_mean",T_ZLK_mean,"T_ZLK_std",T_ZLK_std)
            
                    e_max_num_mean[index_gamma].append(e_max_mean)
                    e_max_num_std[index_gamma].append(e_max_std)

                    T_ZLK_num_mean[index_gamma].append(T_ZLK_mean/L1_div_C2)
                    T_ZLK_num_std[index_gamma].append(T_ZLK_std/L1_div_C2)

            for index_theta0,theta0 in enumerate(theta0_points_an):

                Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)
        
                e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
                T_ZLK = timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max)
                
                e_max_an[index_gamma].append(e_max)
                T_ZLK_an[index_gamma].append(T_ZLK)

        for index_gamma,gamma in enumerate(gamma_points):
            e_max_an[index_gamma] = np.array(e_max_an[index_gamma])
            e_max_num_mean[index_gamma] = np.array(e_max_num_mean[index_gamma])
            e_max_num_std[index_gamma] = np.array(e_max_num_std[index_gamma])

            T_ZLK_an[index_gamma] = np.array(T_ZLK_an[index_gamma])
            T_ZLK_num_mean[index_gamma] = np.array(T_ZLK_num_mean[index_gamma])
            T_ZLK_num_std[index_gamma] = np.array(T_ZLK_num_std[index_gamma])

        data1 = {'e0':e0,'g0':g0,'theta0_points_num':theta0_points_num,'theta0_points_an':theta0_points_an,'gamma_points':gamma_points,'e_max_an':e_max_an,'e_max_num_mean':e_max_num_mean,'e_max_num_std':e_max_num_std, \
            'T_ZLK_an':T_ZLK_an,'T_ZLK_num_mean':T_ZLK_num_mean,'T_ZLK_num_std':T_ZLK_num_std}

        save(data1,filename1)

        print("As a function of gamma for select theta0")
        e0 = args.e0
        g0 = args.g0

        theta0_points = np.array([-0.8,-0.7,-0.6,-0.2,0.0,0.2,0.8])
        gamma_points_an = np.linspace(0.01,0.4,200)
        gamma_points_num = np.linspace(0.01,0.4,20)
        
        e_max_an = [[] for x in range(len(theta0_points))]
        e_max_num_mean = [[] for x in range(len(theta0_points))]
        e_max_num_std = [[] for x in range(len(theta0_points))]

        T_ZLK_an = [[] for x in range(len(theta0_points))]
        T_ZLK_num_mean = [[] for x in range(len(theta0_points))]
        T_ZLK_num_std = [[] for x in range(len(theta0_points))]
        
        for index_theta0,theta0 in enumerate(theta0_points):
            if args.num == True:
                for index_gamma,gamma in enumerate(gamma_points_num):
                
                    print("index_gamma",index_gamma,"index_theta0",index_theta0)
        
                    e_min_mean,e_min_std,e_max_mean,e_max_std,T_ZLK_mean,T_ZLK_std,L1_div_C2 = numerical_integration(args,e0,theta0,gamma,g0)
                    #print("e_min_mean",e_min_mean,"e_min_std",e_min_std,"e_max_mean",e_max_mean,"e_max_std",e_max_std,"T_ZLK_mean",T_ZLK_mean,"T_ZLK_std",T_ZLK_std)
            
                    e_max_num_mean[index_theta0].append(e_max_mean)
                    e_max_num_std[index_theta0].append(e_max_std)

                    T_ZLK_num_mean[index_theta0].append(T_ZLK_mean/L1_div_C2)
                    T_ZLK_num_std[index_theta0].append(T_ZLK_std/L1_div_C2)

            for index_gamma,gamma in enumerate(gamma_points_an):

                Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)
        
                e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
                T_ZLK = timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max)

                e_max_an[index_theta0].append(e_max)
                T_ZLK_an[index_theta0].append(T_ZLK)
        
        for index_gamma,gamma in enumerate(gamma_points):
            e_max_an[index_gamma] = np.array(e_max_an[index_gamma])
            e_max_num_mean[index_gamma] = np.array(e_max_num_mean[index_gamma])
            e_max_num_std[index_gamma] = np.array(e_max_num_std[index_gamma])

            T_ZLK_an[index_gamma] = np.array(T_ZLK_an[index_gamma])
            T_ZLK_num_mean[index_gamma] = np.array(T_ZLK_num_mean[index_gamma])
            T_ZLK_num_std[index_gamma] = np.array(T_ZLK_num_std[index_gamma])

        data2 = {'e0':e0,'g0':g0,'theta0_points':theta0_points,'gamma_points_an':gamma_points_an,'gamma_points_num':gamma_points_num,'e_max_an':e_max_an,'e_max_num_mean':e_max_num_mean,'e_max_num_std':e_max_num_std, \
            'T_ZLK_an':T_ZLK_an,'T_ZLK_num_mean':T_ZLK_num_mean,'T_ZLK_num_std':T_ZLK_num_std}

        save(data2,filename2)

        
        print("As a function of g0 for select gamma")
        num = False
        e0 = args.e0
        theta0 = args.theta0
        g0_points_num = np.linspace(-np.pi,np.pi,20)
        g0_points_an = np.linspace(-np.pi,np.pi,200)
        gamma_points = np.array([0.01,0.05,0.1])

        e_max_an = [[] for x in range(len(gamma_points))]
        e_max_num_mean = [[] for x in range(len(gamma_points))]
        e_max_num_std = [[] for x in range(len(gamma_points))]
        
        T_ZLK_an = [[] for x in range(len(gamma_points))]
        T_ZLK_num_mean = [[] for x in range(len(gamma_points))]
        T_ZLK_num_std = [[] for x in range(len(gamma_points))]
        
        for index_gamma,gamma in enumerate(gamma_points):
            if num == True:
                for index_g0,g0 in enumerate(g0_points_num):
                    print("index_gamma",index_gamma,"index_g0",index_g0)
        
                    e_min_mean,e_min_std,e_max_mean,e_max_std,T_ZLK_mean,T_ZLK_std,L1_div_C2 = numerical_integration(args,e0,theta0,gamma,g0)
                    #print("e_min_mean",e_min_mean,"e_min_std",e_min_std,"e_max_mean",e_max_mean,"e_max_std",e_max_std,"T_ZLK_mean",T_ZLK_mean,"T_ZLK_std",T_ZLK_std)
            
                    e_max_num_mean[index_gamma].append(e_max_mean)
                    e_max_num_std[index_gamma].append(e_max_std)

                    T_ZLK_num_mean[index_gamma].append(T_ZLK_mean/L1_div_C2)
                    T_ZLK_num_std[index_gamma].append(T_ZLK_std/L1_div_C2)

            for index_g0,g0 in enumerate(g0_points_an):

                Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)
        
                e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
                T_ZLK = timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max)
                
                e_max_an[index_gamma].append(e_max)
                T_ZLK_an[index_gamma].append(T_ZLK)

        for index_gamma,gamma in enumerate(gamma_points):
            e_max_an[index_gamma] = np.array(e_max_an[index_gamma])
            e_max_num_mean[index_gamma] = np.array(e_max_num_mean[index_gamma])
            e_max_num_std[index_gamma] = np.array(e_max_num_std[index_gamma])

            T_ZLK_an[index_gamma] = np.array(T_ZLK_an[index_gamma])
            T_ZLK_num_mean[index_gamma] = np.array(T_ZLK_num_mean[index_gamma])
            T_ZLK_num_std[index_gamma] = np.array(T_ZLK_num_std[index_gamma])

        data3 = {'e0':e0,'theta0':theta0,'g0_points_num':g0_points_num,'g0_points_an':g0_points_an,'gamma_points':gamma_points,'e_max_an':e_max_an,'e_max_num_mean':e_max_num_mean,'e_max_num_std':e_max_num_std, \
            'T_ZLK_an':T_ZLK_an,'T_ZLK_num_mean':T_ZLK_num_mean,'T_ZLK_num_std':T_ZLK_num_std}

        save(data3,filename3)



        print("As a function of e0 for select gamma")
        num = False
        g0 = args.g0
        theta0 = args.theta0
        e0_points_num = np.linspace(0.0,0.99,20)
        e0_points_an = np.linspace(0.0,0.99,200)
        gamma_points = np.array([0.01,0.05,0.1])

        e_max_an = [[] for x in range(len(gamma_points))]
        e_max_num_mean = [[] for x in range(len(gamma_points))]
        e_max_num_std = [[] for x in range(len(gamma_points))]
        
        T_ZLK_an = [[] for x in range(len(gamma_points))]
        T_ZLK_num_mean = [[] for x in range(len(gamma_points))]
        T_ZLK_num_std = [[] for x in range(len(gamma_points))]
        
        for index_gamma,gamma in enumerate(gamma_points):
            if num == True:
                for index_e0,e0 in enumerate(e0_points_num):
                    print("index_gamma",index_gamma,"index_e0",index_e0)
        
                    e_min_mean,e_min_std,e_max_mean,e_max_std,T_ZLK_mean,T_ZLK_std,L1_div_C2 = numerical_integration(args,e0,theta0,gamma,g0)
                    #print("e_min_mean",e_min_mean,"e_min_std",e_min_std,"e_max_mean",e_max_mean,"e_max_std",e_max_std,"T_ZLK_mean",T_ZLK_mean,"T_ZLK_std",T_ZLK_std)
            
                    e_max_num_mean[index_gamma].append(e_max_mean)
                    e_max_num_std[index_gamma].append(e_max_std)

                    T_ZLK_num_mean[index_gamma].append(T_ZLK_mean/L1_div_C2)
                    T_ZLK_num_std[index_gamma].append(T_ZLK_std/L1_div_C2)

            for index_e0,e0 in enumerate(e0_points_an):

                Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)
        
                e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
                T_ZLK = timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max)
                
                e_max_an[index_gamma].append(e_max)
                T_ZLK_an[index_gamma].append(T_ZLK)

        for index_gamma,gamma in enumerate(gamma_points):
            e_max_an[index_gamma] = np.array(e_max_an[index_gamma])
            e_max_num_mean[index_gamma] = np.array(e_max_num_mean[index_gamma])
            e_max_num_std[index_gamma] = np.array(e_max_num_std[index_gamma])

            T_ZLK_an[index_gamma] = np.array(T_ZLK_an[index_gamma])
            T_ZLK_num_mean[index_gamma] = np.array(T_ZLK_num_mean[index_gamma])
            T_ZLK_num_std[index_gamma] = np.array(T_ZLK_num_std[index_gamma])

        data4 = {'g0':g0,'theta0':theta0,'e0_points_num':e0_points_num,'e0_points_an':e0_points_an,'gamma_points':gamma_points,'e_max_an':e_max_an,'e_max_num_mean':e_max_num_mean,'e_max_num_std':e_max_num_std, \
            'T_ZLK_an':T_ZLK_an,'T_ZLK_num_mean':T_ZLK_num_mean,'T_ZLK_num_std':T_ZLK_num_std}

        save(data4,filename4)


    try:
        data1 = load(filename1)
        data2 = load(filename2)
        data3 = load(filename3)
        data4 = load(filename4)
    except:
        print("Error loading data files!")
        exit(-1)
        
    if args.plot_fancy==True:
        pyplot.rc('text',usetex=True)
        pyplot.rc('legend',fancybox=True)          

    colors = ['tab:blue','tab:red','tab:green','tab:gray','tab:purple','tab:brown','tab:cyan']
    markers = ['.','x','*','s','8','p','>']
    s=50
    fontsize=24
    labelsize=18
    
    
    ### e_max/T_ZLK as a function of inclination ###
    fig = pyplot.figure(figsize=(8,7))
    plot=fig.add_subplot(1,1,1)

    fig_T = pyplot.figure(figsize=(8,7))
    plot_T=fig_T.add_subplot(1,1,1)

    theta0_points = np.linspace(-1.0,1.0,1000)
    plot.plot(theta0_points,np.sqrt(1.0-(5.0/3.0)*theta0_points**2),color='k',linestyle='dotted',label=r'$\sqrt{1-(5/3) \theta_0^2}$')

    for index_gamma,gamma in enumerate(data1['gamma_points']):
        color=colors[index_gamma]
        marker=markers[index_gamma]
        label = "$\gamma=%s$"%gamma
        if args.num == True:
            plot.scatter(data1['theta0_points_num'],data1['e_max_num_mean'][index_gamma],color=color,label=label,marker=marker,s=s)
            plot.plot(data1['theta0_points_an'],data1['e_max_an'][index_gamma],color=color)            
            
            plot_T.scatter(data1['theta0_points_num'],data1['T_ZLK_num_mean'][index_gamma],color=color,label=label,marker=marker,s=s)
            xs, ys = filter_nans(data1['theta0_points_an'],data1['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color)

            e0 = data1['e0']
            g0 = data1['g0']

            theta0_crit1 = ((-4*np.sqrt(1 - e0**2)*gamma + np.sqrt(70 + 8*(2 + 3*e0**2)*gamma**2 - 40*(3 + e0**2*gamma**2)*np.cos(2*g0) + 50*np.cos(4*g0)))*(1.0/np.sin(g0)**2))/20.
            theta0_crit2 = -((4*np.sqrt(1 - e0**2)*gamma + np.sqrt(70 + 8*(2 + 3*e0**2)*gamma**2 - 40*(3 + e0**2*gamma**2)*np.cos(2*g0) + 50*np.cos(4*g0)))*(1.0/np.sin(g0)**2))/20.
            plot_T.axvline(x=theta0_crit1,color=color,linestyle='dotted')
            plot_T.axvline(x=theta0_crit2,color=color,linestyle='dotted')

        else:
            plot.plot(data1['theta0_points_an'],data1['e_max_an'][index_gamma],color=color,label=label)

            xs, ys = filter_nans(data1['theta0_points_an'],data1['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color,label=label)
            
        plot.axvline(x=-gamma,color=color,linestyle='dashed')


    plot.set_xlim(-1.0,1.0)
    plot_T.set_xlim(-1.0,1.0)
    plot.set_ylim(0.0,1.1)

    plot.set_xlabel(r"$\theta_0$",fontsize=fontsize)
    plot_T.set_xlabel(r"$\theta_0$",fontsize=fontsize)    
    
    plot.set_ylabel("$e_\mathrm{max}$",fontsize=fontsize)
    plot_T.set_ylabel("$T_\mathrm{ZLK}/(L_1/C_2)$",fontsize=fontsize)

    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot_T.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    
    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=fontsize,framealpha=1)

    handles,labels = plot_T.get_legend_handles_labels()
    plot_T.legend(handles,labels,loc="best",fontsize=fontsize,framealpha=1)

    plot.set_title("$e_0 = %s; \,g_0 = %s\,\mathrm{rad}$"%(data1['e0'],data1['g0']),fontsize=fontsize)
    plot_T.set_title("$e_0 = %s; \,g_0 = %s\,\mathrm{rad}$"%(data1['e0'],data1['g0']),fontsize=fontsize)
    
    if args.num == True:
        filename = "figs/fig_emax1_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
        filename_T = "figs/fig_T_ZLK1_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
    else:
        filename = "figs/no_num/fig_emax1_no_num_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
        filename_T = "figs/no_num/fig_T_ZLK1_no_num_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
    fig.savefig(filename,dpi=200)
    fig_T.savefig(filename_T,dpi=200)


    ### e_max/T_ZLK as a function of gamma ###
    fig = pyplot.figure(figsize=(8,7))
    plot=fig.add_subplot(1,1,1)

    fig_T = pyplot.figure(figsize=(8,7))
    plot_T=fig_T.add_subplot(1,1,1)

    for index_theta0,theta0 in enumerate(data2['theta0_points']):
        color=colors[index_theta0]
        marker=markers[index_theta0]
        label = r"$\theta_0=%s$"%theta0
        if args.num == True:
            plot.scatter(data2['gamma_points_num'],data2['e_max_num_mean'][index_theta0],color=color,marker=marker,s=s,label=label)
            plot.plot(data2['gamma_points_an'],data2['e_max_an'][index_theta0],color=color)

            gamma_max = -theta0
            plot.axvline(x=gamma_max,color=color,linestyle='dotted')

            plot_T.scatter(data2['gamma_points_num'],data2['T_ZLK_num_mean'][index_theta0],color=color,marker=marker,s=s,label=label)
            xs, ys = filter_nans(data2['gamma_points_an'],data2['T_ZLK_an'][index_theta0])
            plot_T.plot(xs,ys,color=color)

            e0 = data2['e0']
            g0 = data2['g0']
            gamma_crit = (1.0/(e0**2))*( np.sqrt(1.0-e0**2)*theta0 + np.sqrt( theta0**2 + e0**2*(2.0 - 5.0*np.sin(g0)**2 + (-1.0 + 5.0*np.sin(g0)**2)*theta0**2)) )
            plot_T.axvline(x=gamma_crit,color=color,linestyle='dotted')

        else:
            plot.plot(data2['gamma_points_an'],data2['e_max_an'][index_theta0],color=color,label=label)
            xs, ys = filter_nans(data2['gamma_points_an'],data2['T_ZLK_an'][index_theta0])
            plot_T.plot(xs,ys,color=color,label=label)
            
        plot.axhline(y=np.sqrt(1.0-(5.0/3.0)*theta0**2),color=color,linestyle='dotted')

    plot.set_xlim([-0.02,1.1*data2['gamma_points_an'][-1]])
    
    plot_T.set_xlim(0.0,data2['gamma_points_an'][-1])
    plot.set_xlabel(r"$\gamma$",fontsize=fontsize)
    plot_T.set_xlabel(r"$\gamma$",fontsize=fontsize)
    plot.set_ylabel("$e_\mathrm{max}$",fontsize=fontsize)
    plot_T.set_ylabel("$T_\mathrm{ZLK}/(L_1/C_2)$",fontsize=fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot_T.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    handles,labels = plot_T.get_legend_handles_labels()
    plot_T.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    plot.set_title("$e_0 = %s; \,g_0 = %s\,\mathrm{rad}$"%(data2['e0'],data2['g0']),fontsize=fontsize)
    plot_T.set_title("$e_0 = %s; \,g_0 = %s\,\mathrm{rad}$"%(data2['e0'],data2['g0']),fontsize=fontsize)
    if args.num == True:
        filename = "figs/fig_emax2_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
        filename_T = "figs/fig_T_ZLK2_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
    else:
        filename = "figs/no_num/fig_emax2_no_num_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
        filename_T = "figs/no_num/fig_T_ZLK2_no_num_e0_" + str(args.e0) + "_g0_" + str(args.g0) + ".pdf"
    fig.savefig(filename,dpi=200)
    fig_T.savefig(filename_T,dpi=200)


    ### e_max/T_ZLK as a function of g0 ###
    num = False
    fig = pyplot.figure(figsize=(8,7))
    plot=fig.add_subplot(1,1,1)

    fig_T = pyplot.figure(figsize=(8,7))
    plot_T=fig_T.add_subplot(1,1,1)

    for index_gamma,gamma in enumerate(data3['gamma_points']):
        color=colors[index_gamma]
        marker=markers[index_gamma]
        label = r"$\gamma=%s$"%gamma
        if num == True:
            plot.scatter(data3['g0_points_num'],data3['e_max_num_mean'][index_gamma],color=color,marker=marker,s=s,label=label)
            plot.plot(data3['g0_points_an'],data3['e_max_an'][index_gamma],color=color)

            plot_T.scatter(data3['g0_points_num'],data3['T_ZLK_num_mean'][index_gamma],color=color,marker=marker,s=s,label=label)
            xs, ys = filter_nans(data3['g0_points_an'],data3['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color)
        else:
            plot.plot(data3['g0_points_an'],data3['e_max_an'][index_gamma ],color=color,label=label)
            xs, ys = filter_nans(data3['g0_points_an'],data3['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color,label=label)
            
    plot.set_xlabel(r"$g_0/\mathrm{rad}$",fontsize=fontsize)
    plot_T.set_xlabel(r"$g_0/\mathrm{rad}$",fontsize=fontsize)
    plot.set_ylabel("$e_\mathrm{max}$",fontsize=fontsize)
    plot_T.set_ylabel("$T_\mathrm{ZLK}/(L_1/C_2)$",fontsize=fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot_T.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    handles,labels = plot_T.get_legend_handles_labels()
    plot_T.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    plot.set_title(r"$e_0 = %s; \,\theta_0 = %s$"%(data3['e0'],data3['theta0']),fontsize=fontsize)
    plot_T.set_title(r"$e_0 = %s; \,\theta_0 = %s$"%(data3['e0'],data3['theta0']),fontsize=fontsize)
    if num == True:
        filename = "figs/fig_emax3_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pdf"
        filename_T = "figs/fig_T_ZLK3_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pdf"
    else:
        filename = "figs/no_num/fig_emax3_no_num_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pdf"
        filename_T = "figs/no_num/fig_T_ZLK3_no_num_e0_" + str(args.e0) + "_theta0_" + str(args.theta0) + ".pdf"
    fig.savefig(filename,dpi=200)
    fig_T.savefig(filename_T,dpi=200)


    ### e_max/T_ZLK as a function of e0 ###
    num = False
    fig = pyplot.figure(figsize=(8,7))
    plot=fig.add_subplot(1,1,1)

    fig_T = pyplot.figure(figsize=(8,7))
    plot_T=fig_T.add_subplot(1,1,1)

    for index_gamma,gamma in enumerate(data4['gamma_points']):
        color=colors[index_gamma]
        marker=markers[index_gamma]
        label = r"$\gamma=%s$"%gamma
        if num == True:
            plot.scatter(data4['e0_points_num'],data4['e_max_num_mean'][index_gamma],color=color,marker=marker,s=s,label=label)
            plot.plot(data4['e0_points_an'],data4['e_max_an'][index_gamma],color=color)

            plot_T.scatter(data4['e0_points_num'],data4['T_ZLK_num_mean'][index_gamma],color=color,marker=marker,s=s,label=label)
            xs, ys = filter_nans(data4['e0_points_an'],data4['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color)
        else:
            plot.plot(data4['e0_points_an'],data4['e_max_an'][index_gamma ],color=color,label=label)
            xs, ys = filter_nans(data4['e0_points_an'],data4['T_ZLK_an'][index_gamma])
            plot_T.plot(xs,ys,color=color,label=label)

    plot.set_xlabel(r"$e_0$",fontsize=fontsize)
    plot_T.set_xlabel(r"$e_0$",fontsize=fontsize)
    plot.set_ylabel("$e_\mathrm{max}$",fontsize=fontsize)
    plot_T.set_ylabel("$T_\mathrm{ZLK}/(L_1/C_2)$",fontsize=fontsize)
    
    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    plot_T.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)

    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    handles,labels = plot_T.get_legend_handles_labels()
    plot_T.legend(handles,labels,loc="best",fontsize=0.6*fontsize,framealpha=1)

    plot.set_title(r"$g_0 = %s\,\mathrm{rad}; \,\theta_0 = %s$"%(data4['g0'],data4['theta0']),fontsize=fontsize)
    plot_T.set_title(r"$g_0 = %s\,\mathrm{rad}; \,\theta_0 = %s$"%(data4['g0'],data4['theta0']),fontsize=fontsize)
    if num == True:
        filename = "figs/fig_emax4_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pdf"
        filename_T = "figs/fig_T_ZLK4_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pdf"
    else:
        filename = "figs/no_num/fig_emax4_no_num_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pdf"
        filename_T = "figs/no_num/fig_T_ZLK4_no_num_g0_" + str(args.g0) + "_theta0_" + str(args.theta0) + ".pdf"
    fig.savefig(filename,dpi=200)
    fig_T.savefig(filename_T,dpi=200)
    

    if args.show==True:
        pyplot.show()
    
def filter_nans(xs,ys):
    xs_new = []
    ys_new = []
    
    for i,y in enumerate(ys):
        if y==y:
            xs_new.append(xs[i])
            ys_new.append(y)
    
    xs_new = np.array(xs_new)
    ys_new = np.array(ys_new)
    
    return xs_new,ys_new
    
def plot_function_contours(args):

    if args.plot_fancy==True:
        pyplot.rc('text',usetex=True)
        pyplot.rc('legend',fancybox=True)          

    N_points = 10000
    colors = ['k','tab:grey','tab:blue','tab:red','tab:green','tab:purple','tab:brown','tab:cyan','tab:gray']
    fontsize=22
    labelsize=18


    e0 = 0.5
    g10_values = np.array([0.0,1.0])*np.pi/2.0

    series_theta0_values = [ np.cos( np.array([39.0,40.0,46.0,70.0])*np.pi/180.0 ), np.cos( np.array([170,150.0,120.0,100.0])*np.pi/180.0 )]
    series_gamma_values = [[0.000001],[0.000001,0.1,0.2]]
    
    linewidths = [1.0,2.0,3.0]
    linestyles=['solid','dashed']
    for index_series_theta0,theta0_values in enumerate(series_theta0_values):
        for index_series_gamma,gamma_values in enumerate(series_gamma_values):
            fig = pyplot.figure(figsize=(8,6))
            plot=fig.add_subplot(1,1,1)

            for index_gamma,gamma in enumerate(gamma_values):
                linewidth=linewidths[index_gamma]
                
                for index_theta0,theta0 in enumerate(theta0_values):
                    color = colors[index_theta0]
                    label_theta0 = r"$%s$"
                    for index_g10,g10 in enumerate(g10_values):
                        
                        label=None
                        if index_g10==0 and index_theta0 == 0:
                            label=r"$\gamma=%s$"%round(gamma,1)
                        
                        x0 = 1.0-e0**2
                        H0 = (2.0 + 3.0*e0**2) * (3.0*theta0**2 - 1.0) + 15.0*e0**2 * (1.0 - theta0**2) * np.cos(2.0*g10)
                   
                        cos_g1_points1 = []
                        cos_g1_points2 = []

                        x_points = np.linspace(0.0,1.0,N_points)
                        e_points = np.sqrt(1.0 - x_points)
                        plot_e_points = []
                        
                        Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                        C_ZLK = C_ZLK_function(e0,theta0,gamma,g10)
                        linestyle=linestyles[index_g10]
                                        
                        for index_e,e in enumerate(e_points):
                            x = 1.0 - e**2
                            theta = theta_function(x,Theta_ZLK,gamma)

                            cos_g1_sq = (1.0/2.0) * (1.0 + (H0 - (2.0 + 3*e**2) * (3.0*theta**2 - 1.0))/( 15.0*e**2 * (1.0 - theta**2)))
                            if cos_g1_sq >= 0.0 and cos_g1_sq < 1.0:
                                cos_g1 = np.sqrt(cos_g1_sq)

                                plot_e_points.append(e)
                                cos_g1_points1.append(-cos_g1)
                                cos_g1_points2.append(cos_g1)

                        plot.plot(cos_g1_points1,plot_e_points,color=color,linestyle=linestyle,linewidth=linewidth,label=label) #label='$\omega_{1,0} = %s^\circ$'%round(g10*180.0/np.pi,1)
                        plot.plot(cos_g1_points2,plot_e_points,color=color,linestyle=linestyle,linewidth=linewidth)

                        ### fill up the slits around \cos(\omega_1)=0 (which arise due to a finite number of points taken, and do not disappear unless taking an enormous number of points)
                        plot.plot( [cos_g1_points1[0],cos_g1_points2[0]], [plot_e_points[0],plot_e_points[0]],color=color,linestyle=linestyle,linewidth=linewidth)
                        
                        if C_ZLK < 0.0: ### librating solutions
                            plot.plot( [cos_g1_points1[-1],cos_g1_points2[-1]], [plot_e_points[-1],plot_e_points[-1]],color=color,linestyle=linestyle,linewidth=linewidth)
            alpha=1.0
            if index_series_theta0==0:
                plot.annotate("$39^\circ$",xy=(-0.55, 0.10),fontsize=fontsize*alpha,color=colors[0])
                plot.annotate("$40^\circ$",xy=(-0.05, 0.17),fontsize=fontsize*alpha,color=colors[1])
                plot.annotate("$46^\circ$",xy=(-0.05, 0.39),fontsize=fontsize*alpha,color=colors[2])
                plot.annotate("$70^\circ$",xy=(0.5, 0.74),fontsize=fontsize*alpha,color=colors[3])

                plot.annotate("$39^\circ/$",xy=(-0.8, 0.52),fontsize=fontsize*alpha,color=colors[0])
                plot.annotate("$40^\circ$",xy=(-0.6, 0.52),fontsize=fontsize*alpha,color=colors[1])
                plot.annotate("$46^\circ$",xy=(-0.8, 0.68),fontsize=fontsize*alpha,color=colors[2])
                plot.annotate("$70^\circ$",xy=(-0.8, 0.92),fontsize=fontsize*alpha,color=colors[3])
            if index_series_theta0==1:
                plot.annotate("$170^\circ$",xy=(-0.8,0.42),fontsize=fontsize*alpha,color=colors[0])
                plot.annotate("$150^\circ$",xy=(-0.8,0.6),fontsize=fontsize*alpha,color=colors[1])
                plot.annotate("$120^\circ$",xy=(-0.8,0.7),fontsize=fontsize*alpha,color=colors[2])
                plot.annotate("$100^\circ$",xy=(-1,0.95),fontsize=fontsize*alpha,color=colors[3])

                plot.annotate("$100^\circ$",xy=(0.58,0.7),fontsize=fontsize*alpha,color=colors[3])
                plot.annotate("$120^\circ$",xy=(-0.05,0.7),fontsize=fontsize*alpha,color=colors[2])
                plot.annotate("$150^\circ$",xy=(-0.8,0.26),fontsize=fontsize*alpha,color=colors[1])

            plot.set_xlim(-1.0,1.0)
            plot.set_ylim(0.0,1.0)

            plot.set_xlabel(r"$\cos(g)$",fontsize=fontsize)
            plot.set_ylabel("$e$",fontsize=fontsize)

            plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
                
            handles,labels = plot.get_legend_handles_labels()
            plot.legend(handles,labels,loc="lower right",fontsize=0.8*fontsize,framealpha=1)

            if index_series_gamma==0:
                filename = 'figs/fig_contours_TP_%s.pdf'%index_series_theta0
            else:
                filename = 'figs/fig_contours_%s.pdf'%index_series_theta0
            fig.savefig(filename,dpi=200)


def gamma_stab_function(q1,q2,e2):
    C=2.8
    return (1.0/(2.0*np.sqrt(C))) * pow( np.sqrt(1.0-e2)/(1.0+e2), 1.0/5.0) * (q1/((1.0+q1)**2)) * pow(1.0+q2, 3.0/10.0)/q2

def epsoct_gamma_function(q1,q2,e2,gamma):
    return 4.0*gamma**2*e2 * ((1.0-q1)*(1.0+q1)**3/(q1**2)) * (q2**2/(1.0+q2))

def alpha_da_function(q1,q2,e2,gamma):
    return (1.0/8.0)*(1.0/(gamma**3)) * (q1**3*(1.0+q2)**2)/((1.0+q1)**6*(q2**4))
    
def plot_function_gamma_ranges(args):

    if args.plot_fancy==True:
        pyplot.rc('text',usetex=True)
        pyplot.rc('legend',fancybox=True)          

    N_points = 1000
    colors = ['k','tab:red','tab:blue','tab:red','tab:green','tab:purple','tab:brown','tab:cyan','tab:gray']
    fontsize=22
    labelsize=18

    q1_values = [0.1,1.0]
    q2_values = [0.1,1.0,5]
    e2_values = np.linspace(0.0,1.0,N_points)

    linestyles=['dashed','solid']

#    print("test",gamma_stab_function(1.0,2.0,0.5))
    fig = pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1)
    linewidths=[0.5,1.5,3.0]
    linewidth=1.5
    for index_q1,q1 in enumerate(q1_values):
        linestyle=linestyles[index_q1]
        for index_q2,q2 in enumerate(q2_values):
            color = colors[index_q2]
            linewidth=linewidths[index_q2]
            #color = 'k'
            label = r'$q_1=%s; \, q_2=%s$'%(q1,q2)
            
            ys = []
            for index_e2,e2 in enumerate(e2_values):
                gamma_stab = gamma_stab_function(q1,q2,e2)
                ys.append(gamma_stab)
            
            plot.plot(e2_values,ys,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
    plot.set_xlim(0.0,1.0)

    plot.set_xlabel(r"$e_2$",fontsize=fontsize)
    plot.set_ylabel("$\gamma_\mathrm{stab}$",fontsize=fontsize)

    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
        
    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="upper left",fontsize=0.8*fontsize,framealpha=1)

    filename = 'figs/fig_gamma_range_stab.pdf'
    fig.savefig(filename,dpi=200)

    

#    print("test",epsoct_gamma_function(0.1,1.0,0.2,0.2))
    q1_values = [0.1,0.9]
    q2_values = [0.1,1.0,5]

    e2_values = [0.1,0.99]

    fig = pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,xscale='log',yscale='log')

    for index_q1,q1 in enumerate(q1_values):
        linestyle=linestyles[index_q1]
        for index_q2,q2 in enumerate(q2_values):
            for index_e2,e2 in enumerate(e2_values):
                color = colors[index_q2]
                linewidth=linewidths[index_e2]
            
                gamma_stab = gamma_stab_function(q1,q2,e2)
                gamma_values = pow(10.0,np.linspace(-3.0,np.log10(gamma_stab)))

                if index_e2==0:
                    #label = r'$q_1=%s; \, q_2=%s; \, e_2 = %s$'%(q1,q2,e2)
                    label = r'$q_1=%s; \, q_2=%s$'%(q1,q2)
                else:
                    label=""

                ys = []
                for index_gamma,gamma in enumerate(gamma_values):
                    epsoct_gamma = epsoct_gamma_function(q1,q2,e2,gamma)
                    ys.append(epsoct_gamma)

                plot.plot(gamma_values,ys,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
    #plot.set_title('$e_2=%s$'%e2,fontsize=fontsize)
    
    plot.set_xlim(1e-3,0.7)
    
    plot.set_xlabel(r"$\gamma$",fontsize=fontsize)
    plot.set_ylabel(r"$\epsilon_\mathrm{oct}$",fontsize=fontsize)

    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
        
    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="best",fontsize=0.8*fontsize,framealpha=1)

    #filename = 'figs/fig_gamma_range_oct_e2_' + str(e2) + '.pdf'
    filename = 'figs/fig_gamma_range_oct.pdf'
    fig.savefig(filename,dpi=200)


    fig = pyplot.figure(figsize=(8,6))
    plot=fig.add_subplot(1,1,1,xscale='log',yscale='log')

    q1_values = [0.1,0.9]
    q2_values = [0.1,1.0,10]

    #print("test",alpha_da_function(0.4,0.6,0.2,0.2))
    e2_values = [0.0,0.9]
    for index_q1,q1 in enumerate(q1_values):
        linestyle=linestyles[index_q1]
        for index_q2,q2 in enumerate(q2_values):
            color = colors[index_q2]
        
            for index_e2,e2 in enumerate(e2_values):
                linewidth=linewidths[index_e2]
                gamma_stab = gamma_stab_function(q1,q2,e2)
                gamma_values = pow(10.0,np.linspace(-3.0,np.log10(gamma_stab)))

                label = r'$q_1=%s; \, q_2=%s; \, e_2=%s$'%(q1,q2,e2)

                ys = []
                for index_gamma,gamma in enumerate(gamma_values):
                    alpha_da = alpha_da_function(q1,q2,e2,gamma)
                    ys.append(alpha_da)

                plot.plot(gamma_values,ys,color=color,linestyle=linestyle,linewidth=linewidth,label=label)
   
    plot.axhline(y=1.0,color='tab:red',linestyle='dotted')
    plot.set_xlim(1e-3,1e2)
    plot.set_ylim(1.0e-1,1.0e8)
    plot.set_xlabel(r"$\gamma$",fontsize=fontsize)
    plot.set_ylabel(r"$\alpha_\mathrm{DA}$",fontsize=fontsize)

    plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
        
    handles,labels = plot.get_legend_handles_labels()
    plot.legend(handles,labels,loc="upper right",fontsize=0.55*fontsize,framealpha=1)

    filename = 'figs/fig_gamma_range_da.pdf'
    fig.savefig(filename,dpi=200)


def H_function(e,theta,g):
    return (2.0 + 3.0*e**2)*(3.0*theta**2 - 1.0) + 15.0*e**2*(1.0 - theta**2)*np.cos(2.0*g)

def Theta_ZLK_function(e0,theta0,gamma):
    return ( np.sqrt(1.0-e0**2)*theta0 + gamma*(1.0 - e0**2) )

def theta_function(x,Theta_ZLK,gamma):
    return (Theta_ZLK - gamma*x)/np.sqrt(x)

def C_ZLK_function(e0,theta0,gamma,g0):
    H0 = H_function(e0,theta0,g0)
    return (H0 - 2.0*( 3.0*( np.sqrt(1.0-e0**2)*theta0 - gamma*e0**2)**2 - 1.0 ))/12.0

def timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max):
    x_min = 1.0 - e_max**2
    x_max = 1.0 - e_min**2

    int_args=(C_ZLK,Theta_ZLK,gamma)
    
    eps = 0.0
    integral = integrate.quad(timescale_integrand_function,x_min*(1.0+eps),x_max*(1.0-eps),args=int_args,limit=200, epsrel=1.0e-8, epsabs = 1.0e-12, points = [x_min,x_max])[0]
    
    return 0.5*integral
    
def timescale_function(L1_div_C2,C_ZLK,Theta_ZLK,gamma,e_min,e_max):

    return L1_div_C2*timescale_function_normalised_to_L1_div_C2(C_ZLK,Theta_ZLK,gamma,e_min,e_max)

def timescale_integrand_function(x,*args):
    C_ZLK,Theta_ZLK,gamma = args
    theta = theta_function(x,Theta_ZLK,gamma)

    theta_sq = theta**2
    c0 = 12.0*C_ZLK + 2.0*(3.0*( Theta_ZLK - gamma)**2 - 1.0 )

    return 1.0/np.sqrt( x* (20.0 + c0 - 30.0*theta_sq + 6.0*x*(4.0*theta_sq - 3.0))*(10.0 - c0 + 6.0*x*(theta_sq - 2.0) ) )
    
def e_min_e_max_function(C_ZLK,Theta_ZLK,gamma):
    A = 9 + 13*gamma**4 + 48*gamma*Theta_ZLK - 16*gamma**3*Theta_ZLK + gamma**2*(-6 - 24*C_ZLK + 4*Theta_ZLK**2)
    B = 27 - 27*gamma**2 - 108*C_ZLK*gamma**2 - 99*gamma**4 - 180*C_ZLK*gamma**4 + 35*gamma**6 + 216*gamma*Theta_ZLK - 144*gamma**3*Theta_ZLK - 288*C_ZLK*gamma**3*Theta_ZLK - 264*gamma**5*Theta_ZLK + 306*gamma**2*Theta_ZLK**2 + 438*gamma**4*Theta_ZLK**2 - 208*gamma**3*Theta_ZLK**3
    arg1 = A**3 - B**2
    
    if arg1>=0:
        phi = np.arctan2( np.sqrt(arg1), B)
        x_min = (1.0/(12.0*gamma**2))*( (3.0 + 5.0*gamma**2 + 8.0*gamma*Theta_ZLK) - 2.0*np.sqrt(A) * np.sin( phi/3.0 + np.pi/6.0 ) )
    else:
        arg2 = B + np.sqrt(-arg1)
        
        if arg2 < 0.0:
            x_min =  (1.0/(12.0*gamma**2))*( (3.0 + 5.0*gamma**2 + 8.0*gamma*Theta_ZLK) - A*pow(-arg2,-1.0/3.0) - pow(-arg2,1.0/3.0) )
        else:
            x_min =  (1.0/(12.0*gamma**2))*( (3.0 + 5.0*gamma**2 + 8.0*gamma*Theta_ZLK) + A*pow(arg2,-1.0/3.0) + pow(arg2,1.0/3.0) )

    if C_ZLK < 0.0: ### librating
        x_max = (1.0/(12.0*gamma**2))*( (3.0 + 5.0*gamma**2 + 8.0*gamma*Theta_ZLK) + 2.0*np.sqrt(A) * np.sin( phi/3.0 - np.pi/6.0 ) )
    else: ### circulating
        x_max = (1.0 + gamma*Theta_ZLK - np.sqrt( 1.0 + gamma*(gamma*(-2.0 + 2.0*C_ZLK + (gamma-Theta_ZLK)**2) + 2.0*Theta_ZLK) ) )/(gamma**2)

#    print("x_min",x_min,"x_max",x_max)
#    print("x_min",x_min,"x_max",x_max,"xflip",Theta_ZLK/gamma,"Theta",Theta_ZLK,"gamma",gamma)

    e_min = np.sqrt(1.0 - x_max)
    e_max = np.sqrt(1.0 - x_min)
    return e_min,e_max
    
def compute_a2_from_gamma(gamma,m1,m2,m3,a1,e2):
    M1 = m1+m2
    M2 = M1+m3
    a2 = 0.25 * a1 * (1.0/(1.0-e2**2)) * (M2/M1) * ( (m1*m2/(m3*M1))/gamma )**2
    return a2

def compute_L1_div_C2(CONST_G,m1,m2,m3,a1,a2,e2):
    M1 = m1+m2
    M2 = M1+m3

    L1 = (m1*m2/M1)*np.sqrt(CONST_G*a1*M1)    
    C2 = ((1.0/16.0)*CONST_G*m1*m2*m3/(M1*a2)) * (a1/a2)**2 * pow(1.0-e2**2,-3.0/2.0)

    return L1/C2
    
def root_function_m(m2,*args):
    gamma,a1,a2,e2,m1,m3 = args
    gamma_num = 0.5*np.sqrt( (1.0/(1.0-e2**2)) * (a1/a2) * (m1+m2+m3)/(m1+m2) ) * (m1*m2/((m1+m2)*m3))
    return gamma - gamma_num
    
def numerical_integration(args,e0,theta0,gamma,g0):
    
    m1 = args.m1
    m2 = args.m2
    m3 = args.m3
    a1 = args.a1
    i1 = args.i1
    i2 = np.arccos(theta0)
    LAN1 = args.LAN1
    LAN2 = args.LAN2
    AP1 = g0
    AP2 = args.AP2
    e2 = args.e2

    M1 = m1+m2
    M2 = M1+m3

    a2 = compute_a2_from_gamma(gamma,m1,m2,m3,a1,e2)

    """
    args_root = (gamma,a1,a2,e2,m1,m3)
    sol = optimize.root_scalar(root_function_m,
        x0 = 1.0e-10,
        x1 = m1,
#        1.0e-10,
        method = 'secant',
        #method = 'bounded',
        #bounds = [1.0e-10,2.0],
        #maxiter=1000,
        args=args_root)
    m2 = sol.root
    print("m2",m2)
    print("test",gamma,.5*np.sqrt( (1.0/(1.0-e2**2)) * (a1/a2) * (m1+m2+m3)/(m1+m2) ) * (m1*m2/((m1+m2)*m3)))
    """

    if a2 <= a1:
        print("numerical_integration -- ERROR: a2=",a2,"< a1 = ",a1,"; please decrease gamma")
        exit(0)

    particles = Tools.create_nested_multiple(3, [m1,m2,m3],[a1,a2],[e0,e2],[i1,i2],[AP1,AP2],[LAN1,LAN2])
    binaries = [x for x in particles if x.is_binary==True]
    
    code = SecularMultiple()
    CONST_G = code.CONST_G
    
    code.add_particles(particles)
    inner_binary = binaries[0]
    outer_binary = binaries[1]
    
    inner_binary.check_for_stationary_eccentricity = True

    code.include_quadrupole_order_terms = True
    code.include_octupole_order_binary_pair_terms = False
    code.include_octupole_order_binary_triplet_terms = False
    code.include_hexadecupole_order_binary_pair_terms = False
    code.include_dotriacontupole_order_binary_pair_terms = False
        
    e_print = []
    INCL_print = []
    rel_INCL_print = []
    t_print = []
    
    start = time.time()

    t = 0.0
    N = args.Nsteps
    
    L1_div_C2 = compute_L1_div_C2(CONST_G,m1,m2,m3,a1,a2,e2)
    
    tend = 10.0*L1_div_C2

    t_print_indices_min = []
    t_print_indices_max = []

    i_print = 0
    dt = tend/float(N)
    while t<=tend:
        t+=dt

        code.evolve_model(t)
        t = code.model_time
        flag = code.flag
        
        if args.verbose==True:
            print( 't',t,'es',[x.e for x in binaries],'INCL_parent',inner_binary.INCL_parent,"flag",flag,"max e",[x.maximum_eccentricity_has_occurred for x in binaries],"min e",[x.minimum_eccentricity_has_occurred for x in binaries])

        rel_INCL_print.append(inner_binary.INCL_parent)
        e_print.append(inner_binary.e)
        INCL_print.append(inner_binary.INCL)
        t_print.append(t)
        i_print += 1
    
        if flag==2:
            if inner_binary.minimum_eccentricity_has_occurred == True:
                t_print_indices_min.append(i_print-1)
            if inner_binary.maximum_eccentricity_has_occurred == True:
                t_print_indices_max.append(i_print-1)

            inner_binary.minimum_eccentricity_has_occurred = False
            inner_binary.maximum_eccentricity_has_occurred = False

            ### Integrate for a short time without stationary eccentricity root finding, to avoid "getting stuck" and finding the same stationary point over and over again. ###
            inner_binary.check_for_stationary_eccentricity = False
            code.evolve_model(t+dt)
            t=code.model_time
            inner_binary.check_for_stationary_eccentricity = True
            
        
    if args.verbose==True:
        print('wall time',time.time()-start)
    
    t_print = np.array(t_print)
    rel_INCL_print = np.array(rel_INCL_print)
    e_print = np.array(e_print)
    
    code.reset()

    T_ZLKs = np.array([t_print[t_print_indices_max][i+1] - t_print[t_print_indices_max][i] for i in range(len(t_print_indices_max)-1)])
    
    if HAS_MATPLOTLIB==True and args.diag==True:
        if args.plot_fancy==True:
            pyplot.rc('text',usetex=True)
            pyplot.rc('legend',fancybox=True)          

        
        fig=pyplot.figure(figsize=(8,8))
        fontsize=22
        labelsize=18
        plot1=fig.add_subplot(2,1,1)
        plot2=fig.add_subplot(2,1,2,yscale="log")
        norm_factor = (1.0/L1_div_C2)
        plot1.plot(norm_factor*t_print,rel_INCL_print*180.0/np.pi)
        plot2.plot(norm_factor*t_print,1.0-e_print)
        
        plot1.scatter(norm_factor*t_print[t_print_indices_min],rel_INCL_print[t_print_indices_min]*180.0/np.pi,color='b',label="$\mathrm{Local\,minima}$")
        plot1.scatter(norm_factor*t_print[t_print_indices_max],rel_INCL_print[t_print_indices_max]*180.0/np.pi,color='r',label="$\mathrm{Local\,maxima}$")

        plot2.scatter(norm_factor*t_print[t_print_indices_min],1.0-e_print[t_print_indices_min],color='b',label="$\mathrm{Local\,}e_\mathrm{in}\mathrm{-minima}$")
        plot2.scatter(norm_factor*t_print[t_print_indices_max],1.0-e_print[t_print_indices_max],color='r',label="$\mathrm{Local\,}e_\mathrm{in}\mathrm{-maxima}$")

        for x in norm_factor*t_print[t_print_indices_max]:
            plot1.axvline(x=x,color='k',linestyle='dotted')
            plot2.axvline(x=x,color='k',linestyle='dotted')

        plot1.axhline(y=90.0,color='k',linestyle='dashed')
    
        plot1.set_xlim(0.0,0.1*norm_factor*t_print[-1])
        plot2.set_xlim(0.0,0.1*norm_factor*t_print[-1])

        plot2.set_xlabel("$t/(L_1/C_2)$",fontsize=fontsize)
        plot1.set_ylabel("$i_\mathrm{rel}/\mathrm{deg}$",fontsize=fontsize)
        plot2.set_ylabel("$1-e$",fontsize=fontsize)
        
        ticks = plot2.get_yticks()
        plot1.set_xticklabels([])
        plot1.set_title(r"$\gamma=%s; \, \theta_0=%s$"%(gamma,theta0),fontsize=fontsize)
        
        plot1.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
        plot2.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
    
        fig.subplots_adjust(hspace=0,wspace=0)

        fig.savefig('figs/diag/diag_gamma_' + str(args.gamma) + '_e0_' + str(args.e0) + '_theta0_' + str(args.theta0) + '_g0_' + str(args.g0) + '.pdf')
        
        if args.show==True:
            pyplot.show()

    return np.mean(e_print[t_print_indices_min]), np.std(e_print[t_print_indices_min]),np.mean(e_print[t_print_indices_max]), np.std(e_print[t_print_indices_max]), np.mean(T_ZLKs), np.std(T_ZLKs), L1_div_C2
    
def plot_function_flip(args):

    if args.plot_fancy==True:
        pyplot.rc('text',usetex=True)
        pyplot.rc('legend',fancybox=True)          

    
    fontsize=22
    labelsize=18
    
    #gamma_points = np.linspace(0.02,0.3,3)
    gamma_points = [0.02,0.1,0.3]
    N=200
    e0_points = np.linspace(0.0,1.0,N)
    theta0_points = np.linspace(-1.0,1.0,N)
    g0_points = [0.0,0.5,1.5]

    N_row = len(g0_points)
    N_col = len(gamma_points)
    
    fig, axs = pyplot.subplots(N_row, N_col,figsize=(12,12))
        
    #ic=0
    #ir=0
    for index_gamma,gamma in enumerate(gamma_points):
        #ir+=1
        for index_g0,g0 in enumerate(g0_points):
            #ic+=1
            print("index_gamma",index_gamma,"index_g0",index_g0)
            
            e0_flip = []
            theta0_flip = []

            for index_e0,e0 in enumerate(e0_points):
                for index_theta0,theta0 in enumerate(theta0_points):
                
                    Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
                    C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)

                    e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
                    e_flip = np.sqrt(1.0 - Theta_ZLK/gamma)
                    if (e_flip >= 0 and e_flip < 1 and e_flip >= e_min and e_flip < e_max):
                        e0_flip.append(e0)
                        theta0_flip.append(theta0)
          
                    #H0 = H_function(e0,theta0,g0)
                    #x_flip = Theta_ZLK/gamma
                    #xdot = np.sqrt(x_flip)*np.sqrt((20.0 + H0 - 18.0*x_flip)*(10.0 - H0 - 12.0*x_flip))
                    #print("xdot",xdot)
            plot=axs[index_g0,index_gamma]
            plot.scatter(theta0_flip,e0_flip,color='k',s=20)
            plot.annotate("$\gamma=%s;\,g_0=%s\,\mathrm{rad}$"%(gamma,g0),xy=(0.1,0.1),xycoords = 'axes fraction',fontsize=fontsize, bbox=dict(boxstyle="Round", fc="white", ec="k", lw=1))

            plot.tick_params(axis='both', which ='major', labelsize = labelsize,bottom=True, top=True, left=True, right=True)
            plot.set_xlim(-1.0,1.0)
            plot.set_ylim(0.0,1.0)
        
            if index_g0 == N_row-1:
                plot.set_xlabel(r"$\theta_0$",fontsize=fontsize)
                ticks = plot.get_xticks()
                plot.set_xticks([-0.5,0,0.5,1.0])
            else:
                plot.set_xticklabels([])
            if index_gamma in [0,3,6]:
                plot.set_ylabel("$e_0$",fontsize=fontsize)
                ticks = plot.get_yticks()
                plot.set_yticks(ticks[1:])
            else:
                plot.set_yticklabels([])
                
    fig.subplots_adjust(wspace=0,hspace=0)
    fig.savefig("figs/fig_flip.pdf")

    pyplot.show()

    
if __name__ == '__main__':
    args = parse_arguments()

    if args.mode==0:
        e0 = args.e0
        theta0 = args.theta0
        gamma = args.gamma
        g0 = args.g0
        
        Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
        C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)
    
        L1_div_C2 = 1.0
        e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
        T_ZLK_an = timescale_function(L1_div_C2,C_ZLK,Theta_ZLK,gamma,e_min,e_max)
        
        print("="*100)
        print("Analytical")

        print("e_min",e_min,"e_max",e_max,"T_ZLK/(L1/C2)",T_ZLK_an)

    if args.mode==1:
        e0 = args.e0
        theta0 = args.theta0
        gamma = args.gamma
        g0 = args.g0
        e_min_mean,e_min_std,e_max_mean,e_max_std,T_ZLK_mean,T_ZLK_std,L1_div_C2 = numerical_integration(args,e0,theta0,gamma,g0)
        print("="*100)
        print("Numerical")
        print("e_min ",e_min_mean," +/- ",e_min_std,"e_max",e_max_mean,"+/-",e_max_std,"T_ZLK/Myr",T_ZLK_mean*1e-6," +/- ",T_ZLK_std*1e-6)
        
        Theta_ZLK = Theta_ZLK_function(e0,theta0,gamma)
        C_ZLK = C_ZLK_function(e0,theta0,gamma,g0)

        e_min,e_max = e_min_e_max_function(C_ZLK,Theta_ZLK,gamma)
        T_ZLK_an = timescale_function(L1_div_C2,C_ZLK,Theta_ZLK,gamma,e_min,e_max)
        
        print("="*100)
        print("Analytical")

        print("e_min",e_min,"e_max",e_max,"T_ZLK/Myr",T_ZLK_an*1e-6,"T_ZLK/(L1/C2)=",T_ZLK_an/L1_div_C2)

        print("="*100)
        print("Fractional difference numerical/analytical -- e_min",(e_min_mean-e_min)/e_min_mean,"e_max",(e_max_mean-e_max)/e_max_mean,"T_ZLK",(T_ZLK_mean-T_ZLK_an)/T_ZLK_mean)
   
    if args.mode==2:
        plot_function_series(args)

    if args.mode==3:
        plot_function_contours(args)

    if args.mode==4:
        plot_function_gamma_ranges(args)

    if args.mode==5:
        plot_function_flip(args)
