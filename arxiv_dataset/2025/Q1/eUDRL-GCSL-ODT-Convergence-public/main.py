# main script for generating all the figures

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import envs
import udrl as udrl


Container = envs.Container

enable_plot_f_dynamics = False 
enable_plot_h_dynamics = False 
enable_plot_z_dynamics = False 
enable_plot_oscilations = False 
enable_plot_tiny_gridworld = False
enable_plot_tiny_gridworld_det_ODT = False
enable_plot_reg_tiny_gridworld_ODT = False
enable_plot_bandit_bounds = False 
enable_plot_reg_bandit_bounds = False 
enable_plot_discontinuity_boundary = False 
enable_plot_discontinuity_det = False
save_figures = True
show_figures = True


ussage_message = """
Usage:

python main.py experiment [donotshowfigs]

where experiment is one of:
f_dynamics, h_dynamics, z_dynamics,
oscilatinos,
tiny_gridworld, tiny_gridworld_det_ODT, reg_tiny_gridworld_ODT, 
bandit_bounds, reg_bandit_bounds,
discontinuity_boundary, discontinuity_det.

The optional donotshowfigs switch disibles plotting figures on screen in blocking mode (useful for noninteractive runs).
"""

assert len(sys.argv) in set([2,3]), "The script accepts one or two arguments, but different number of arguments were given. Abborting."+ussage_message
assert sys.argv[1] in set(["f_dynamics","h_dynamics","z_dynamics",
  "oscilatinos",
  "tiny_gridworld","tiny_gridworld_det_ODT","reg_tiny_gridworld_ODT",
  "bandit_bounds","reg_bandit_bounds",
  "discontinuity_boundary","discontinuity_det"]), f"The {sys.argv[1]} is not a valid argument. Aborting."+ussage_message
if len(sys.argv) == 3:
  assert sys.argv[2] == "donotshowfigs", f"The optinal second argument could be only donotshowfigs but {sys.argv[2]} was given. Aborting."+ussage_message
  show_figures = False
globals()["enable_plot_"+sys.argv[1]] = True




# calculation two figs in the row
# > 433.62pt.
# l.1342     \showthe\linewidth
inches_per_pt = 1.0/72.27
linewidth = inches_per_pt*433.62
two_figs_in_row = 0.48*linewidth # originaly setting: 8,8
grid_fig_size = two_figs_in_row #0.25*linewidth
#two_figs_in_row_axis = (0.125,0.2,0.95-0.125,0.95-0.2)
two_figs_in_row_axis = (0.190,0.130,0.95-0.190,0.95-0.130)
two_figs_in_row_axis_osc = (two_figs_in_row_axis[0]-0.04, two_figs_in_row_axis[1]-0.01, two_figs_in_row_axis[2]+0.04, two_figs_in_row_axis[3]+0.01 )
two_figs_in_row_axis_dyn = (two_figs_in_row_axis[0]-0.03, two_figs_in_row_axis[1]-0.01, two_figs_in_row_axis[2]+0.03, two_figs_in_row_axis[3]+0.01 )
two_figs_in_row_axis_bandit = two_figs_in_row_axis_dyn # two_figs_in_row_axis_osc
two_figs_in_row_axis_reg_bandit = two_figs_in_row_axis_dyn
# computing centred grid axis
gw = 0.5*two_figs_in_row_axis_bandit[2]
gh = 0.5*two_figs_in_row_axis_bandit[3]
gcx = two_figs_in_row_axis_bandit[0]+gw
gcy = two_figs_in_row_axis_bandit[1]+gh
grid_fig_axis = (gcx-0.5*gw,gcy-0.5*gw,gw,gw)


one_figs_in_row = 0.75*linewidth
one_figs_in_row_axis = (0.105,0.085,0.95-0.105,0.95-0.085)




plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({'font.size': 7}) # was 17 default 10; this is for default 0.5\textwidth scale
plt.rcParams.update({'axes.labelsize': 10})
plt.rcParams.update({'legend.fontsize': 10})
plt.rcParams.update({'xtick.labelsize': 8})
plt.rcParams.update({'ytick.labelsize': 8})
plt.rcParams.update({'xtick.major.size':1.5})   #3.5 default
plt.rcParams.update({'ytick.major.size':1.5})
plt.rcParams.update({'xtick.minor.size':0.9})   #2 default
plt.rcParams.update({'ytick.minor.size':0.9})
plt.rcParams.update({'lines.linewidth':0.75})    #1.5 default
plt.rcParams.update({'axes.linewidth':0.25})    #0.8 default
plt.rcParams.update({'xtick.major.width':0.25}) #0.8 default
plt.rcParams.update({'ytick.major.width':0.25})
plt.rcParams.update({'xtick.minor.width':0.20}) #0.6 default
plt.rcParams.update({'ytick.minor.width':0.20})

plt.rcParams.update({'lines.markeredgewidth':0.35}) #1.0 default
plt.rcParams.update({'lines.markersize':3})         #6 default
plt.rcParams.update({'legend.fontsize':10})  # default


if save_figures :
  import os
  try:
    os.mkdir("fig")
  except FileExistsError as error: #OSError
    #print(error)
    pass


# generate f and h dynamic figures

if enable_plot_f_dynamics :
  def f(x,gamma) :
    return x / (x+gamma)


  itN = 13
  xN = 20
  x0 = np.linspace(0,1,xN)
  xn = np.zeros([itN,xN])
  xn[0] = x0

  gamma = 0.65
  for n in range(1,itN) : 
    xn[n] = f(xn[n-1],gamma)

  xs = udrl.x_star(gamma)

  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_dyn)
  plt.plot(xn,"k-",alpha = 0.5)
  plt.plot(np.array([0,itN-1]),xs*np.ones([2]),"r--",alpha = 1,label=r'$x^{*}(\gamma)$')

  plt.legend()
  plt.xlabel(r'$n$')
  plt.ylabel(r'$f_{\gamma}^{\circ n} (x)$')
  plt.xlim(0,itN-1)
  plt.ylim(0,1)
  if save_figures :
    plt.savefig("fig/dynf1.svg")


  gamma = 0.3
  for n in range(1,itN) : 
    xn[n] = f(xn[n-1],gamma)

  xs = udrl.x_star(gamma)

  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_dyn)
  plt.plot(xn,"k-",alpha = 0.5)
  plt.plot(np.array([0,itN-1]),xs*np.ones([2]),"r--",alpha = 1,label=r'$x^{*}(\gamma)$')

  plt.legend()
  plt.xlabel(r'$n$')
  plt.ylabel(r'$f_{\gamma}^{\circ n} (x)$')
  plt.xlim(0,itN-1)
  plt.ylim(0,1)
  if save_figures :
    plt.savefig("fig/dynf2.svg")

  if show_figures :
    plt.show()


if enable_plot_h_dynamics :
  def h(x,b,N) :
    return x**(2*N)/(x**(2*N) + b)

  N = 2
  b0 = udrl.get_b0(N)
  b = b0/2

  itN = 13
  xN = 20
  x0 = np.linspace(0,1,xN)
  xn = np.zeros([itN,xN])
  xn[0] = x0

  for n in range(1,itN) : 
    xn[n] = h(xn[n-1],b,N)

  xl,xu = udrl.get_xl_xu(N,b)

  plt.figure().set_size_inches(one_figs_in_row, one_figs_in_row)
  plt.axes(one_figs_in_row_axis)
  plt.plot(xn,"k-",alpha = 0.5)
  plt.plot(np.array([0,itN-1]),xu*np.ones([2]),"r--",alpha = 1,label=r'$x_u(b)$')
  plt.plot(np.array([0,itN-1]),xl*np.ones([2]),"b--",alpha = 1,label=r'$x_l(b)$')

  plt.legend()
  plt.xlabel(r'$n$')
  plt.ylabel(r'$h_{b}^{\circ n} (x)$')
  plt.xlim(0,itN-1)
  plt.ylim(0,1)
  if save_figures :
    plt.savefig("fig/dynh.svg")
  if show_figures :
    plt.show()

if enable_plot_z_dynamics :
  def z(x,gamma,epsilon,M,A) :
    return (1-epsilon)*x/(x + gamma) + epsilon*M/A

  A = 4
  M = 1
  gamma = 0.4
  epsilon = 0.1

  itN = 13
  xN = 20
  x0 = np.linspace(0,1,xN)
  xn = np.zeros([itN,xN])
  xn[0] = x0

  for n in range(1,itN) : 
    xn[n] = z(xn[n-1],gamma,epsilon,M,A)

  #xl,xu = udrl.get_xl_xu(N,b)
  xs = udrl.reg_get_xstar(gamma,epsilon,M,A)

  plt.figure().set_size_inches(one_figs_in_row, one_figs_in_row)
  plt.axes(one_figs_in_row_axis)
  plt.plot(xn,"k-",alpha = 0.5)
  plt.plot(np.array([0,itN-1]),xs*np.ones([2]),"r--",alpha = 1,label=r'$x^*(\gamma,\epsilon,M), \epsilon=0.1$')


  epsilon = 0.3
  xn = np.zeros([itN,xN])
  xn[0] = x0
  for n in range(1,itN) : 
    xn[n] = z(xn[n-1],gamma,epsilon,M,A)
  xs = udrl.reg_get_xstar(gamma,epsilon,M,A)

  plt.plot(xn,"g-",alpha = 0.5)
  plt.plot(np.array([0,itN-1]),xs*np.ones([2]),"c--",alpha = 1,label=r'$x^*(\gamma,\epsilon,M), \epsilon=0.3$')

  
  plt.legend()
  plt.xlabel(r'$n$')
  plt.ylabel(r'$z_{\gamma,\epsilon,M}^{\circ n} (x)$')
  plt.xlim(0,itN-1)
  plt.ylim(0,1)
  if save_figures :
    plt.savefig("fig/dynz.svg")
  if show_figures :
    plt.show()


def compute_udrl_trajectories(env_call, itN, ker_stoch_range, icN, pi0ar = None, ic_mode_ray = False, ic_mode_ker_common = False, ic_mode_scaled = True , epsilon=0.0) :
  env_det = env_call(tr_kernel_stochasticty= 0)  # envs.make_env_cycle(tr_kernel_stochasticty = 0,N = 2,S = 2 ) #teleport_prob = 2**(-2) , alpha = 1-2**(-2)
  the_interesting_states, Opt_det, rho_det, supp_d00 = udrl.get_the_interesting_states(env_det)
  computeM1bouds = udrl.test_M1(env_det, the_interesting_states, Opt_det) # computateon of x_u,x_l bound if applicable
  computeMUbouds = udrl.test_mu_supp(env_det,the_interesting_states) # computateon of x_star bound if applicable
  computeregbouds = (epsilon > 0.0)

  if computeM1bouds :
    N = env_det.N
    b0 = udrl.get_b0(N)

  if True:  #pi0ar is None :
    # generation of initial conditions
    pi_eps = 1e-3 # to guarantie pi > 0
    pi_init = env_det.pi_init_uniform

    pi0ar = np.random.random([kerN, icN]+list(pi_init.shape)) + pi_eps if pi0ar is None else pi0ar
    action_dim = len(pi0ar.shape)-4
    pi0ar = pi0ar/pi0ar.sum(axis = action_dim, keepdims = True )
    if ic_mode_ray :
      pi0ar[...] = pi0ar[:,0,...][:,None]
    if ic_mode_ker_common :
      pi0ar[...] = pi0ar[0,...]
    if ic_mode_scaled :
      min_piMreq = np.linspace(pi_eps,1,icN+2)
      min_piMreq = min_piMreq[1:-1]
      min_piMreq = np.outer(np.ones([kerN]),min_piMreq )
      min_piM = udrl.min_piM_projection(pi0ar,Opt_det,the_interesting_states)
      M = (Opt_det.pi_supp)
      non_M = np.logical_not(M)      
      for ki in range(kerN) :
        for ici in range(icN) :
          pi = pi0ar[ki,ici]
          #min_piM[ki,ici]
          #min_piMreq[ki,ici]
          for s in range(env_det.S) :
            for h in range(env_det.N) :
              for g in range(env_det.G) :
                piM = np.sum(pi[:,s,h,g]*(M[:,s,h,g].astype(float)))
                newpiM = 1-(1-piM)/(1-min_piM[ki,ici])*(1-min_piMreq[ki,ici])
                pi[M[:,s,h,g],s,h,g] = pi[M[:,s,h,g],s,h,g]*(newpiM/piM)
                if piM < 1 :
                  pi[non_M[:,s,h,g],s,h,g] = pi[non_M[:,s,h,g],s,h,g]/np.sum(pi[non_M[:,s,h,g],s,h,g])*(1-newpiM)
          pi0ar[ki,ici] = pi
                
    # # generation of initial conditions
    # pi_eps = 1e-3 # to guarantie pi > 0
    # pi_init = env_det.pi_init_uniform
    # pi0ar = np.random.random([icN]+list(pi_init.shape)) + pi_eps
    # action_dim = len(pi0ar.shape)-4
    # pi0ar = pi0ar/pi0ar.sum(axis = action_dim, keepdims = True )
  else :
    if icN != len(pi0ar) :
      raise Exception("if pi0ar is given its length must agree with icN")

  piar = np.zeros([kerN,icN,itN+1] + list(env_det.pi_init_uniform.shape))
  Jar = np.zeros([kerN,icN,itN])
  

  deltaar = np.zeros([kerN])
  if computeMUbouds :
    piMboundar = np.zeros([kerN])
  if computeM1bouds :
    xuboundar = np.zeros([kerN])
    xlboundar = np.zeros([kerN])
    bar = np.zeros([kerN])
  if computeregbouds :
    regpiMboundar = np.zeros([kerN])


  for ki in range(kerN) :
    print(f"{ki}/{kerN} done")
    for ici in range(icN) :
      env = env_call(tr_kernel_stochasticty= ker_stoch_range[ki]) # envs.make_env_cycle(tr_kernel_stochasticty = ker_stoch_range[ki],N = 2,S = 2 ) #teleport_prob = 2**(-2) , alpha = 1-2**(-2)
      pi_init = pi0ar[ki,ici]    
      CUDRL = udrl.classical_udrl_theory2(env,itN,pi_init,epsilon=epsilon)
      piar[ki,ici,1:] = CUDRL.pi_saved
      piar[ki,ici,0] = CUDRL.pi_init
      deltaar[ki] = udrl.norm1_dist(env.E,env_det.E)
      Jar[ki,ici,:] = CUDRL.J_saved      

      if computeMUbouds :
        #alpha, beta, nu, betaN = udrl.get_alpha_beta_nu(env,env_det)
        alpha, betas, beta, nu, gamma, betaN, nuN, gammaN = udrl.get_alpha_beta_nu(env,env_det)
        #piMboundar[ki] = udrl.x_star(betaN,alpha)
        piMboundar[ki] = udrl.x_star(gammaN)
      if computeM1bouds :
        b, xl, xu = udrl.get_b_xl_xu_from_env(env,env_det) #udrl.get_b_xl_xu(env,env_det,b0)        
        bar[ki] = b
        xuboundar[ki] = xu
        xlboundar[ki] = xl
      if computeregbouds :
        alpha, betas, beta, nu, gamma, betaN, nuN, gammaN, xsmin = udrl.reg_get_alpha_beta_nu(env,env_det,epsilon)
        regpiMboundar[ki] = xsmin


  piMar = udrl.min_piM_projection(piar,Opt_det,the_interesting_states)
  
  traj = Container()
  traj.computeM1bouds = computeM1bouds
  traj.computeMUbouds = computeMUbouds
  traj.computeregbouds = computeregbouds
  traj.deltaar = deltaar
  traj.piMar = piMar
  traj.piar = piar
  traj.Jar = Jar
  if computeMUbouds :
    traj.piMboundar = piMboundar
  if computeM1bouds :
    traj.xuboundar = xuboundar
    traj.xlboundar = xlboundar
    traj.bar = bar
  if computeregbouds :
    traj.regpiMboundar = regpiMboundar

  return traj


if enable_plot_oscilations :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 20 #10
  icN = 10 #1 # number of initial conditions
  kerN = 5 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.linspace(0,0.5,kerN) #np.logspace(-5,0,kerN)

  def env_call(tr_kernel_stochasticty) :
    #env = envs.make_env_cycle(tr_kernel_stochasticty = tr_kernel_stochasticty,N = 2,S = 2)
    env = envs.make_env_cycle(tr_kernel_stochasticty = tr_kernel_stochasticty,N = 8,S = 3) # some non-monotony in J ki=1/5 ici=6/10
    #env = envs.make_env_cycle(tr_kernel_stochasticty = tr_kernel_stochasticty,N = 15,S = 5) # some non-monotony in J ki=1/5 ici=2/10
    return env

  # call for picking interesting initial conditions
  traj = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN)

  #pi0ar = np.random.random([kerN, icN]+list(pi_init.shape)) + pi_eps
  #piar = np.zeros([kerN,icN,itN+1] + list(env_det.pi_init_uniform.shape))  

  #icN = 1
  pi0ar = np.zeros(traj.piar[:,:,0].shape)
  pi0ar[:,:] = traj.piar[1,:,0] 

  #pi0ar = None, ic_mode_ray = False, ic_mode_ker_common = False, ic_mode_scaled = True 
  traj = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, pi0ar = pi0ar, ic_mode_ray = False, ic_mode_ker_common = False, ic_mode_scaled = False )



  colors = sns.color_palette("colorblind", icN//2)
  ic_highlight = 6

  #plt.rcParams.update({'font.size': 12}) # default 19; this is for 0.75\textwidth scale

  plt.figure().set_size_inches(one_figs_in_row, one_figs_in_row)  
  plt.axes(one_figs_in_row_axis)
  nax = np.arange(0,itN+1)
  txtpos = itN-4.5
  idx = itN-4
  for ki in range(kerN) :
    for ici in range(0,icN,2) : #range(6,7):
      #ici = 0
      alpha = 0.5
      color=colors[ici//2]
      if ici == ic_highlight :
        alpha = 1.0
        color = (0,0,0)
        plt.text(txtpos, traj.piMar[ki,ici,idx]+0.01,r'$\delta = '+ f"{traj.deltaar[ki]}" + r'$') #, ic={ici}
      plt.plot(nax, traj.piMar[ki,ici,:], color=color,alpha = alpha ) #ici/icN (ki/kerN,0,0)

  plt.xlabel(r'$n$')  
  plt.ylabel(r'$\min_{\bar{s}\in \bar{S}_{\lambda_0}}\pi(\mathcal{O}(\bar{s})|\bar{s})$',labelpad = 0.0)  #labelpad = 0.0
  plt.xlim(0,itN)
  plt.ylim(0,1.1)
  if save_figures :
    plt.savefig("fig/oscpi.svg")

  #plt.rcParams.update({'font.size': 17}) # default 19; this is for 0.75\textwidth scale
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_osc)
  nax = np.arange(0,itN)
  for ki in range(kerN) : #range(1,2):
    for ici in range(0,icN,2) : #range(2,3)
      #ici = 0  
      alpha = 0.5
      color=colors[ici//2]
      if ici == ic_highlight :
        alpha = 1.0
        color = (0,0,0)        
        plt.text(txtpos, traj.Jar[ki,ici,idx]-0.0055,r'$\delta = '+ f"{traj.deltaar[ki]}" + r'$', verticalalignment='top') #, ic={ici}      
      plt.plot(nax, traj.Jar[ki,ici,:], color=color, alpha = alpha ) #ici/icN color=(ki/kerN,0,0)            

  plt.xlabel(r'$n$')  
  plt.ylabel(r'$J^{\pi_n}$')  
  plt.xlim(0,itN-1)
  #plt.ylim(0,1.1)
  if save_figures :
    plt.savefig("fig/oscJ.svg")


  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_osc)
  nax = np.arange(0,itN)  
  #for ki in range(kerN) : #range(1,2):
    #for ici in range(0,icN,2) : #range(2,3)

  ki = 1
  ici = ic_highlight
  alpha = 0.5
  color=colors[ici//2]
  if ici == ic_highlight :
    alpha = 1.0
    color = (0,0,0)
    plt.text(txtpos, traj.Jar[ki,ici,idx]+0.001,r'$\delta = '+ f"{traj.deltaar[ki]}" + r'$') #, ic={ici}
  plt.plot(nax, traj.Jar[ki,ici,:], color=color, alpha = alpha ) #ici/icN color=(ki/kerN,0,0)  

  plt.xlabel(r'$n$')  
  plt.ylabel(r'$J^{\pi_n}$',labelpad = 0.0)  
  plt.xlim(0,itN-1)
  #plt.ylim(0,1.1)
  if save_figures :
    plt.savefig("fig/oscJdetail.svg")


  #if not save_figures :
  if show_figures :
    plt.show()


def traj_piM_plot(traj,icN,itN,ki_list,ki_highlight = None) :
  colors = sns.color_palette("colorblind", len(ki_list))
  nax = np.arange(0,itN+1)
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_bandit)
  #ax = plt.axes(projection='3d')
  for i,ki in enumerate(ki_list) : #range(kerN) :
    for ici in range(icN) :
      #ax.plot(nax, traj.deltaar[ki]*np.ones(itN+1), traj.piMar[ki,ici,:], color=(ki/kerN,0,0))    
      if ici == 0 :
        plt.plot(nax, traj.piMar[ki,ici,:], color=colors[i], label=r'$\delta = '+f"{traj.deltaar[ki]:.2f}"+r'$')
      else:
        plt.plot(nax, traj.piMar[ki,ici,:], color=colors[i])
  plt.xlabel(r'$n$')  
  plt.ylabel(r'$\min_{\bar{s}\in \bar{S}_{\lambda_0}}\pi_n(\mathcal{O}(\bar{s})|\bar{s})$')
  plt.legend()
  plt.ylim(0,1)
  plt.xlim(0,itN)


def asymptotic_piM_plot(traj,icN,itN, plot_other_it = None, just_x_star = False, just_x_star_reg = False, onefig = False) :  
  colors = sns.color_palette("colorblind", 4)
  if onefig :
    plt.figure().set_size_inches(one_figs_in_row, one_figs_in_row)
    plt.axes(one_figs_in_row_axis)
  else:
    plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
    plt.axes(two_figs_in_row_axis_bandit)
  plt.xscale("log")
  if plot_other_it is not None:    
    for i,it in enumerate(plot_other_it) :
      if it == 0 :
        color = colors[0]
      elif it == itN :
        print(f"coloring it={it}, by {colors[3]} ") 
        color = colors[3]
      else :
        color = np.ones([3])*i/len(plot_other_it)
      if it == 0 :
        plt.plot(np.kron(traj.deltaar,np.ones(icN)),  traj.piMar[:,:,it].flatten() ,".",markersize=1.5,color = color,label=r'$n=' + f"{it}"  + r'$')
      else :
        plt.plot(np.kron(traj.deltaar,np.ones(icN)),  traj.piMar[:,:,it].flatten() ,".",color = color,label=r'$n=' + f"{it}"  + r'$')
  else :    
    plt.plot(np.kron(traj.deltaar,np.ones(icN)),  traj.piMar[:,:,itN].flatten(),".",color = colors[3],label=r'$n=' + f"{itN}" + r'$') 
    if traj.computeMUbouds and not just_x_star_reg :
      plt.plot(traj.deltaar,traj.piMboundar,color = colors[0],label=r'$x^{*}(\gamma_N)$')
    if traj.computeM1bouds and not just_x_star and not just_x_star_reg :
      plt.plot(traj.deltaar,traj.xuboundar,color = colors[1],label=r'$x_u(\delta)$')
      plt.plot(traj.deltaar,traj.xlboundar,color = colors[2],label=r'$x_l(\delta)$')
    if traj.computeregbouds and not just_x_star :
      plt.plot(traj.deltaar,traj.regpiMboundar,color = colors[0],label=r'$\min_{M}x^{*}(\gamma_N,\epsilon,M)$')


  plt.xlabel(r'$\delta$')
  if just_x_star_reg :
    plt.ylabel(r'$\min_{\bar{s}\in \bar{S}_{\lambda_0}}\pi_{n,\epsilon}(\mathcal{O}(\bar{s})|\bar{s})$')  
  else:   
    plt.ylabel(r'$\min_{\bar{s}\in \bar{S}_{\lambda_0}}\pi_n(\mathcal{O}(\bar{s})|\bar{s})$')  
  plt.xlim(traj.deltaar[0],traj.deltaar[-1])
  plt.ylim(0,1)
  plt.legend()
  

def plot_grid( env_det, goal_state = None, shift_arrows_off_center = False) :
  #env_det = env_call(tr_kernel_stochasticty= 0)
  the_interesting_states, Opt_det, rho_det, supp_d00 = udrl.get_the_interesting_states(env_det)
  H,W = env_det.map.shape
  fig = plt.figure().set_size_inches(grid_fig_size, grid_fig_size) #figsize = (0.5*W,0.5*H)  
  #fig.set_size_inches(3,3)
  plt.axes(grid_fig_axis)
  img = np.ones(list(env_det.map.shape) + [3])
  img[env_det.map == 1,:] = 0
  for (s,h,g) in np.argwhere(the_interesting_states) :
    i,j = env_det.state_pos[s]
    img[i,j] = (1.0,1.0,0)
  if goal_state is not None :
    img[goal_state[0],goal_state[1]] = (1.0,0,0)
  plt.imshow(img,alpha=0.5)
  def plotarrow(pos,dir,scale,color) :
    odir = dir.copy()
    odir[0] = dir[1]
    odir[1] = -dir[0]
    pts = np.zeros([4,2])
    pts[0] = dir
    pts[1] = odir/4
    pts[2] = -odir/4
    pts[3] = dir
    if shift_arrows_off_center :
      pts += dir/3
    pts = pts*scale
    pts[:,0] += pos[1]
    pts[:,1] += pos[0]
    plt.plot(pts[:,0],pts[:,1],color=color)
    return
  def plotoptactions(goal_state,color) :
    gi,gj = goal_state
    goal = env_det.goal_projection[env_det.map_states[gi,gj]]
    for s in range(env_det.S) :
      if the_interesting_states[s,:,goal].sum() > 0 :
        minh = np.min(np.argwhere(the_interesting_states[s,:,goal]))
        M = Opt_det.pi_supp[:,s,minh,goal]
        scale = 0.25
        #print(f"(s,h,g) = {s,minh,goal}, (i,j)={env_det.state_pos[s]}, M={M}, h={the_interesting_states[s,:,goal]}")
        if M[0] :        
          plotarrow(env_det.state_pos[s],np.array([1,0]),scale,color) # right
        if M[1] :        
          plotarrow(env_det.state_pos[s],np.array([-1,0]),scale,color) # left
        if M[2] :        
          plotarrow(env_det.state_pos[s],np.array([0,-1]),scale,color) # up
        if M[3] :        
          plotarrow(env_det.state_pos[s],np.array([0,1]),scale,color) # down
  if goal_state is not None :
    plotoptactions(goal_state,color=(1.0,0,0))
  for i in range(H) :
    plt.plot(i*np.ones([2])-0.5,[0-0.5,W-0.5],"k-",alpha=0.5)
  for j in range(W) :
    plt.plot([0-0.5,H-0.5], j*np.ones([2])-0.5, "k-",alpha=0.5)
  plt.xticks(np.arange(W, dtype=int))
  plt.yticks(np.arange(H, dtype=int))  


if enable_plot_tiny_gridworld :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 40 #10
  icN = 20 #20 #2 # number of initial conditions
  kerN = 40 #5 # 20 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.logspace(-8,0,kerN)  #np.linspace(0,0.5,kerN) #-21

  def env_call(tr_kernel_stochasticty) :
    env = envs.make_env_gridworld_tiny(tr_kernel_stochasticty= tr_kernel_stochasticty)  
    return env

  traj = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN)

  traj_piM_plot(traj,icN,itN, ki_list = [0,36,37,38])
  if save_figures :
    plt.savefig(f"fig/gridpitraj.svg")

  asymptotic_piM_plot(traj,icN,itN, just_x_star = True)
  if save_figures :
    plt.savefig(f"fig/gridpilimstar.svg")

  asymptotic_piM_plot(traj,icN,itN, plot_other_it = [0,1,2,5,10,20,itN])
  if save_figures :
    plt.savefig(f"fig/gridpilimmap.svg")


  env_det = env_call(tr_kernel_stochasticty= 0)
  #goal_state = env_det.goal_state
  goal_state = np.array([0,2],dtype = int) #None #np.array([0,2],dtype = int)

  plot_grid( env_det, goal_state = goal_state, shift_arrows_off_center = True)
  if save_figures :
    plt.savefig(f"fig/gridmap.svg")
  
  if show_figures :
    plt.show()





if enable_plot_tiny_gridworld_det_ODT :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 20 #400
  icN = 10 #5 #20 # number of initial conditions
  kerN = 80 #20 #5 # 20 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.logspace(-5,0,kerN)  #-8,0


  def env_call(tr_kernel_stochasticty) :
    env = envs.make_env_gridworld_tiny_det(tr_kernel_stochasticty= tr_kernel_stochasticty)
    env = envs.transform_to_ODT(env=env,K=3)
    return env

  traj = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN)

  traj_piM_plot(traj,icN,itN, ki_list = [0,72,73,74,75]) #[0,15,16,17,18]
  if save_figures :
    plt.savefig(f"fig/griddetpitraj.svg")


  asymptotic_piM_plot(traj,icN,itN)
  if save_figures :
    plt.savefig(f"fig/griddetpilim.svg")

  asymptotic_piM_plot(traj,icN,itN, plot_other_it = [0,1,2,5,10,itN])
  if save_figures :
    plt.savefig(f"fig/griddetpilimmap.svg")



  env_det = env_call(tr_kernel_stochasticty= 0).under
  goal_state = env_det.goal_state
  #goal_state = np.array([0,2],dtype = int) #None #np.array([0,2],dtype = int)
  plot_grid( env_det, goal_state = goal_state)
  if save_figures :
    plt.savefig(f"fig/griddetmap.svg")
  
  if show_figures :
    plt.show()


if enable_plot_reg_tiny_gridworld_ODT :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 20 #400
  icN = 5 #20 # number of initial conditions
  kerN = 40 #20 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.logspace(-14,0,kerN)  #-8 #-18,0

  def env_call(tr_kernel_stochasticty) :
    env = envs.make_env_gridworld_tiny_nondet(tr_kernel_stochasticty= tr_kernel_stochasticty)
    env = envs.transform_to_ODT(env=env,K=3)
    return env

  traj01 = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, epsilon=0.1)


  asymptotic_piM_plot(traj01,icN,itN, just_x_star_reg = True)
  if save_figures :
    plt.savefig(f"fig/reggridpilim.svg")


  env_det = env_call(tr_kernel_stochasticty= 0).under
  goal_state = env_det.goal_state
  #goal_state = np.array([0,2],dtype = int) #None #np.array([0,2],dtype = int)
  plot_grid( env_det, goal_state = goal_state)
  if save_figures :
    plt.savefig(f"fig/reggridmap.svg")
  
  if show_figures :
    plt.show()




if enable_plot_bandit_bounds :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 100 #400
  icN = 40 #20 # number of initial conditions
  kerN = 40 #5 # 20 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.logspace(-2,0,kerN)  #np.linspace(0,0.5,kerN) #-21

  def env_call(tr_kernel_stochasticty) :
    env = envs.make_env_bandit(tr_kernel_stochasticty = tr_kernel_stochasticty)  
    return env

  traj = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN)


  traj_piM_plot(traj,icN,itN, ki_list = [0,10,20,30,34])
  if save_figures :
    plt.savefig(f"fig/banditpitraj.svg")


  asymptotic_piM_plot(traj,icN,itN, just_x_star = True)
  if save_figures :
    plt.savefig(f"fig/banditpilimstar.svg")


  asymptotic_piM_plot(traj,icN,itN, onefig = True)
  if save_figures :
    plt.savefig(f"fig/banditpilim.svg")

  asymptotic_piM_plot(traj,icN,itN, plot_other_it = [0,1,2,5,10,20,itN])
  if save_figures :
    plt.savefig(f"fig/banditpilimmap.svg")


  if show_figures :
    plt.show()


if enable_plot_reg_bandit_bounds :

  ic_seed = 12345
  np.random.seed(ic_seed)

  itN = 100 #400
  icN = 40 #20 # number of initial conditions
  kerN = 40 #5 # 20 # number of kernels (varying stochasticity of transition kernel)
  ker_stoch_range = np.logspace(-4,0,kerN)  #np.linspace(0,0.5,kerN) #-21

  def env_call(tr_kernel_stochasticty) :
    env = envs.make_env_bandit(tr_kernel_stochasticty = tr_kernel_stochasticty)  
    return env

  traj01 = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, epsilon = 0.1)
  traj002 = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, epsilon = 0.02)
  traj02 = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, epsilon = 0.2)
  traj04 = compute_udrl_trajectories(env_call= env_call, itN = itN, ker_stoch_range=ker_stoch_range, icN=icN, epsilon = 0.4)



  # reg limit points + reg bounds for various eps
  colors = sns.color_palette("colorblind", 4)
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_reg_bandit)
  plt.xscale("log")
  plt.plot(np.kron(traj01.deltaar,np.ones(icN)),  traj002.piMar[:,:,itN].flatten(),".",color = colors[0]) #,label=r'$n=' + f"{itN}" + r'$'
  plt.plot(traj01.deltaar,traj002.regpiMboundar,color = colors[0],label=r'$\epsilon=0.02$')    
  plt.plot(np.kron(traj01.deltaar,np.ones(icN)),  traj01.piMar[:,:,itN].flatten(),".",color = colors[3]) #,label=r'$n=' + f"{itN}" + r'$'
  plt.plot(traj01.deltaar,traj01.regpiMboundar,color = colors[3],label=r'$\epsilon=0.1$')
  plt.plot(np.kron(traj01.deltaar,np.ones(icN)),  traj02.piMar[:,:,itN].flatten(),".",color = colors[1]) #,label=r'$n=' + f"{itN}" + r'$'
  plt.plot(traj01.deltaar,traj02.regpiMboundar,color = colors[1],label=r'$\epsilon=0.2$')
  plt.plot(np.kron(traj01.deltaar,np.ones(icN)),  traj04.piMar[:,:,itN].flatten(),".",color = colors[2]) #,label=r'$n=' + f"{itN}" + r'$'
  plt.plot(traj01.deltaar,traj04.regpiMboundar,color = colors[2],label=r'$\epsilon=0.4$')

  plt.xlabel(r'$\delta$')  
  plt.ylabel(r'$\min_{\bar{s}\in \bar{\mathcal{S}}_{\lambda_0}}\pi_{n,\epsilon}(\mathcal{O}(\bar{s})|\bar{s}), n=100$')  
  plt.xlim(traj01.deltaar[0],traj01.deltaar[-1])
  plt.ylim(0,1)
  plt.legend()

  if save_figures :
    plt.savefig(f"fig/regbanditpilimstar.svg")

  # plot of all the bounds + reg bounds for varios eps

  colors = sns.color_palette("colorblind", 4)
  plt.figure().set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis_reg_bandit)
  plt.xscale("log")
  if traj01.computeMUbouds:
    plt.plot(traj01.deltaar,traj01.piMboundar,color = colors[0],linestyle='dashed', label=r'$x^{*}(\gamma_N)$')
  if traj01.computeM1bouds:
    plt.plot(traj01.deltaar,traj01.xuboundar,color = colors[1],linestyle='dashed', label=r'$x_u(\delta)$')
    plt.plot(traj01.deltaar,traj01.xlboundar,color = colors[2],linestyle='dashed', label=r'$x_l(\delta)$')
  if traj01.computeregbouds:
    plt.plot(traj01.deltaar,traj002.regpiMboundar,color = colors[0],label=r'$\epsilon=0.02$')
    plt.plot(traj01.deltaar,traj01.regpiMboundar,color = colors[3],label=r'$\epsilon=0.1$')    
    plt.plot(traj01.deltaar,traj02.regpiMboundar,color = colors[1],label=r'$\epsilon=0.2$')
    plt.plot(traj01.deltaar,traj04.regpiMboundar,color = colors[2],label=r'$\epsilon=0.4$')

  plt.xlabel(r'$\delta$')  
  plt.xlim(traj01.deltaar[0],traj01.deltaar[-1])
  plt.ylim(0,1)
  plt.legend()
  if save_figures :
     plt.savefig(f"fig/regbanditbndcompar.svg")

  if show_figures :
    plt.show()



def compute_ray(a,trmatrix_call, goal_init_dist) :
  N = len(a)
  J = np.zeros([N])
  pi_comp = np.zeros([N])
  for n in range(N) :
    trmatrix = trmatrix_call(a[n])
    env = envs.make_env_bandit_matrix(trmatrix, goal_init_dist)
    CUDRL = udrl.classical_udrl_theory2(env=env, NconvT=2 )
    pi_comp[n] = CUDRL.pi[2,0,0,1] #A,S,N,G
    Q,V = udrl.pi_values(env,CUDRL.pi)
    J[n] = np.sum(env.PC_init*V)
  return J,pi_comp 


def plot_discontinuity(trmatrix_call_A, trmatrix_call_C, goal_init_dist, suffix = "", ticks=None, intro=False) :
  eps = 1e-3
  N = 100
  a = np.linspace(0+eps,0.6,N)
  a0 = np.zeros([1]) 

  J_A,pi_comp_A = compute_ray(a, trmatrix_call_A, goal_init_dist)
  J_B,pi_comp_B = compute_ray(a0, trmatrix_call_A, goal_init_dist)
  J_C,pi_comp_C = compute_ray(a, trmatrix_call_C, goal_init_dist)

  colors = sns.color_palette("colorblind", 3)

  #plt.rcParams.update({'font.size': 16}) # default 10; this is for 0.25\textwidth scale
  plt.figure(1).set_size_inches(two_figs_in_row, two_figs_in_row)  
  plt.axes(two_figs_in_row_axis)
  plt.plot(a,pi_comp_A ,"-",color=colors[0], label=r'$A$')
  plt.plot(a0,pi_comp_B,"o--",color=colors[2], label=r'$B$')
  plt.plot(a,pi_comp_B*np.ones([N]),"--",color=colors[2])
  plt.plot(a,pi_comp_C ,"-",color=colors[1], label=r'$C$')
  plt.legend()  
  plt.xlabel(r'$\alpha$')
  plt.ylabel(r'$\pi_{2,\alpha}(a=2|g=1)$')
  plt.xlim(0,a[-1])
  # if ticks is not None :
  #   plt.yticks(ticks=ticks["pi"]["pos"], labels=ticks["pi"]["labels"])
  if save_figures :
    plt.savefig(f"fig/dispi{suffix}.svg")

  #plt.rcParams.update({'font.size': 16}) # default 10; this is for 0.25\textwidth scale
  plt.figure(2).set_size_inches(two_figs_in_row, two_figs_in_row)
  plt.axes(two_figs_in_row_axis)
  plt.plot(a,J_A ,"-",color=colors[0], label=r'$A$')
  plt.plot(a0,J_B,"o--",color=colors[2], label=r'$B$')
  plt.plot(a,J_B*np.ones([N]),"--",color=colors[2])
  plt.plot(a,J_C ,"-",color=colors[1], label=r'$C$')
  plt.legend()
  plt.xlabel(r'$\alpha$')
  if intro :
    suffix = suffix+"intro"
    plt.ylabel(r'$J_{\alpha}$')
  else :
    plt.ylabel(r'$J^{\pi_{2}}_{\alpha}$')

  plt.xlim(0,a[-1])
  # if ticks is not None :
  #   plt.yticks(ticks=ticks["J"]["pos"], labels=ticks["J"]["labels"])
  if save_figures :
    plt.savefig(f"fig/disJ{suffix}.svg")

  if show_figures :
    plt.show()



if enable_plot_discontinuity_boundary :

  def trmatrix_call_A(a) :
    return np.array([[1-a,a/4,3*a/4],[3*a/4,1-a,a/4],[1/2,1/2,0]])

  def trmatrix_call_C(a) :
    return np.array([[1-a,3*a/4,a/4],[a/4,1-a,3*a/4],[1/2,1/2,0]])

  goal_init_dist = np.array([0.5,0.0,0.5])
  
  plot_discontinuity(trmatrix_call_A, trmatrix_call_C, goal_init_dist, "boundry", ticks={"pi":{"pos":[0.2,0.3,0.4], "labels":["0.2","0.3","0.4"]},"J":{"pos":[0.4,0.45], "labels":[".40",".45"]}})
  plot_discontinuity(trmatrix_call_A, trmatrix_call_C, goal_init_dist, "boundry", ticks={"pi":{"pos":[0.2,0.3,0.4], "labels":["0.2","0.3","0.4"]},"J":{"pos":[0.4,0.45], "labels":[".40",".45"]}},intro=True)


if enable_plot_discontinuity_det :

  def trmatrix_call_A(a) :
    return np.array([[1-a,a,0],[0,1-a,a],[a,1-a,0]])

  def trmatrix_call_C(a) :
    return np.array([[1-a,0,a],[a,1-a,0],[0,1-a,a]])

  goal_init_dist = np.ones([3])/3

  plot_discontinuity(trmatrix_call_A, trmatrix_call_C, goal_init_dist, "det", ticks={"pi":{"pos":[0.3,0.4,0.5,0.6], "labels":["0.3","0.4","0.5","0.6"]},"J":{"pos":[0.45,0.50,0.55], "labels":[".45",".50",".55"]}})
  plot_discontinuity(trmatrix_call_A, trmatrix_call_C, goal_init_dist, "det", ticks={"pi":{"pos":[0.3,0.4,0.5,0.6], "labels":["0.3","0.4","0.5","0.6"]},"J":{"pos":[0.45,0.50,0.55], "labels":[".45",".50",".55"]}},intro=True)





