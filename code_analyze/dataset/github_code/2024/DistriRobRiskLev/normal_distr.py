# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize
import casadi as ca


# %%
# Build a family of normal distributions
np.random.seed(0)
N_distr= 25
µ_list = np.random.uniform(-1,1,N_distr)
sig_list = np.random.uniform(1,2,N_distr)
#eps_grid = np.linspace(1e-6,1,101)
# Find the nominal distribution with the smallest maximum RVD
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
RVD_max = ca.SX.sym('RVD_max')
J=RVD_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    RVD = sig_nom/sig*ca.exp(0.5*((µ_nom-µ)**2/(sig_nom**2-sig**2)))
    g.append(RVD_max-RVD)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(np.max(sig_list))
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,RVD_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_RVD=res['x'][0]
sig_nom_RVD=res['x'][1]
M_RVD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum RVD is: µ=',µ_nom_RVD,' sig=',sig_nom_RVD,' with maximum RVD=',M_RVD)
# %%
# Now we calculate the KL divergence / relative entropy
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M = 1/2*((sig/sig_nom)**2+(µ-µ_nom)**2/sig_nom**2-1+ca.log(sig_nom**2/sig**2))
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(0)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_KLD=res['x'][0]
sig_nom_KLD=res['x'][1]
M_KLD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum KL divergence is: µ=',µ_nom_KLD,' sig=',sig_nom_KLD,' with maximum KL divergence=',M_KLD)
# %%
# Now we calculate the Hellinger distance
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M = ca.sqrt(1-ca.sqrt((2*sig*sig_nom)/(sig**2+sig_nom**2))*ca.exp(-0.25*((µ-µ_nom)**2/(sig**2+sig_nom**2))))
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(0)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_HD=res['x'][0]
sig_nom_HD=res['x'][1]
M_HD=res['x'][2]

# %%
print('The nominal distribution with the smallest maximum Hellinger distance is: µ=',µ_nom_HD,' sig=',sig_nom_HD,' with maximum Hellinger distance=',M_HD)
# %%
# Now we calculate the chi_square distance
# But for this we need to evaluate the integral numerically
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
x_linspace = np.linspace(np.min([µ_list])-20*np.max([sig_list]),np.max([µ_list])+20*np.max([sig_list]),100)
# Define a normal distribution as a casadi function
x = ca.SX.sym('x')
ND = ca.Function('ND',[x,µ_nom,sig_nom],[ca.exp(-0.5*((x-µ_nom)/sig_nom)**2)/(sig_nom*ca.sqrt(2*ca.pi))])
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M=0
    print('Iteration: ',i)
    for j in range(len(x_linspace)-1):
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*((ND(x_linspace[j],µ,sig)-ND(x_linspace[j],µ_nom,sig_nom))**2)/(ND(x_linspace[j],µ_nom,sig_nom))
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*((ND(x_linspace[j+1],µ,sig)-ND(x_linspace[j+1],µ_nom,sig_nom))**2)/(ND(x_linspace[j+1],µ_nom,sig_nom))
    
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve
res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_CSD=res['x'][0]
sig_nom_CSD=res['x'][1]
M_CSD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum chi_square distance is: µ=',µ_nom_CSD,' sig=',sig_nom_CSD,' with maximum chi_square distance=',M_CSD)

# %% Now we calculate the total variation distance
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
x_linspace = np.linspace(np.min([µ_list])-20*np.max([sig_list]),np.max([µ_list])+20*np.max([sig_list]),100)
# Define a normal distribution as a casadi function
x = ca.SX.sym('x')
ND = ca.Function('ND',[x,µ_nom,sig_nom],[ca.exp(-0.5*((x-µ_nom)/sig_nom)**2)/(sig_nom*ca.sqrt(2*ca.pi))])
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M=0
    print('Iteration: ',i)
    for j in range(len(x_linspace)-1):
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*(0.5*ca.fabs(ND(x_linspace[j],µ,sig)-ND(x_linspace[j],µ_nom,sig_nom)))
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*(0.5*ca.fabs(ND(x_linspace[j+1],µ,sig)-ND(x_linspace[j+1],µ_nom,sig_nom)))
    
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(1)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve
res= solver(x0=[0.01,1.6,0.5],lbg=lb_g,ubg=ub_g)
µ_nom_TVD=res['x'][0]
sig_nom_TVD=res['x'][1]
M_TVD=res['x'][2]

# %%
print('The nominal distribution with the smallest maximum total variation distance is: µ=',µ_nom_TVD,' sig=',sig_nom_TVD,' with maximum total variation distance=',M_TVD)
# %% Plot all the rescaled epsilons for the different metrics
eps_list = np.linspace(0,1,101)
eps_KL=np.zeros((len(eps_list),1))
eps_RVD=np.zeros((len(eps_list),1))
eps_HD=np.zeros((len(eps_list),1))
eps_CSD=np.zeros((len(eps_list),1))
eps_TVD=np.zeros((len(eps_list),1))
opt_func = lambda  lam,KL_div, eps_val : -(np.exp(-KL_div)*(lam+1)**eps_val-1)/(lam)
for i in range(len(eps_list)):
    eps=eps_list[i]

    _,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,M_KLD,eps_list[i]),1e-6,1e6,full_output=True)
    eps_KL[i]=-corr_eps_KL
    eps_RVD[i]=eps_list[i]/M_RVD
    eps_HD[i]=np.maximum(np.sqrt(eps_list[i])-M_HD,0)**2
    #if Jiang_CSD_metric:
    eps_CSD[i] = eps_list[i]- (np.sqrt(M_CSD**2+4*M_CSD*(eps_list[i]-eps_list[i]**2))-(1-2*eps_list[i])*M_CSD)/(2*M_CSD+2)
    #else:
    #    eps_CSD[i]=eps_grid[i]+ M_CSD/2-np.sqrt(eps_grid[i]*M_CSD+M_CSD**2/4)
    eps_TVD[i]=eps_list[i]-M_TVD
print('PRL for RVD for an epsilon of 0.01 is: ',eps_RVD[1])
print('PRL for KL for an epsilon of 0.01 is: ',eps_KL[1])
print('PRL for HD for an epsilon of 0.01 is: ',eps_HD[1])
print('PRL for CSD for an epsilon of 0.01 is: ',eps_CSD[1])
print('PRL for TVD for an epsilon of 0.01 is: ',eps_TVD[1])
# %%
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(eps_list,eps_KL,label='KL')
ax.plot(eps_list,eps_RVD,label='RVD')
ax.plot(eps_list,eps_HD,label='HD')
ax.plot(eps_list,eps_CSD,label='CSD')
ax.plot(eps_list,eps_TVD,label='TVD')
ax.set_ylabel('$\hat{\epsilon}$')
ax.set_xlabel('$\epsilon$')
ax.set_title('Rescaled $\epsilon$ for different metrics')
ax.set_ylim([1e-6,1])
ax.set_yscale('log')
ax.legend()
fig.tight_layout()


# %% Plot eps_hat/eps
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(eps_list,[eps_KL[i]/eps_list[i] for i in range(len(eps_list))],label='Kullback-Leibler distance')
ax.plot(eps_list,[eps_RVD[i]/eps_list[i] for i in range(len(eps_list))],label='Relative variation distance')
ax.plot(eps_list,[eps_HD[i]/eps_list[i] for i in range(len(eps_list))],label='Hellinger distance')
ax.plot(eps_list,[eps_CSD[i]/eps_list[i] for i in range(len(eps_list))],label='Chi square distance')
ax.plot(eps_list,[eps_TVD[i]/eps_list[i] for i in range(len(eps_list))],label='Total variation distance')
ax.set_ylabel('PRL over risk level $\hat{\epsilon}_M/\epsilon$')
ax.set_xlabel('Risk level $\epsilon$')
#ax.set_title('Rescaled $\epsilon$ for different metrics')
ax.set_ylim([1e-2,1])
ax.set_yscale('log')
ax.legend()
fig.tight_layout()


# %%
fig.savefig('normal_distr_eps_hat_normalized.pdf',bbox_inches='tight')

# %% Get some normal distributions, for which the RVD is smaller than 2
# Setup optimization problem finding the biggest sigam for given mu
µ = ca.SX.sym('µ')
µ1 = ca.SX.sym('µ1')
sigma = ca.SX.sym('sigma')
sigma_max = ca.SX.sym('sigma_max')
RVD_max = ca.SX.sym('RVD_max')
J = sigma
g = []
lb_g = []
ub_g = []

g.append(RVD_max-sigma_max/sigma*ca.exp(0.5*((µ-µ1)**2/(sigma_max**2-sigma**2))))
lb_g.append(0)
ub_g.append(0)
g.append(sigma_max-sigma)
lb_g.append(0)
ub_g.append(ca.inf)
g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(sigma),'f':J,'g':g,'p':ca.vertcat(µ,µ1,sigma_max,RVD_max)}
opts = {'ipopt': {'print_level': 5 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve the optimization problem
µ1_val = 0
sigma_max_val = 1
RVD_max_val =[2,4]

µ_list=np.arange(-2,2,0.1)
sig_list = np.zeros((len(µ_list),len(RVD_max_val)))
for j in range(len(RVD_max_val)):
    for i in range(len(µ_list)):
        res = solver(x0=0.1,lbx=1e-12,ubx=ca.inf,lbg=lb_g,ubg=ub_g,p=ca.vertcat(µ_list[i],µ1_val,sigma_max_val,RVD_max_val[j]))
        if solver.stats()['success']:
            sig_list[i,j] = res['x'][0]
        else:
            sig_list[i,j] = -1


# %% Plot the normal distributions
if len(RVD_max_val)>1:
    fig, ax = plt.subplots(1,len(RVD_max_val),sharex=True,sharey=True,figsize=(6,3))
    # Plot the base normal distribution in thick black
    x = np.linspace(-3,3,1000)
    for j in range(len(RVD_max_val)):
        ax[j].plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
        # Plot the normal distributions for the different µ values in grey thin lines
        for i in range(len(µ_list)):
            if sig_list[i,j]>0:
                y = stats.norm.pdf(x,µ_list[i],sig_list[i,j])
                if i == µ_list.size//2:
                    ax[j].plot(x,y,'grey',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
                else:
                    ax[j].plot(x,y,'grey',alpha=0.5)
        ax[j].set_xlabel('$\delta$')

    ax[0].set_ylabel('f($\delta$)')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
else:
    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(3,3))
    # Plot the base normal distribution in thick black
    x = np.linspace(-3,3,1000)
    ax.plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
    # Plot the normal distributions for the different µ values in grey thin lines
    for i in range(len(µ_list)):
        if sig_list[i,0]>0:
            y = stats.norm.pdf(x,µ_list[i],sig_list[i,0])
            if i == µ_list.size//2:
                ax.plot(x,y,'grey',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
            else:
                ax.plot(x,y,'grey',alpha=0.5)
    ax.set_xlabel('$\delta$')
    ax.set_ylabel('f($\delta$)')
    ax.set_ylim([0,2])
    ax.legend()
    fig.tight_layout()





# %%
fig.savefig('normal_distr_diff_mu_RVD_{}.pdf'.format(RVD_max_val[0]),bbox_inches='tight')

# %%
