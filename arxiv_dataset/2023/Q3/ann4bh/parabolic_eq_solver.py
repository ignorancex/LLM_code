# Analysis of Black Hole Solutions in Parabolic Class Using Neural Networks
# The script returns:
# Three dfs of b0, fm, fa of sizes (Nt0 by s0) as .csv
# Two df of train and val losses of sizes (Nepochs by s0) as .csv
# The vector of Nt0 as .txt
# s0 figures of losses
# The final figure of interval estimates


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn
from neurodiffeq import diff  # the differentiation operation
from neurodiffeq.conditions import IVP  # the initial condition
from neurodiffeq.networks import FCNN  # fully-connect neural network
from neurodiffeq.networks import Swish
from neurodiffeq.solvers import Solver1D

# Global Arguments.
s0 = 100       # Number of simulations
Nepoch0 = 3000  # Number of epochs
Nt0 = 1000     # Number of generated values
Upper0 = 3    # Upper bound
Lower0 = 0.5     # Lower bound
w = 1.65
exp_path = 'results/experiment1/' #IMPORTANT: remember to create this folder or any other to save the results

# DataFrames for Estimates
b0 = [[0] * s0 for j in range(Nt0)]
fm = [[0] * s0 for j in range(Nt0)]
fa = [[0] * s0 for j in range(Nt0)]

b0_df = pd.DataFrame(b0)
fm_df = pd.DataFrame(fm)
fa_df = pd.DataFrame(fa)

# DataFrames for losses
train_ls = [[0] * s0 for j in range(Nepoch0)]
val_ls = [[0] * s0 for j in range(Nepoch0)]

df_train_ls = pd.DataFrame(train_ls)
df_val_ls = pd.DataFrame(val_ls)

# Chose a dimension for the equations of motion
# d=4,5,6,7,8,9
d=5

# black holes equations of motion (the ODE system) for the parabolic case
pbl_ode_system = lambda b0, u, v, t : [
         diff(b0, t) * 2*(d-2)*b0 *v**2-(w**2*t+(t**2-b0**2))* (-2*w*diff(u, t)+t*diff(u, t)**2+t*diff(v, t)**2),
        -diff(u, t, order=2)*(4*(d-2)*t*b0**2*(-t**2+b0**2)*v**2)+w**3*t**3-2*(d-2)*v**2*w*t*b0**2-3*w**2*t**4*diff(u, t)
        +4*(d-2)*v**2*t**2*b0**2*diff(u, t)+w**2*t**2*b0**2*diff(u, t)-2*(d-2)**2*v**2*b0**4*diff(u, t)+3*w*t**5*diff(u, t)**2
        -w*t**3*b0**2*diff(u, t)**2-2*w*t*b0**4*diff(u, t)**2-t**6*diff(u, t)**3+t**2*b0**4*diff(u, t)**3
        +4*(d-2)*v*w*t**2*b0**2*diff(v, t)-4*(d-2)*v*t**3*b0**2*diff(u, t)*diff(v, t)+4*(d-2)*v*t*b0**4*diff(u, t)*diff(v, t)
        +w*t**5*diff(v, t)**2-w*t**3*b0**2*diff(v, t)**2-t**6*diff(u, t)*diff(v, t)**2+t**2*b0**4*diff(u, t)*diff(v, t)**2,
        -diff(v, t, order=2) *(4*(d-2)*t*b0**2*(-t**2+b0**2)*v**2)+2*(d-2)*v*w**2*t*b0**2-4*(d-2)*v*w*t**2*b0**2*diff(u, t)
        +2*(d-2)*v*t**3*b0**2*diff(u, t)**2-2*(d-2)*v*t*b0**4*diff(u, t)**2-w**2*t**4*diff(v, t)+4*(d-2)*v**2*t**2*b0**2*diff(v, t)
        -w**2*t**2*b0**2*diff(v, t)-2*(d-2)**2*v**2*b0**4*diff(v, t)+2*w*t**5*diff(u, t)*diff(v, t)-2*w*t*b0**4*diff(u, t)*diff(v, t)
        -t**6*diff(u, t)**2*diff(v, t)+t**2*b0**4*diff(u, t)**2*diff(v, t)-2*(d-2)*v*t**3*b0**2*diff(v, t)**2
        +2*(d-2)*v*t*b0**4*diff(v, t)**2-t**6*diff(v, t)**3+t**2*b0**4*diff(v, t)**3
]


# The initial conditions
condition_ode_system = [
    IVP(t_0=0.0, u_0=1.0),                # 1.0 is the value of b0 at t_0 = 0.0
    IVP(t_0=0.0, u_0=0.0, u_0_prime=0),   # 0.0 is the value of u and u' at t_0 = 0.0
    IVP(t_0=0.0, u_0=0.321, u_0_prime=0)  # 0.0 is the value of fa and fa' at t_0 = 0.0
]

ts = np.linspace(Lower0, Upper0, Nt0)

for i in range(len(b0_df.columns)):
    # specify the network to be used to approximate each dependent variable
    # the input units and output units default to 1 for FCNN
    nets_pbl = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16)),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16)),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16)),
    ]
    # Instantiate a solver instance
    solver = Solver1D(ode_system=pbl_ode_system,
                      conditions=condition_ode_system,
                      t_min=Lower0,
                      t_max=Upper0,
                      nets=nets_pbl,
                      )
    solver.fit(max_epochs=Nepoch0)
    solution_pbl = solver.get_solution()
    # Plot the loss function
    plt.figure()
    plt.plot(solver.metrics_history['train_loss'], label='training loss')
    plt.plot(solver.metrics_history['valid_loss'], label='validation loss')
    plt.yscale('log')
    plt.title('loss during training')
    plt.legend()
    plt.savefig(exp_path + 'loss_w_' + str(w) + '_exp_' + str(i) + '.png', bbox_inches='tight')
    plt.show()
    df_train_ls[i] = pd.DataFrame(solver.metrics_history['train_loss'])
    df_val_ls[i] = pd.DataFrame(solver.metrics_history['valid_loss'])
    b0_df[i], fm_df[i], fa_df[i] = solution_pbl(ts, to_numpy=True)
    # Plot the solutions
    plt.figure()
    plt.plot(ts, b0_df[i], label='$b_0$', color='green', linewidth=1)
    plt.plot(ts, fm_df[i], label='$fm$', color='red', linewidth=1)
    plt.plot(ts, fa_df[i], label='$fa$', color='blue', linewidth=1)
    plt.title('solutions for the iteration')
    plt.legend()
    plt.show()
    plt.savefig(exp_path + 'solution_w_' + str(w) + '_exp_' + str(i) + '.png', bbox_inches='tight')

# Saves the dfs as .csv and .txt
np.savetxt(exp_path + 'ts_pbl4.txt', ts)
b0_df.to_csv(exp_path + 'b0_pbl4_df.csv', index=False)
fm_df.to_csv(exp_path + 'fm_pbl4_df.csv', index=False)
fa_df.to_csv(exp_path + 'fa_pbl4_df.csv', index=False)
df_val_ls.to_csv(exp_path + 'val_pbl4_ls.csv', index=True)
df_train_ls.to_csv(exp_path + 'train_pbl4_ls.csv', index=True)

b0_df_net = b0_df.transpose()
fm_df_net = fm_df.transpose()
fa_df_net = fa_df.transpose()

b0_res = b0_df_net.quantile([0.025, .50, 0.975], axis=0)
fm_res = fm_df_net.quantile([0.025, .50, 0.975], axis=0)
fa_res = fa_df_net.quantile([0.025, .50, 0.975], axis=0)


plt.figure()
plt.plot(ts, b0_res.iloc[0], label='Lower $b_0$', color='green', linewidth=1, linestyle='dashed')
plt.plot(ts, b0_res.iloc[1], label='Median $b_0$', color='green', linewidth=1)
plt.plot(ts, b0_res.iloc[2], label='Upper $b_0$', color='green', linewidth=1, linestyle='dashed')
plt.plot(ts, fm_res.iloc[0], label="Lower $fm$", color='blue', linewidth=1, linestyle='dashed')
plt.plot(ts, fm_res.iloc[1], label="Median $fm$", color='blue', linewidth=1)
plt.plot(ts, fm_res.iloc[2], label="Upper $fm$", color='blue', linewidth=1, linestyle='dashed')
plt.plot(ts, fa_res.iloc[0], label='Lower $fa$', color='red', linewidth=1, linestyle='dashed')
plt.plot(ts, fa_res.iloc[1], label='Median $fa$', color='red', linewidth=1)
plt.plot(ts, fa_res.iloc[2], label='Upper $fa$', color='red', linewidth=1, linestyle='dashed')
plt.ylabel('Critical Solutions')
plt.xlabel('z')
plt.title('The ANN-based Estimates of Parabolic Functions')
plt.legend()
plt.savefig(exp_path + 'statistic_analysis_parabolic.pdf', bbox_inches='tight')
plt.show()