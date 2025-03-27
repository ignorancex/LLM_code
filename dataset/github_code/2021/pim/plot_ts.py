import numpy as np
import matplotlib.pyplot as plt

ys = np.load('data/time_series/y_true_pred_lstm_mst.npy')
yprev = np.load('data/time_series/y_prev_lstm_mst.npy')
piw = np.load('data/time_series/piw_lstm_mst.npy')
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

t_prev = np.arange(-10,0)+1
t_next = np.arange(30)+1
piw_up = ys[1][-1] + piw
piw_dn = ys[1][-1] - piw
std_resid_m = 1.96 * np.std(ys[1][-1]-ys[0][-1]) * np.sqrt(1+1/10) # piw/2 of mean forecasts
fig, ax = plt.subplots(1)
ax.plot(np.append(t_prev, t_next), np.append(yprev[-1,:], ys[0][-1]), color=cmap[0])
ax.plot(t_next, ys[1][-1], color=cmap[1])
ax.plot(t_next, ys[1][-1] + std_resid_m, 'k')
ax.plot(t_next, ys[1][-1] - std_resid_m, 'k')
ax.plot(t_next, piw_up, color='gray', lw=0.1)
ax.plot(t_next, piw_dn, color='gray', lw=0.1)
ax.fill_between(t_next, piw_dn, piw_up, color='gray', alpha=0.2)
ax.tick_params(axis='both', labelsize=13)
ax.legend(['Observed', r'Predicted'], loc='best', prop={'size':13})
ax.set_xlabel('Days', labelpad=5, fontsize=14)
ax.set_ylabel('Scaled price', labelpad=10, fontsize=14)
plt.tight_layout()
plt.savefig('paper/lstm_pim.pdf')
plt.show()