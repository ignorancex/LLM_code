# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:09:08 2024

@author: David J. Kedziora
"""

import calc

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

dir_data = "../data/ml"
dir_output = "../results/ml"

substring = "sps_cumsum"

experimental_contexts = [None,
                         "1p2uW_3000cps",
                         "30uW_7000cps",
                         None]

unit_delay = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

# Set up a zoom for histogram plots.
xlim_closeup = [np.mean(knowns["delay_mpe"]) - knowns["period_pulse"]*5/2,
                np.mean(knowns["delay_mpe"]) + knowns["period_pulse"]*5/2]

sr_centres = pd.Series(range(128, (500096+256), 256))*1e-12

fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex = True)#, sharey = True)
# len_max = 0
idx_context = 0
for experimental_context in experimental_contexts:
    if not experimental_context is None:
        df_data = pd.read_csv("%s/%s_%s.csv" % (dir_data, substring, experimental_context))

        idx_plot = 0
        for idx_row in [0, 1000, 10000, len(df_data["events"])-1]:

            if idx_row not in [0, len(df_data["events"])-1]:
                idx_row = (df_data["events"] - idx_row).abs().idxmin()

            time = (idx_row + 1)*10

            sr_hist = df_data.iloc[idx_row, :1954]
            num_events = df_data["events"].iloc[idx_row]

            g = df_data["g_fit"].iloc[idx_row]
            rate_bg = df_data["rate_bg"].iloc[idx_row]
            rate_env = df_data["rate_env"].iloc[idx_row]
            decay_peak = df_data["decay_peak"].iloc[idx_row]
            delay_mpe = df_data["delay_mpe"].iloc[idx_row]

            params = {"rate_bg": rate_bg,
                      "period_pulse": knowns["period_pulse"],
                      "delay_mpe": delay_mpe,
                      "rate_env": rate_env,
                      "g2_zero": g,
                      "factor_env": 0,
                      "decay_peak": decay_peak}

            sr_fit = calc.func_pulsed(params, sr_centres,
                                      duration = time)
            
            print(delay_mpe)

            ax = axes[idx_plot, idx_context]
            ax.plot(sr_centres/unit_delay, sr_hist, 
                    label = "Time: %i s, Events: %i" % (time, num_events))
            # ax.plot(sr_centres/unit_delay, sr_fit, 
            #         label = r"$g$: %.2f, $R_b$: %.2e, $R_p$: %.2e, $\gamma_p^{-1}$: %.2e, $\tau_0$: %.2e" 
            #         % (g, rate_bg, rate_env, decay_peak, delay_mpe))
            ax.plot(sr_centres/unit_delay, sr_fit, 
                    label = r"$g$: %.2f, $R_b$: %.2e, $R_p$: %.2e, $\gamma_p^{-1}$: %.2e" 
                    % (g, rate_bg, rate_env, decay_peak))

            _, ylim_max = ax.get_ylim()
            ax.set_ylim([0, ylim_max])
            if idx_row == 0:
                ax.set_title(experimental_context)

            ax.legend(loc = "upper left")

            idx_plot += 1

        idx_context += 1
    
ax.set_xlim(np.array(xlim_closeup)/unit_delay)

fig.supxlabel("Raw Delay (ns)")
bin_size = (sr_centres[1]-sr_centres[0])/unit_delay
fig.supylabel("Detected Events (Bin Size: %s ns)" % (bin_size))

plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()

# Show the plot.
plt.show()

fig.savefig(dir_output + "/hist.png", bbox_inches="tight")