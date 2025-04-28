import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd



def plot_regret(regretsdict, save = False, paramsdict = None, banditenv = None, mode = 1):
    """Plot the regret of the different algorithms and save the regrets as an option.

    args:
        regretsdict: dictionary of numpy arrays of shape (N, T) where N is the number of runs and T the number of time steps.
        label: list of strings of length len(regrets) containing the name of the algorithms.
        save: boolean, if True, save the regrets in a folder.
        params: dictionary of dictionaries containing the parameters of the algorithms.
    
        
    """
    plt.style.use('default')
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"],
            "font.size" : "30"
            }
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(16, 9))

    now = datetime.now()
    dt_string = now.strftime("%y_%m_%d_%H:%M:%S")


    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for key in regretsdict:

        regrets = regretsdict[key]
        N = regretsdict[key].shape[0]
        T = regretsdict[key].shape[1]
        cumregret = np.cumsum(regrets, axis = 1)
        sorted = np.argsort(cumregret[:,-1])
        std = np.std(cumregret, axis=0)

        worse = cumregret[sorted[int(0.95*N)],:]
        best = cumregret[sorted[int(0.05*N)],:]

        meancumregret = np.mean(cumregret, axis=0)
        # std = np.std(cumregret, axis=0)

        if mode == 1:
            plt.plot(np.arange(T),meancumregret, label=key)
            plt.fill_between(np.arange(T), meancumregret - std*1.96, meancumregret + std*1.96, alpha=0.2)
        elif mode == 2:
            for i in range(N):
                plt.plot(np.arange(T),cumregret[i,:], label=key+"_"+str(i), color = colors[i])
        elif mode == 3:
            plt.plot(np.arange(T),meancumregret, label=key)
            plt.fill_between(np.arange(T), best, worse, alpha=0.2)

        
    
    plt.ylim(bottom=-1)
    plt.xlim(left = -1)
    plt.xlabel("t", fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylabel("Cumulative Regret", fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc='upper left', fontsize=30)

    
    if not os.path.exists(f"experiments"):
        os.makedirs(f"experiments")

    if not os.path.exists(f"experiments/{banditenv.name}"):
        os.makedirs(f"experiments/{banditenv.name}")

    root = f"experiments/{banditenv.name}/d{banditenv.d}_Deltamin{banditenv.Deltamin:.2f}"
    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(f"{root}/mode{mode}_regret{dt_string}.png", dpi=300)

    if save:  
        for algo in regretsdict:
            if not os.path.exists(f"{root}/regret_{algo}"):
                os.makedirs(f"{root}/regret_{algo}")
            if not os.path.exists(f"{root}/regret_{algo}/T{T}"):
                os.makedirs(f"{root}/regret_{algo}/T{T}")
            if not os.path.exists(f"{root}/regret_{algo}/T{T}/regrets"):
                df = pd.DataFrame(regretsdict[algo], columns = [str(t) for t in range(T)])
                for key in paramsdict[algo]:
                    df[key] = [paramsdict[algo][key] for _ in range(N)]
                df.to_csv(f"{root}/regret_{algo}/T{T}/regrets.csv", header=True)
            else:
                df = pd.DataFrame(regretsdict[algo], columns = [str(t) for t in range(T)])
                for param in paramsdict[algo]:
                    df[key] = [paramsdict[algo][param] for _ in range(N)]
                df.to_csv(f"{root}/regret_{algo}/T{T}/regrets.csv", mode='a', header=False)

    