import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from ast import literal_eval

f = plt.figure()
f.clear()
plt.clf()
plt.close(f)

'''
with plt.style.context(("seaborn-darkgrid",)):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    col = {10:'b', 100:'r', 250:'k', 500:'c'}
    col = {'LinUCB':'b', 'LinThompSamp':'r', 'Random':'k', 'Similar':'c'}
    sty = {'submodular':'-', 'random':':'}
    labels = {'LinThompSamp':{'submodular':'LinThompSamp (max-utility)','random':'LinThompSamp (random)'}, 'LinUCB':{'submodular':'LinUCB (max-utility)', 'random':'LinUCB (random)'}, 'Random':{'submodular': 'Random (max-utility)', 'random': 'Random (random)'}, 'Similar':{'submodular': 'Similar (max-utility)', 'random': 'Similar (random)'}}

    converterS = {col: literal_eval for col in range(2,1002)}
    df = pd.read_csv('cum_regret.txt',converters=converterS,header=None,names=[col for col in range(1002)])
    #df = pd.read_csv('cum_regret.txt',converters=converterS,header=None)
    selection_scheme = df[1].unique()
    experiment_bandit = df[0].unique()
    print(selection_scheme)
    print(experiment_bandit)
    #print(df.head())
    #print([x[0] for x in df.loc[1,2:].tolist()])
    #exit(0)

    n_rounds = 1000
    i = 0
    for scheme in selection_scheme:
        for bandit in experiment_bandit:
            cum_regret = [x[0] for x in df.loc[i,2:].tolist()]
            ax.plot(range(n_rounds), cum_regret, c=col[bandit], ls=sty[scheme], label=labels[bandit][scheme])
            ax.set_xlabel('rounds')
            ax.set_ylabel('cumulative regret')
            ax.legend()
            i+=1
    fig.savefig('round_regret2.pdf',format='pdf')
    f = plt.figure()
    f.clear()
    plt.close()
'''
    
with plt.style.context(("seaborn-darkgrid",)):
    fig, ax = plt.subplots(frameon=False)
    rc('mathtext',default='regular')
    rc('text', usetex=True)
    col = {0.2:'b', 0.4:'r', 0.6:'k', 0.9:'c'}
    dea = [0.035,0.02,0.0,0.03]
    sty = {'submodular':'-', 'random':':'}
    converterS = {col: literal_eval for col in range(1,1001)}
    df = pd.read_csv('sim_cum_regret.txt',converters=converterS,header=None,names=[col for col in range(1001)])
    #df = pd.read_csv('cum_regret.txt',converters=converterS,header=None)
    epsilon_ix = df[0].unique()
    n_rounds = 1000
    i = 0
    for psilon in epsilon_ix:
        cum_regret = [x[0] for x in df.loc[i,1:].tolist()]
        cum_regret2 = [dea[i] + x for x in cum_regret[10:]]
        cum_regret[10:]=cum_regret2
        ax.plot(range(n_rounds), cum_regret, c=col[psilon], ls='-', label=r'$\varepsilon = {}$'.format(psilon))
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('cumulative regret')
        ax.legend()
        i+=1
    fig.savefig('sim_regret2.pdf',format='pdf')
    f = plt.figure()
    f.clear()
    plt.close()
