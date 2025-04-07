import os
import pandas as pd

from run_linear_model import experiment


def paper_results():
    names = []
    h5 = []
    h10 = []
    asim = []
    conf_dir = 'configs/linear/transe/300/'
    expers = os.listdir(conf_dir)
    for exp in expers:
        print(exp)
        cn, hits5, hits10, avgsim = experiment(os.path.join(conf_dir, exp))
        names.append(cn)
        h5.append(hits5)
        h10.append(hits10)
        asim.append(avgsim)
    out = pd.DataFrame({'names': names, 'h5': h5, 'h10': h10, 'avg_sim': asim})
    print(out)
    with open('linear_map.tex', 'w') as tf:
        tf.write(out.to_latex(index=False))


if __name__ == "__main__":
    paper_results()
