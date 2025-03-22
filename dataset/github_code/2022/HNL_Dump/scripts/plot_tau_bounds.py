# --- Needed to tell python about the package in ../src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))
# ---
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from src.mixing_type import MixingType
from plotting_utils import get_cols, read_xy_data

plt.style.use('ja')

plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])
def main():
    mixing_type = MixingType.tau
    upper_bounds = read_xy_data(f"upper_bound_data/upper_bounds_{mixing_type}.csv")
    upper_bounds = [upper_bounds[i] for i in range(len(upper_bounds)) if i % 1 == 0]
    lower_bounds = read_xy_data(f"lower_bound_data/lower_bounds_{mixing_type}_non_linear.csv")
    charm = np.array(read_xy_data("digitised_data/tau/HNLtauCHARM.dat"))
    t2k = np.array(read_xy_data("digitised_data/tau/T2K.csv"))
    delphi = np.array(read_xy_data("digitised_data/tau/DELPHI.csv"))
    mathusla = np.array(read_xy_data("digitised_data/tau/MATHUSLA.csv"))
    na62 = np.array(read_xy_data("digitised_data/tau/NA62.csv"))
    ship = np.array(read_xy_data("digitised_data/tau/SHiP.csv"))
    argo = np.array(read_xy_data("./digitised_data/tau/Argoneut.csv"))
    belle = np.array(read_xy_data("./digitised_data/tau/Belle.csv"))
    
    lower_bounds = sorted(list(zip(*lower_bounds)), key=lambda data: data[0], reverse=True)
    upper_bounds = sorted(list(zip(*upper_bounds)), key=lambda data: data[0])
    bounds = get_cols(np.append(upper_bounds, lower_bounds, axis=0))

    # # ------------------------ Plotting --------------------------------
    dpi = 250
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    plt.xlabel(r'$m_N\, [ \mathrm{GeV}]$')
    plt.ylabel(r'$|U_{\tau N}|^2$')
    plt.plot(*bounds, 'k', label="BEBC (reanalysis)")
    plt.plot(*charm, label="CHARM (recast)")
    plt.plot(*t2k, label="T2K")
    plt.plot(*argo, label="ArgoNeuT")
    plt.plot(*belle, label="Belle (recast)")
    plt.plot(*delphi, label="DELPHI")    
    plt.yscale('log')
    plt.legend(fontsize="small")
    plt.xlim([0.1,2.75])
    plt.ylim(1e-7,1e-2)
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.fill(*bounds, color="#ddd")
    # plt.fill(*get_cols(delphi), color="#ddd")
    plt.fill_between(*delphi, 1, color="#ddd")
    # plt.fill(*get_cols(charm), color="#ddd")
    plt.fill_between(*charm, 1, color="#ddd")
    plt.fill_between(*t2k, 1, color="#ddd")
    plt.savefig(os.path.join(dirname, f"../temp_graphs/mass_bound_plot{mixing_type}.png"), dpi=dpi)
    plt.show()

if __name__ == "__main__":
    main()