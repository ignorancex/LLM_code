import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as mtick

from eos import cs2_qcd_fct
from style import font_choice, linewidth_choice

hbarc=0.1973

def cs2_fct(T_in_fm):
    return cs2_qcd_fct(T_in_fm)


plt.rc('font', **font_choice)


plt.figure()
plt.yticks([])
plt.axes().yaxis.set_minor_formatter(NullFormatter())
plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
plt.gca().yaxis.set_ticks_position('both')

plt.xlim(0.1,.6)
plt.ylim(1./8.,.5)
plt.yticks([1/6.,1/4.,1/3.,1/2.],('1/6','1/4','1/3','1/2'))

plt.xlabel(r"$T$ (GeV)")
plt.ylabel(r'$c_s^{2}$')

T_range=np.arange(0.1,.6,.001)
cs2_list= np.array(list(map(cs2_fct,T_range/hbarc)))


plt.plot(T_range, cs2_list,"-",color='red',label='HRG (SMASH) + HotQCD', linewidth=linewidth_choice)
plt.axhline(y=1./3., color='black', linestyle='--', linewidth=1)

plt.legend()
plt.tight_layout()
plt.savefig("cs2_hrg_hotqcd.pdf")
plt.show()
