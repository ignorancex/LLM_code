from main_circuit import setpartition
import numpy as np
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.quantum_info import state_fidelity, Choi
from qiskit import *
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy
import scipy.linalg as la

# print( (3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2))) )
# exit()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)
    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))

def distinguishing_probability(hypothesis, gate, theta_oracle, theta_x):
    file_hypo_1 = f'data/dict_prob_initial_hypo_identity_oracle_ang_0.0_theta_x_0.0_initial_initialization_{gate}.p'
    file_hypo_2 = f'data/dict_prob_initial_hypo_{hypothesis}_oracle_ang_{theta_oracle}_theta_x_{theta_x}_initial_initialization_{gate}.p'
    dict_hypo_1 = pickle.load(open(file_hypo_1, "rb"))
    dict_hypo_2 = pickle.load(open(file_hypo_2, "rb"))
    list_key_hypo_1 = []
    list_key_hypo_2 = []
    for k in dict_hypo_1.keys():
        list_key_hypo_1.append(k)
    for k in dict_hypo_2.keys():
        list_key_hypo_2.append(k)
    common_bits = []
    for i in list_key_hypo_1:
        for j in list_key_hypo_2:
            if i == j:
                common_bits.append(i)
    x = 0
    for k in list_key_hypo_2:
        if k not in common_bits:
            x += dict_hypo_2[k]
    return x

def operations_vs_linearly_independent_state(ax1):
    subsystem_sim_list = []
    partition_list = []
    for aritra_dar_dimension in range(2, 12):
        partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
        subsystem_sim_list.append(len(partition))
        partition_list.append(len(partition)*len(partition[0]))
    ax1.plot( subsystem_sim_list, partition_list, '-x' )
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.semilogy( [3], [6], 'ro', alpha=.3, ms=9, lw=3)
    ax1.set_ylabel('Number of operations', fontsize = 14)
    ax1.set_xlabel('$r$', fontsize = 14)

def operations_vs_subsystem_dim(ax2):
    aritra_dar_dimension = 4
    subsystem_sim_list = []
    partition_list = []
    for aritra_dar_dimension in range(2, 12):
        partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
        subsystem_sim_list.append(aritra_dar_dimension)
        partition_list.append(len(partition)*len(partition[0]))
    ax2.plot( subsystem_sim_list, partition_list, '-x' )
    ax2.set_yscale('log')
    ax2.semilogy( [4], [6], 'ro', alpha=.3, ms=9, lw=3, label = 'Simulation presented in article')
    ax2.set_xlabel('$N_{A}$',fontsize = 14)
    ax2.legend()


def oracle_type(theta, type):
    qc = QuantumCircuit(3)
    if type == 'rx-c-swap':
        qc.rx(theta, [2])
        qc.cswap([2], [1], [0])
    elif type == 'id':
        qc.id([0])
        qc.id([1])
        qc.id([2])
    return qc
def oracle_type_choi(theta, type):
    qc = QuantumCircuit(3)
    if type == 'rx-c-swap':
        qc.rx(theta, [2])
        qc.cswap([2], [1], [0])
        qc.rx(-theta, [2])
    if type == 'iswap':
        qc.rxx(theta, 0, 1)
        qc.ryy(theta, 0, 1)
        qc.id([2])
    elif type == 'id':
        qc.id([0])
        qc.id([1])
        qc.id([2])
    return qc

def DeltaT(dm_i, dm_j):
    dm_i = np.divide( dm_i.data,  np.trace(dm_i.data))
    dm_j = np.divide( dm_j.data,  np.trace(dm_j.data))
    mat = np.asmatrix(dm_i - dm_j)
    dist = np.real(0.5* np.trace( la.sqrtm(mat.getH()@ mat )))
    return dist

def DeltaB(dm_i, dm_j):
    dm_i = np.divide( dm_i.data,  np.trace(dm_i.data))
    dm_j = np.divide( dm_j.data,  np.trace(dm_j.data))
    fid = np.trace( la.sqrtm( (la.sqrtm(dm_i) @ dm_j) @ dm_i) ).real**2
    dist = np.sqrt(2*(1-np.sqrt(fid)))
    return dist

def DeltaHS(dm_i, dm_j):
    dm_i = np.divide( dm_i.data,  np.trace(dm_i.data))
    dm_j = np.divide( dm_j.data,  np.trace(dm_j.data))
    mat = np.asmatrix(dm_i - dm_j)
    dist = np.trace( mat.getH()@ mat ).real/2
    return dist

def oracle_distance():
    fig, ax1 = plt.subplots(1, 1, figsize=(4.7,4) )
    qc_id = oracle_type_choi(0, 'id')
    # result = execute(qc_id, Aer.get_backend('statevector_simulator')).result()
    # mxd_state = np.asmatrix(result.get_statevector( qc_id ))
    # mxd_choi_mat = mxd_state.getH() @ mxd_state
    mxd_choi_mat = Choi(qc_id)
    theta_range = np.arange(0, 4*np.pi, np.pi/10)
    d1,d2,d3 = [],[],[]
    # prob = [] #HACK
    for t in theta_range:
        qc_cry = oracle_type_choi(t, 'iswap')
        # result = execute(qc_cry, Aer.get_backend('statevector_simulator')).result()
        # cry_state = np.asmatrix(result.get_statevector( qc_cry ))
        # cry_choi_op = cry_state.getH() @ cry_state
        cry_choi_op = Choi(qc_cry)
        # print(cry_choi_op)
        dist1 = DeltaT(mxd_choi_mat,cry_choi_op)
        dist2 = DeltaB(mxd_choi_mat,cry_choi_op)
        dist3 = DeltaHS(mxd_choi_mat,cry_choi_op)
        d1.append(dist1)
        d2.append(dist2)
        d3.append(dist3)
        # prob.append(dist1*0.60205999132) #HACK
    
    ax1.plot(theta_range, d1, 'r--v', markerfacecolor='none', label = "Trace distance")
    ax1.plot(theta_range, d2, 'k-', label = "Bures distance")
    ax1.plot(theta_range, d3, 'g--o', markerfacecolor='none', label = "Hilbert-Schmidt distance")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    # ax2.plot(prob) #HACK
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_ylabel("$ \\Delta [U_{\\small\\textrm{orc}}^{H_0}(0,0), U_{\\small\\textrm{orc}}^{H_1}(\\pi,\\theta_y)] $")
    ax1.set_xlabel("$\\theta_y$")
    ax1.legend(loc = 'center', ncol = 2, bbox_to_anchor=(0.5, 1.12), fontsize = 10)
    fig.tight_layout()
    plt.show()
    exit()
    plt.savefig('plot/diff_process_dist.pdf')
    plt.savefig('plot/diff_process_dist.png')

def theoretical_case_error_prob():
    fig, ax1 = plt.subplots(1, 1, figsize=(4.8,4) )
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]
    qc_id = oracle_type(0, 'id')
    result = execute(qc_id, Aer.get_backend('statevector_simulator')).result()
    mxd_state = np.asmatrix(result.get_statevector( qc_id ))
    mxd_choi_mat = mxd_state.getH() @ mxd_state
    dist = []
    if hypothesis == "swap-ry":
        theta_x_list = np.arange(0, 4*np.pi, np.pi/20)

    
    list_data = []
    list_angle = []
    for theta_x in theta_x_list:
        for theta_oracle in [0.0]:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = (3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2)))
            qc_cry = oracle_type(theta_x, 'rx-c-swap')
            result = execute(qc_cry, Aer.get_backend('statevector_simulator')).result()
            cry_state = np.asmatrix(result.get_statevector( qc_cry ))
            cry_choi_op = cry_state.getH() @ cry_state
            delta = DeltaT(mxd_choi_mat,cry_choi_op)
            dist.append(delta)
            list_data.append(1- delta*(1-prob))
            list_angle.append(theta_x)
    print(dist)
    # exit()
    ax1.plot(list_angle, list_data, 'r-o', label = "$\\theta_x = \\pi$, variation with $\\theta_y$", markerfacecolor='none')
    ax1.hlines((3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2))), xmin=[0.0], xmax=[4*np.pi])
    # ax1.plot([(3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2)))]*len(list_data), list_data, 'k-', label = "prac err prob [Chiribella]", markerfacecolor='none')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    
    ax1.set_ylabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{theo}}$')
    plt.tight_layout()
    plt.show()
    exit()
    plt.savefig('plot/theor_prob_vs_thetas.pdf')
    plt.savefig('plot/theor_prob_vs_thetas.png')

def practical_case_error_prob():
    fig, ax1 = plt.subplots(1, 1, figsize=(4.8,4) )
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]

    if hypothesis == "identity":
        theta_oracle_list = [0.0]
        theta_x_list = [0.0]
    elif hypothesis == "swap-ry":
        theta_oracle_list = np.arange(0, np.pi, np.pi/20)
        theta_x_list = np.arange(0, np.pi, np.pi/20)
    
    list_data = []
    list_angle = []
    for theta_oracle in theta_oracle_list:
        for theta_x in [np.pi]:#theta_x_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            list_angle.append(theta_oracle)
            list_data.append(prob)
    ax1.plot(list_angle, list_data, 'r-o', label = "$\\theta_x = \\pi$, variation with $\\theta_y$", markerfacecolor='none')
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    # ax1.set_xlabel('$\\theta$')
    ax1.set_ylabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{prac}}$')

    list_data = []
    list_angle = []
    for theta_x in theta_x_list:
        for theta_oracle in [0.0]:#theta_oracle_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            list_angle.append(theta_x)
            list_data.append(prob)
    ax1.plot(list_angle, list_data, 'b-x', label = "$\\theta_y = 0.0$, variation with $\\theta_x$")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_xlabel('$\\theta$')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()
    exit()
    plt.savefig('plot/prac_prob_vs_thetas.pdf')
    plt.savefig('plot/prac_prob_vs_thetas.png')

def prac_prob_err_3d():
    
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]
    theta_oracle_list, theta_x_list = np.arange(0, 2*np.pi, np.pi/20), np.arange(0, 2*np.pi, np.pi/20)
    sx, sy = theta_oracle_list.size, theta_x_list.size
    theta_oracle_list_plot, theta_x_list_plot = numpy.tile(theta_oracle_list, (sy, 1)), numpy.tile(theta_x_list, (sx, 1)).T
    prac_prob_plot = np.zeros((len(theta_oracle_list), len(theta_x_list)))

    for no, theta_oracle in enumerate(theta_oracle_list):
        U = []
        for theta_x in theta_x_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            U.append(prob)
        prac_prob_plot[no] = U
    
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4.6),subplot_kw=dict(projection='3d'))
    norm = plt.Normalize( theta_oracle_list.min(), theta_oracle_list.max() )
    colors = cm.viridis( norm(theta_oracle_list_plot) )
    rcount, ccount, _ = colors.shape
    
    # theta_x
    surf = ax1.plot_surface(theta_oracle_list_plot, theta_x_list_plot, prac_prob_plot, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi)) # theta_y np.pi/2
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2)) # theta_x np.pi/2
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_xlabel('$\\theta_y$', labelpad=10)
    ax1.set_ylabel('$\\theta_x$', labelpad=10)
    ax1.set_zlabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{prac}}$', labelpad=10)
    ax1.view_init(elev=None, azim=15) # theta_x
    
    # theta_y
    surf = ax2.plot_surface(theta_oracle_list_plot, theta_x_list_plot, prac_prob_plot, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2)) # theta_y np.pi/2
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi)) # theta_x np.pi/2
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax2.set_xlabel('$\\theta_y$', labelpad=10)
    ax2.set_ylabel('$\\theta_x$', labelpad=10)
    ax2.set_zlabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{prac}}$', labelpad=10)
    ax2.view_init(40, azim=75)
    plt.savefig('plot/3d_prac_err_prob.png')
    plt.savefig('plot/3d_prac_err_prob.pdf')


if __name__ == "__main__":
    # practical_case_error_prob()
    # prac_prob_err_3d()


    # exit()
    oracle_distance()
    theoretical_case_error_prob()