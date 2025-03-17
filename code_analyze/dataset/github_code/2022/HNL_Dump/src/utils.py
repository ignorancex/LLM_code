import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
from .momentum4 import Momentum4
from .logger import Logger
from .experimental_constants import get_experimental_constants



plt.style.use('ja')

plt.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'])

def DEBUG_PLOT_MOMENTA(particle, range):
    samples = []
    for momentum in particle.momenta:
        pt = momentum.get_transverse_momentum()
        pp = momentum.get_parallel_momentum()
        samples.append([pp, pt])
    samples = np.array(samples)

    fig = plt.figure()
    plt.hist2d(samples[:,0], samples[:,1], bins=100, range=range)
    plt.show()

def PLOT_ENERGY_ANGLE(momenta, range, filename, detector_cut=False, horizontal_line=0, xLabel = r'$E$', yLabel = r'$\theta$'):
    if not Logger().DEBUG:
        return
    samples = []
    for momentum in momenta:
        th = np.arccos(momentum.get_cos_theta())
        e = momentum.get_energy()
        samples.append([e, th])
    samples = np.array(samples)
    fig = plt.figure()
    plt.xlabel(xLabel + r'$ \, [ \mathrm{GeV}]$')
    plt.ylabel(yLabel + r'$\, [\mathrm{rad}]$')
    plt.hist2d(samples[:,0], samples[:,1], bins=100, range=range)
    y_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:.2f}$'.format(x) for x in y_values])
    if detector_cut:
        plt.plot(samples[:,0], np.linspace(DETECTOR_OPENING_ANGLE, DETECTOR_OPENING_ANGLE, len(samples[:,0])), 'k')
    if horizontal_line != 0:
        plt.hlines(horizontal_line,range[0][0],range[0][1],ls = '--', color = 'white')
        plt.text(200,horizontal_line, r'$\mathrm{BEBC} \, \theta_\mathrm{max} $', ha = 'right',
                 va = 'bottom', color = 'white')
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dirname, f"../temp_graphs/{filename}.png"), dpi=250)
    
def PLOT_ENERGY_ANGLE_JACOBIAN(momenta, range, filename, detector_cut=False, horizontal_line=0, xLabel = r'$E$', yLabel = r'$\theta$'):
    if not Logger().DEBUG:
        return
    samples = []
    for momentum in momenta:
        th = np.arccos(momentum.get_cos_theta())
        e = momentum.get_energy()
        samples.append([e, 10**3*th**2])
    samples = np.array(samples)
    fig = plt.figure()
    plt.xlabel(xLabel + r'$ \, [ \mathrm{GeV}]$')
    plt.ylabel(yLabel + r'$\, [\mathrm{rad}^2]$')
    plt.hist2d(samples[:,0], samples[:,1], bins=100, range=range)
    y_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:.2f}$'.format(x) for x in y_values])
    if detector_cut:
        plt.plot(samples[:,0], np.linspace(DETECTOR_OPENING_ANGLE, DETECTOR_OPENING_ANGLE, len(samples[:,0])), 'k')
    if horizontal_line != 0:
        plt.hlines(10**3*horizontal_line**2,range[0][0],range[0][1],ls = '--', color = 'white')
        plt.text(200,10**3*horizontal_line**2, r'$\mathrm{BEBC} \, \theta_\mathrm{max}^2 $', ha = 'right',
                 va = 'bottom', color = 'white')
    dirname = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(dirname, f"../temp_graphs/{filename}.png"), dpi=250)

def allowed_e1_e2_three_body_decays(e1, e2, e_parent, m1, m2, m3):
    x = e_parent**2 + m1**2 + m2**2 - m3**2
    F = x**2 - 4*m1*m1*m2*m2 - 8*e_parent*(e2*e1**2 + e1*e2**2) \
        + 4*(e_parent**2 + m2**2)*e1**2 + 4*(e_parent**2 + m1**2)*e2**2 + 4*(3*e_parent**2 + m1**2 + m2**2 - m3**2)*e1*e2 \
        - 4*x*e_parent*e1 - 4*x*e_parent*e2
    return F < 0
    
def DEBUG_AVERAGE_MOMENTUM(particle, text):
    total_p = 0
    for momentum in particle.momenta:
        total_p += momentum.get_total_momentum()
    Logger().log(f"{text}: {total_p/len(particle.momenta)}")

def e_cos_theta_to_momentum4(samples, mass):
    momentum4_samples = []
    for sample in samples:
        e, cos_theta = sample
        momentum4_samples.append(Momentum4.from_polar(e, cos_theta, np.random.uniform(0, 2*np.pi), mass))
    return np.array(momentum4_samples)

def e_cos_theta_phi_to_momentum4(samples, mass):
    momentum4_samples = []
    for sample in samples:
        e, cos_theta, phi = sample
        momentum4_samples.append(Momentum4.from_polar(e, cos_theta, phi, mass))
    return np.array(momentum4_samples)
    
def get_two_body_momenta(parent, particle, other_particle, num_samples, in_plane=False):
    e0 = (parent.m**2 + particle.m**2 - other_particle.m**2) / (2*parent.m)
    samples = []
    for i in range(num_samples):
        cos = np.random.uniform(0.0, 1.0)
        phi = 0 if in_plane else np.random.uniform(0, 2*np.pi)
        samples.append([e0, cos, phi])
    sample_momenta = e_cos_theta_phi_to_momentum4(samples, particle.m)
    return sample_momenta

def sample_2d_dist(dist_func, x, y, n):
    X, Y = np.meshgrid(x, y)
    dist = dist_func(X, Y) 
    dist = dist / dist.sum()
    # Create a flat copy of the array
    flat = dist.flatten()
    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=n)
    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, dist.shape)
    x_samples = x[adjusted_index[1]]
    y_samples = y[adjusted_index[0]]
    return np.array(list(zip(x_samples, y_samples)))

def generate_samples(*x, dist_func, n_samples, region=lambda *x: True) -> np.ndarray:
    X = np.meshgrid(*x)
    dist = dist_func(*X)
    # If the points are outside the region then dist = 0 at that point
    dist = np.where(region(*X), dist, np.zeros(dist.shape))
    dist = dist / dist.sum()
    # Create a flat copy of the array
    flat = dist.flatten()
    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat, size=n_samples)
    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, dist.shape)
    samples = []
    # Pick out the coordinate values from the indices
    for i in range(len(x)):
        samples.append(x[i][adjusted_index[len(x)-1-i]])
    # Zip them up in (x, y, ...) tuples
    return np.array(list(zip(*samples)))

def plot_surface(func, x, y, range):
    x_range, y_range = range
    x_min, x_max = x_range
    y_min, y_max = y_range
    x = x[np.where(x <= x_max)]
    x = x[np.where(x >= x_min)]
    y = y[np.where(y >= y_min)]
    y = y[np.where(y <= y_max)]
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, func(X,Y))
    plt.show()

def sliced_hist(samples_2d, var_to_fix, cut):
    lower, upper = cut
    sliced = samples_2d[np.where(samples_2d[:,var_to_fix] < upper)]
    sliced = sliced[np.where(sliced[:,var_to_fix] > lower)]
    var = 1 - var_to_fix
    return sliced[:, var]

def write_to_file_csv(samples, filepath):
    f = open(filepath, "w")
    for arr in samples:
        row = ""
        for i in range(len(arr)):
            row += f"{arr[i]}"
            row += "," if i != len(arr) - 1 else "\n"
        f.write(row)
    f.close()

def get_lepton_momenta_lab_frame(lepton_energies, hnl_momentum, parent, lepton1, lepton2):
    e1, e2 = lepton_energies
    e3 = parent.m - (e1 + e2)
    p1 = np.sqrt(e1**2 - lepton1.m**2)
    p2 = np.sqrt(e2**2 - lepton2.m**2)
    cos_th_12 = (lepton1.m**2 + lepton2.m**2 + 2*e1*e2 - parent.m**2 + 2*parent.m*e3)/(2*p1*p2)
    if abs(cos_th_12) > 1:
        Logger().log(f"invalid cos: {cos_th_12}")
        return None
    theta_12 = np.arccos(cos_th_12) # takes values between 0, pi
    theta_1 = np.random.uniform(0, 2*np.pi) # angle of one of the leptons uniformly between 0, 2pi. This is like the cone angle
    theta_2 = theta_1 + theta_12
    cos_theta_1 = np.cos(theta_1)
    cos_theta_2 = np.cos(theta_2)
    p1 = Momentum4.from_polar(e1, cos_theta_1, 0, lepton1.m).boost(-hnl_momentum)
    p2 = Momentum4.from_polar(e2, cos_theta_2, 0, lepton2.m).boost(-hnl_momentum)
    p_tot = p1 + p2
    return p1,p2,p_tot