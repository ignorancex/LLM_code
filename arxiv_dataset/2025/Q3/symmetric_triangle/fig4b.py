import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import rc

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18

# Create a figure and axis
fig = plt.figure(figsize=(5, 5))
ax = fig.subplots()

# Generate GHZ boundary
b_threshold = Polynomial([3, -8, 0, 0, 4]).roots()[2].real
b = np.linspace(1, b_threshold)
p1 = -1/2*b**3 + b**2 - 4/3*b + (1/8)*np.sqrt(4 - 3*b**2)*(4*b**2 - 16/3*b) + 1
q1 = -1/2*b**3 + (1/2)*b**2*np.sqrt(4 - 3*b**2) + b**2
b = np.linspace(b_threshold, 1)
p2 = -1/2*b**3 + (1/2)*b**2*np.sqrt(4 - 3*b**2) + b**2
q2 = -1/2*b**3 + b**2 - 4/3*b + (1/8)*np.sqrt(4 - 3*b**2)*(4*b**2 - 16/3*b) + 1
p = np.hstack((p1, p2))
q = np.hstack((q1, q2))

# Generate what I think is the GHZ boundary in fig. 5 of the NSI paper (322 models)
c = np.linspace(1, 0)
p_local_NSI = c**2
q_local_NSI = (1-c)**2

# NSI bound
pNSI1 = np.linspace(1, (3*np.sqrt(2) - 2)/8)
qNSI1 = pNSI1 + (1/54)*(-432*pNSI1 - 4)/(4374*pNSI1**2 + 162*pNSI1 + 162*np.sqrt(3)*np.sqrt(243*pNSI1**4 + 2*pNSI1**3) + 1)**(1/3) - 2/27*(4374*pNSI1**2 + 162*pNSI1 + 162*np.sqrt(3)*np.sqrt(243*pNSI1**4 + 2*pNSI1**3) + 1)**(1/3) + 25/27
qNSI2 = np.linspace((3*np.sqrt(2) - 2)/8, 1)
pNSI2 = qNSI2 + (1/54)*(-432*qNSI2 - 4)/(4374*qNSI2**2 + 162*qNSI2 + 162*np.sqrt(3)*np.sqrt(243*qNSI2**4 + 2*qNSI2**3) + 1)**(1/3) - 2/27*(4374*qNSI2**2 + 162*qNSI2 + 162*np.sqrt(3)*np.sqrt(243*qNSI2**4 + 2*qNSI2**3) + 1)**(1/3) + 25/27
pNSI = np.hstack((pNSI1, pNSI2))
qNSI = np.hstack((qNSI1, qNSI2))

data = np.loadtxt('pq_class_boundary_322.txt')
data = np.vstack(([[0,1]], data))

# Overlay your plot with transparency
plt.fill_between([0, 0.], [0, 0.], [0, 0.], color='white', label=r'$\bigtriangleup$-local [27]')
ax.fill_between(p_local_NSI, q_local_NSI, [1]*len(p_local_NSI), color='#9048E1', label=r'$\bigtriangleup$-local')
ax.fill_between(p, q, [1]*len(p), color='#FFDD69', label='Unknown')
plt.plot(data[:, 0], data[:, 1], 'k--', linewidth=1, label=r'Non-$\bigtriangleup$-local')
ax.fill_between(pNSI, qNSI, [1]*len(pNSI), color='#6ACCD0', label=r'Non-$\bigtriangleup$')
ax.fill_between(np.arange(0, 1.01, 0.01), 1-np.arange(0, 1.01, 0.01), [1]*101, color='#BFBFBF', label='Non-dist.')

# ax.fill((1, 0, 1), (0, 1, 1), color='#BFBFBF')
# ax.fill(pNSI, qNSI, color='#6ACCD0')
# ax.fill(np.hstack((pNSI, p)), np.hstack((qNSI, q)), color='#FFDD69')
# ax.fill(np.hstack((p, p_local_NSI)), np.hstack((q, q_local_NSI)), color='#9048E1')
# # Customize and add labels
ax.set_aspect('equal')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r"$\lambda_+$")
ax.set_ylabel(r"$\lambda_-$")
# ax.grid()


hand, lab = ax.get_legend_handles_labels()
hand[0].set_edgecolor('black')
hand[0].set_linewidth(0.2)
ax.legend(hand, lab, loc='upper right')
plt.legend()
# Show the result
# plt.show()

# Save figure
fig.savefig('NSI_fig_5_comparison.pdf', bbox_inches='tight')

