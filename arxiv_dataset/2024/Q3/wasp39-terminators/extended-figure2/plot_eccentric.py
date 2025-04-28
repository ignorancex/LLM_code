import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

plt.figure(figsize = (10,5))

x_labels = [r'$\times 3$ Inflation', r'$\times 5$ Inflation', r'$\times 10$ Inflation']
dn = 0.1
x_number = [0-dn, 1-dn, 2-dn]
x_number2 = [0+dn,1.+dn, 2+dn]

y = [760.5, 237, -2295]
yerr = [[402., 580., 816.], [400., 581., 832.]]
yerr3sigma = [[1025., 1431, 2124], [1061., 1542, 2140]]

y2 = [1995., 2086., -1222]
yerr2 = [[696., 1019.5, 1439], [700., 1051., 1480]]
yerr3sigma2 = [[1783., 2542., 3706], [1845., 2638., 3914]]

#yerr = [[139., 138.], [138., 139.], [193., 191.], [196., 195.], [339., 333.], [336., 335.]]
#yerr3sigma = [[367., 354], [372., 364.], [493., 482.], [502., 496], [864., 855.], [864., 864.]]

plt.plot([-0.5,2.5], [0, 0], '--', color = 'tomato')
plt.errorbar(x_number, y, yerr, fmt = 'o', ms = 15, mew = 3, elinewidth = 3, mfc = 'white', mec = 'black', ecolor = 'black', label = 'All wavelengths')
plt.errorbar(x_number, y, yerr3sigma, fmt = 'o', ms = 15, elinewidth = 1, mfc = 'white', mec = 'black', ecolor = 'black')

plt.errorbar(x_number2, y2, yerr2, fmt = 'o', ms = 15, mew = 3, elinewidth = 3, mfc = 'white', mec = 'tomato', ecolor = 'tomato', label = r'Wavelengths $> 4\mu m$')
plt.errorbar(x_number2, y2, yerr3sigma2, fmt = 'o', ms = 15, elinewidth = 1, mfc = 'white', mec = 'tomato', ecolor = 'tomato')

plt.title('Error inflation impact on morning/evening detection (eccentric case)', fontsize = 18)
plt.ylabel('Average $\Delta $ Evening - Morning (ppm)', fontsize = 16)

plt.legend(fontsize = 14)#, frameon = False)

plt.yticks(fontsize = 14)
plt.xticks([0, 1, 2], x_labels, fontsize = 14, rotation=70)

plt.xlim(-0.5,2.5)
plt.ylim(-3000,3000)

plt.tight_layout()
plt.savefig('ecc.pdf')
#plt.show()
