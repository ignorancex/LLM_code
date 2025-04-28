import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

plt.figure(figsize = (10,5))

x_labels = [r'$\times 3$ Inflation', r'$\times 5$ Inflation', r'$\times 10$ Inflation']
dn = 0.1
x_number = [0-dn, 1-dn, 2-dn]
x_number2 = [0+dn,1.+dn, 2+dn]

y = [456., 508., 716.]
yerr = [[139., 193., 339.], [138., 191., 333.]]
yerr3sigma = [[367., 493., 864.], [354., 482., 855.]]

y2 = [818.5, 863., 1018.]
yerr2 = [[242., 305., 523.6], [242., 305., 512.5]]
yerr3sigma2 = [[647., 817., 1348.], [621., 782., 1337.]]

#yerr = [[139., 138.], [138., 139.], [193., 191.], [196., 195.], [339., 333.], [336., 335.]]
#yerr3sigma = [[367., 354], [372., 364.], [493., 482.], [502., 496], [864., 855.], [864., 864.]]

plt.plot([-0.5,2.5], [0, 0], '--', color = 'tomato')
plt.errorbar(x_number, y, yerr, fmt = 'o', ms = 15, mew = 3, elinewidth = 3, mfc = 'white', mec = 'black', ecolor = 'black', label = 'All wavelengths')
plt.errorbar(x_number, y, yerr3sigma, fmt = 'o', ms = 15, elinewidth = 1, mfc = 'white', mec = 'black', ecolor = 'black')

plt.errorbar(x_number2, y2, yerr2, fmt = 'o', ms = 15, mew = 3, elinewidth = 3, mfc = 'white', mec = 'tomato', ecolor = 'tomato', label = r'Wavelengths $> 4\mu m$')
plt.errorbar(x_number2, y2, yerr3sigma2, fmt = 'o', ms = 15, elinewidth = 1, mfc = 'white', mec = 'tomato', ecolor = 'tomato')

plt.title('Error inflation impact on morning/evening detection', fontsize = 18)
plt.ylabel('Average $\Delta $ Evening - Morning (ppm)', fontsize = 16)

plt.legend(fontsize = 14)#, frameon = False)

plt.yticks(fontsize = 14)
plt.xticks([0, 1, 2], x_labels, fontsize = 14, rotation=70)

plt.xlim(-0.5,2.5)
plt.ylim(-250,1500)

plt.tight_layout()
plt.savefig('circular.pdf')
#plt.show()
