import numpy as np
from astropy.io import ascii
from astropy.table import Table
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


# leclavier des etangs 1992 AA 328 311 - Table 1
# beta pic photometry
t_lde = Table.read( """     JD          Vmag
                            4914.780    3.834
                            4914.857    3.836
                            4917.804    3.824
                            4917.857    3.824
                            4918.628    3.805
                            4918.720    3.835
                            4918.786    3.838
                            4918.856    3.845
                            4919.802    3.823
                            4919.853    3.824
                            4920.787    3.828
                            4920.859    3.828
                            4925.791    3.839
                            4925.847    3.839
                    """, format='ascii')

t = ascii.read('table', format='cds', readme='ReadMe')
print(t.meta)
print(t.colnames)

#tbp = t[(t['HD'] == 39060)]

f = plt.figure(figsize=(16,6))

ax1 = f.add_subplot(111)

ax1.scatter(t['JD']-2440000., t['Vmag'])


print(t_lde)
ax1.scatter(t_lde['JD'], t_lde['Vmag'], color='red',alpha=0.4, s=100)

ax1.set_xlim((2000,9000))
ax1.set_ylim((3.87, 3.80))

plt.show()


