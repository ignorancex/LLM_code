import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sb
import argparse
import json
import glob

errorFiles = glob.glob('operator-norm*')

errorData = []

for file in errorFiles:
    with open(file) as f:
        ret = json.load(f)
        errorData.append(ret)

errorData.sort(key = lambda error : error['cellFraction'])

numberOfErrors = len(errorFiles)
cellFractions = np.zeros((numberOfErrors))
operatorNorms = np.zeros((numberOfErrors))

for i in range(0, numberOfErrors):
    cellFractions[i] = errorData[i]['cellFraction']
    operatorNorms[i] = errorData[i]['operatorNorm']

print(cellFractions)

current_palette = sb.color_palette()

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

tick_spacing = 6

ax[0, 0].plot(cellFractions, operatorNorms, linestyle='dotted', marker='o',label="Operator norm", color=current_palette[0])

# ax[0, 0].set_yscale('log')
# ax[0, 0].set_xscale('log')

ax[0, 0].set_box_aspect(1)
ax[0, 0].set_title(f'Operator norm')
# ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
# ax[0, 0].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32)))
# ax[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(N))
# ax[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[0, 0].set_xlabel('Cell fraction')
ax[0, 0].set_ylabel('Operator norm')
ax[0, 0].legend()


plt.show()
