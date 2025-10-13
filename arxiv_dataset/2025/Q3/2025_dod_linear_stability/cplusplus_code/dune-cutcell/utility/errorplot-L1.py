import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sb
import argparse
import json
import glob

errorFiles = glob.glob('error*')

errorData = []

for file in errorFiles:
    with open(file) as f:
        ret = json.load(f)
        errorData.append(ret)

errorData.sort(key = lambda error : error['numberOfCellsPerAxis'])
m = len(errorData[0]['L-infError'])

numberOfErrors = len(errorFiles)
N = np.zeros((numberOfErrors))
LInfError = np.zeros((m, numberOfErrors))
L1Error = np.zeros((m, numberOfErrors))

for i in range(0, numberOfErrors):
    N[i] = errorData[i]['numberOfCellsPerAxis']

    for j in range(0, m):
        LInfError[j, i] = errorData[i]['L-infError'][j]
        L1Error[j, i] = errorData[i]['L1Error'][j]

current_palette = sb.color_palette()

fig, ax = plt.subplots(nrows=2, ncols=m, squeeze=False)

expectedOrder = errorData[0]['order']

tick_spacing = 6

for i in range(0, m):
    LInfOrder = np.polyfit(np.log(N), np.log(LInfError[i, :]), 1)

    ax[0, i].plot(N, LInfError[i, :],linestyle='dotted', marker='o',label="error", color=current_palette[0])
    ax[0, i].plot(N, np.exp(LInfOrder[1]) * N**(LInfOrder[0]), label="order {:.4f}".format(-LInfOrder[0]), color=current_palette[0])
    ax[0, i].plot(N, np.exp(LInfOrder[1]) * N[0]**expectedOrder * N[0]**LInfOrder[0] * N**(-expectedOrder), label="order {:.4f}".format(expectedOrder), color=current_palette[1])

    ax[0, i].set_yscale('log')
    ax[0, i].set_xscale('log')

    ax[0, i].set_box_aspect(1)
    ax[0, i].set_title(f'$L^{{\\infty}}$-error $u_{i}$')
    ax[0, i].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
    ax[0, i].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32)))
    ax[0, i].xaxis.set_minor_locator(ticker.FixedLocator(N))
    ax[0, i].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[0, i].legend()

    L1Order = np.polyfit(np.log(N), np.log(L1Error[i, :]), 1)

    ax[1, i].plot(N, L1Error[i, :],linestyle='dotted', marker='o',label="error", color=current_palette[0])
    ax[1, i].plot(N, np.exp(L1Order[1]) * N**(L1Order[0]), label="order {:.4f}".format(-L1Order[0]), color=current_palette[0])
    ax[1, i].plot(N, np.exp(L1Order[1]) * N[0]**expectedOrder * N[0]**L1Order[0] * N**(-expectedOrder), label="order {:.4f}".format(expectedOrder), color=current_palette[1])

    ax[1, i].set_yscale('log')
    ax[1, i].set_xscale('log')

    ax[1, i].set_box_aspect(1)
    ax[1, i].set_title(f'$L^1$-error $u_{i}$')
    ax[1, i].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
    ax[1, i].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32)))
    ax[1, i].xaxis.set_minor_locator(ticker.FixedLocator(N))
    ax[1, i].xaxis.set_minor_formatter(ticker.NullFormatter())
    ax[1, i].legend()

plt.show()
