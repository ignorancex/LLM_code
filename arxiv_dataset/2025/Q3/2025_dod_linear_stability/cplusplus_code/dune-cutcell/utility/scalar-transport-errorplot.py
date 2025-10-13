import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sb
import argparse
import json
import glob

# parser = argparse.ArgumentParser(description='Plot errors')

# parser.add_argument('m', type=int,
#                     help='number of components')

# args = parser.parse_args()

errorFiles = glob.glob('*-deg/error*')

errorData = []

for file in errorFiles:
    with open(file) as f:
        ret = json.load(f)
        errorData.append(ret)

errorData.sort(key = lambda error : (error['vfAngle'], error['numberOfCellsPerAxis']))
m = len(errorData[0]['L-infError'])

vfAngle = errorData[0]['vfAngle']
cutAngle = errorData[0]['cutAngle']
cflSafetyFactor = errorData[0]['cflFactor']
tau = errorData[0]['tau']

current_palette = sb.color_palette()

mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 24

fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)
fig.set_figheight(7)
fig.set_figwidth(15)
fig.set_tight_layout(True)

startIndex = 0
endIndex = len(errorData)
currentIndex = 0
numberOfErrors = 0
colorIndex = 0
N = None
tick_spacing = 6

while startIndex != endIndex:
    currentAngle = errorData[startIndex]['vfAngle']

    while currentIndex != endIndex and errorData[currentIndex]['vfAngle'] == currentAngle:
        currentIndex += 1

    numberOfErrors = currentIndex - startIndex

    N = np.zeros((numberOfErrors))
    LInfError = np.zeros((m, numberOfErrors))
    L2Error = np.zeros((m, numberOfErrors))
    seminormError = np.zeros((m, numberOfErrors))

    for i in range(startIndex, currentIndex):
        N[i - startIndex] = errorData[i]['numberOfCellsPerAxis']

        for j in range(0, m):
            LInfError[j, i - startIndex] = errorData[i]['L-infError'][j]
            L2Error[j, i - startIndex] = errorData[i]['L2Error'][j]
            seminormError[j, i - startIndex] = errorData[i]['seminormError'][j]

    expectedOrder = errorData[0]['order']

    labelText = f'${np.round(np.rad2deg(currentAngle))}^{{\\circ}}$'
    color = current_palette[colorIndex]

    LInfOrder = np.polyfit(np.log(N), np.log(LInfError[0, :]), 1)

    space = ''

    if (np.round(np.rad2deg(currentAngle)) < 10.0):
        space = '  '

    # ax[1, 0].plot(N, LInfError[0, :],linestyle='dotted', marker='o',label=labelText + space + " {:.4f}".format(-LInfOrder[0]), color=color)
    # ax[0, 0].plot(N, np.exp(LInfOrder[1]) * N**(LInfOrder[0]), label="order {:.4f}".format(-LInfOrder[0]), color=current_palette[0])
    # ax[0, 0].plot(N, np.exp(LInfOrder[1]) * N[0]**expectedOrder * N[0]**LInfOrder[0] * N**(-expectedOrder), label="order {:.4f}".format(expectedOrder), color=current_palette[1])

    L2Order = np.polyfit(np.log(N), np.log(L2Error[0, :]), 1)

    ax[0, 0].plot(N, L2Error[0, :],linestyle='dotted', marker='o',label=labelText + space + " {:.2f}".format(-L2Order[0]), color=color)
    # ax[1, 0].plot(N, np.exp(L2Order[1]) * N**(L2Order[0]), label="order {:.4f}".format(-L2Order[0]), color=current_palette[0])
    # ax[1, 0].plot(N, np.exp(L2Order[1]) * N[0]**expectedOrder * N[0]**L2Order[0] * N**(-expectedOrder), label="order {:.4f}".format(expectedOrder), color=current_palette[1])

    seminormOrder = np.polyfit(np.log(N), np.log(seminormError[0, :]), 1)

    ax[0, 1].plot(N, seminormError[0, :],linestyle='dotted', marker='o',label=labelText + space + " {:.2f}".format(-seminormOrder[0]), color=color)
    # ax[0, 1].plot(N, np.exp(seminormOrder[1]) * N**(seminormOrder[0]), label="order {:.4f}".format(-seminormOrder[0]), color=current_palette[0])
    # ax[0, 1].plot(N, np.exp(seminormOrder[1]) * N[0]**expectedOrder * N[0]**seminormOrder[0] * N**(-expectedOrder), label="order {:.4f}".format(expectedOrder), color=current_palette[1])

    startIndex = currentIndex
    colorIndex += 1

ax[0, 0].plot(N, N**(-expectedOrder) * 1.0, label="order  {:.1f}".format(expectedOrder), color=current_palette[colorIndex])
# ax[1, 0].plot(N, N**(-expectedOrder) * 1.5, label="ref     {:.4f}".format(expectedOrder), color=current_palette[colorIndex])
ax[0, 1].plot(N, N**(-expectedOrder) * 20.0, label="order  {:.1f}".format(expectedOrder), color=current_palette[colorIndex])
ax[0, 1].plot(N, N**(-(expectedOrder / 2.0)) * 2.5, label="order  {:.1f}".format(expectedOrder / 2.0), color=current_palette[colorIndex + 1])

ax[0, 0].set_box_aspect(1)
ax[0, 0].set_title(f'Convergence $L^2$-error')

ax[0, 0].set_yscale('log')
ax[0, 0].set_xscale('log')

ticks = [100, 200, 400, 800]

# ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
# ax[0, 0].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32).astype(str)))
ax[0, 0].xaxis.set_major_locator(ticker.FixedLocator(ticks))
ax[0, 0].xaxis.set_major_formatter(ticker.FixedFormatter([str(tick) for tick in ticks]))
ax[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(N))
ax[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
# ax[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.005))
ax[0, 0].yaxis.set_major_locator(ticker.LogLocator(subs=[0.02, 0.01, 0.005, 0.002]))
ax[0, 0].yaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 2)))
ax[0, 0].legend()

ax[0, 0].set_xlabel('N=$\\frac{1}{h}$')
ax[0, 0].set_ylabel('Error')

ax[0, 1].set_box_aspect(1)
ax[0, 1].set_title(f'Convergence $\\beta$-seminorm error')

ax[0, 1].set_yscale('log')
ax[0, 1].set_xscale('log')

# ax[0, 1].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
# ax[0, 1].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32).astype(str)))
ax[0, 1].xaxis.set_major_locator(ticker.FixedLocator(ticks))
ax[0, 1].xaxis.set_major_formatter(ticker.FixedFormatter([str(tick) for tick in ticks]))
ax[0, 1].xaxis.set_minor_locator(ticker.FixedLocator(N))
ax[0, 1].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[0, 1].yaxis.set_major_locator(ticker.LogLocator(subs=[0.02, 0.01, 0.005, 0.002]))
ax[0, 1].yaxis.set_major_formatter(ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 2)))
ax[0, 1].legend()

ax[0, 1].set_xlabel('N=$\\frac{1}{h}$')
ax[0, 1].set_ylabel('Error')

# ax[1, 0].set_box_aspect(1)
# ax[1, 0].set_title(f'Convergence $L^{{\infty}}$-error')
# ax[1, 0].xaxis.set_major_locator(ticker.FixedLocator(N[::tick_spacing].astype(np.int32)))
# ax[1, 0].xaxis.set_major_formatter(ticker.FixedFormatter(N[::tick_spacing].astype(np.int32)))
# ax[1, 0].xaxis.set_minor_locator(ticker.FixedLocator(N))
# ax[1, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
# ax[1, 0].legend()

# ax[1, 0].set_yscale('log')
# ax[1, 0].set_xscale('log')

# ax[1, 0].set_xlabel('N=$\\frac{1}{h}$')
# ax[1, 0].set_ylabel('error')

# ax[1, 1].set_box_aspect(1)
# ax[1, 1].set_axis_off()
# ax[1, 1].text(0.1, 0.7, f'$\\beta$-angle: {np.round(np.rad2deg(vfAngle))}')
# ax[1, 1].text(0.1, 0.6, f'cut angle: {np.round(np.rad2deg(cutAngle) - 90.0)}')
# ax[1, 1].text(0.1, 0.5, f'CFL safety factor: {cflSafetyFactor}')
# ax[1, 1].text(0.1, 0.4, f'$\\tau$: {tau}')

# fig.subplots_adjust(wspace=0)

# plt.show()
plt.savefig('theorem-cfl-error-plots.pdf')
