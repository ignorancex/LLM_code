import sys, os, platform
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot_conf import ColorPalette, concise_fmt


def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb = colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict


cdict = NonLinCdict([0, 1], ['#ffffff', ColorPalette.CC4[0]])
ccmap = colors.LinearSegmentedColormap('test', cdict)


def main():
    timer = Timer()
    timer.start()

    sample_file = open('../wrangling/sample_bowtie_users.txt', 'r')
    complete_file = open('../wrangling/complete_bowtie_users.txt', 'r')

    sample_lscc = set(sample_file.readline().rstrip().split(','))
    sample_in = set(sample_file.readline().rstrip().split(','))
    sample_out = set(sample_file.readline().rstrip().split(','))
    sample_tube = set(sample_file.readline().rstrip().split(','))
    sample_tendrils = set(sample_file.readline().rstrip().split(','))
    sample_disc = set(sample_file.readline().rstrip().split(','))
    sample_list = [sample_lscc, sample_in, sample_out, sample_tube, sample_tendrils, sample_disc]

    num_total = sum([len(x) for x in sample_list])
    print('size of sample bow-tie', [len(x) for x in sample_list], num_total)
    print('ratio of sample bow-tie', [100 * len(x) / num_total for x in sample_list])

    complete_lscc = set(complete_file.readline().rstrip().split(','))
    complete_in = set(complete_file.readline().rstrip().split(','))
    complete_out = set(complete_file.readline().rstrip().split(','))
    complete_tube = set(complete_file.readline().rstrip().split(','))
    complete_tendrils = set(complete_file.readline().rstrip().split(','))
    complete_disc = set(complete_file.readline().rstrip().split(','))
    complete_list = [complete_lscc, complete_in, complete_out, complete_tube, complete_tendrils, complete_disc]

    num_total = sum([len(x) for x in complete_list])
    print('size of complete bow-tie', [len(x) for x in complete_list], num_total)
    print('ratio of complete bow-tie', [100 * len(x) / num_total for x in complete_list])

    col_labels = ['LSCC', 'IN', 'OUT', 'Tubes', 'Tendrils', 'Disc.', 'Missing', 'Total']
    row_labels = ['LSCC', 'IN', 'OUT', 'Tubes', 'Tendrils', 'Disc.', 'Total']
    n_row = len(row_labels)
    n_col = len(col_labels)
    confusion_mat = np.zeros(shape=(n_row, n_col))
    confusion_mat_rate = np.zeros(shape=(n_row, n_col))
    confusion_mat_annot = [[[] for _ in range(n_col)] for _ in range(n_row)]
    for i in range(n_row - 1):
        cnt0 = cnt = len(complete_list[i])
        print('from complete {0}'.format(row_labels[i]))
        for j in range(n_col - 2):
            tmp = len(complete_list[i].intersection(sample_list[j]))
            print('>>> to sample {0}: '.format(col_labels[j]), tmp, tmp/cnt0)
            confusion_mat[i, j] = tmp
            confusion_mat_rate[i, j] = tmp/cnt0
            cnt -= tmp
            if tmp > 0:
                confusion_mat_annot[i][j] = '{0}\n{1:.1f}%'.format(concise_fmt(tmp, None), 100*tmp/cnt0)
            else:
                confusion_mat_annot[i][j] = '{0}'.format(concise_fmt(tmp, None))
        print('>>> to missing: ', cnt/cnt0)
        confusion_mat[i, -2] = cnt
        if cnt > 0:
            confusion_mat_annot[i][-2] = '{0}\n{1:.1f}%'.format(concise_fmt(cnt, None), 100*cnt / cnt0)
        else:
            confusion_mat_annot[i][-2] = '{0}'.format(concise_fmt(cnt, None))
        confusion_mat_rate[i, -2] = cnt / cnt0
        confusion_mat[i, -1] = cnt0
        # confusion_mat_rate[i, -1] = 0

    for j in range(n_row):
        # confusion_mat_annot[j][-1] = '{0}\n{1:.1f}%'.format(concise_fmt(confusion_mat[j, -1]), 100*confusion_mat[j, -1]/sum(confusion_mat[:-1, -1]))
        confusion_mat_annot[j][-1] = '{0}'.format(concise_fmt(confusion_mat[j, -1], None))
        # confusion_mat_annot[-1][j] = '{0}\n{1:.1f}%'.format(concise_fmt(sum(confusion_mat[:-1, j])), 100*sum(confusion_mat[:-1, j]) / sum(confusion_mat[:-1, -1]))
        confusion_mat_annot[-1][j] = '{0}'.format(concise_fmt(sum(confusion_mat[:-1, j]), None))
        # confusion_mat_rate[-1, j] = 0
    # confusion_mat_annot[-1][-1] = '{0}\n{1:.0f}%'.format(concise_fmt(sum(confusion_mat[:-1, -1])), 100)
    confusion_mat_annot[-1][-1] = '{0}'.format(concise_fmt(sum(confusion_mat[:-1, -1]), None))

    confusion_mat_annot = np.array(confusion_mat_annot)

    fig, ax1 = plt.subplots(1, 1)
    sns.heatmap(confusion_mat_rate, annot=confusion_mat_annot, cmap=ccmap,
                fmt='s', ax=ax1,
                cbar_kws={'label': 'ratio from complete bow-tie to sample bow-tie', 'shrink': .6})

    ax1.set_title('sample set')
    ax1.set_title('complete set', loc='left')
    ax1.set_xticklabels(col_labels, ha='center')
    ax1.set_yticklabels(row_labels, rotation=90, va='center')
    ax1.xaxis.tick_top()
    ax1.hlines(y=0, xmin=n_col-1, xmax=n_col)
    ax1.hlines(y=n_row-1, xmin=0, xmax=n_col-1)
    ax1.hlines(y=n_row, xmin=0, xmax=n_col)
    ax1.vlines(x=0, ymin=n_row-1, ymax=n_row)
    ax1.vlines(x=n_col-1, ymin=0, ymax=n_row-1)
    ax1.vlines(x=n_col, ymin=0, ymax=n_row)

    cbar_ax = fig.axes[-1]
    cbar_ax.set_frame_on(True)

    timer.stop()

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    plt.savefig('../images/measure_bowtie_flow.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
