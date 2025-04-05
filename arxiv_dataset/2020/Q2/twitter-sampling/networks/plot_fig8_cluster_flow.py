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

    n_cluster = 6

    complete_cluster_size_list = []
    for i in range(n_cluster):
        with open('{0}_cluster{1}.txt'.format('complete', i), 'r') as fin:
            complete_nodes = set(fin.readline().split(','))
            num_entities = len(complete_nodes)
            num_users = len([x for x in complete_nodes if x.startswith('u')])
            num_hashtags = num_entities - num_users
            complete_cluster_size_list.append((num_entities, num_users, num_hashtags))
    complete_sorted_by_size = sorted(enumerate(complete_cluster_size_list), key=lambda x: x[1][0], reverse=True)
    complete_sorted_by_size_copy = []
    complete_sorted_by_size_copy.append(complete_sorted_by_size[0])
    complete_sorted_by_size_copy.append(complete_sorted_by_size[4])
    complete_sorted_by_size_copy.append(complete_sorted_by_size[5])
    complete_sorted_by_size_copy.append(complete_sorted_by_size[3])
    complete_sorted_by_size_copy.append(complete_sorted_by_size[2])
    complete_sorted_by_size_copy.append(complete_sorted_by_size[1])
    complete_sorted_by_size = complete_sorted_by_size_copy
    print(complete_sorted_by_size)

    sample_cluster_size_list = []
    for i in range(n_cluster):
        with open('{0}_cluster{1}.txt'.format('sample', i), 'r') as fin:
            sample_nodes = set(fin.readline().split(','))
            num_entities = len(sample_nodes)
            num_users = len([x for x in sample_nodes if x.startswith('u')])
            num_hashtags = num_entities - num_users
            sample_cluster_size_list.append((num_entities, num_users, num_hashtags))
    sample_sorted_by_size = sorted(enumerate(sample_cluster_size_list), key=lambda x: x[1][0], reverse=True)
    sample_sorted_by_size_copy = []
    sample_sorted_by_size_copy.append(sample_sorted_by_size[0])
    sample_sorted_by_size_copy.append(sample_sorted_by_size[1])
    sample_sorted_by_size_copy.append(sample_sorted_by_size[4])
    sample_sorted_by_size_copy.append(sample_sorted_by_size[2])
    sample_sorted_by_size_copy.append(sample_sorted_by_size[5])
    sample_sorted_by_size_copy.append(sample_sorted_by_size[3])
    sample_sorted_by_size = sample_sorted_by_size_copy
    print(sample_sorted_by_size)

    complete_clusters_list = []
    for i, _ in complete_sorted_by_size:
        with open('{0}_cluster{1}.txt'.format('complete', i), 'r') as fin:
            complete_nodes = set(fin.readline().split(','))
            complete_clusters_list.append(complete_nodes)

    sample_clusters_list = []
    for i, _ in sample_sorted_by_size:
        with open('{0}_cluster{1}.txt'.format('sample', i), 'r') as fin:
            sample_nodes = set(fin.readline().split(','))
            sample_clusters_list.append(sample_nodes)

    col_labels = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'Missing', 'Total']
    row_labels = ['CC1', 'CC2', 'CC3', 'CC4', 'CC5', 'CC6', 'Total']
    n_row = len(row_labels)
    n_col = len(col_labels)
    confusion_mat = np.zeros(shape=(n_row, n_col))
    confusion_mat_rate = np.zeros(shape=(n_row, n_col))
    confusion_mat_annot = [[[] for _ in range(n_col)] for _ in range(n_row)]
    for i in range(n_row - 1):
        cnt0 = cnt = complete_sorted_by_size[i][1][0]
        print('from complete cluster {0}'.format(i + 1), cnt0)
        for j in range(n_col - 2):
            tmp = len(complete_clusters_list[i].intersection(sample_clusters_list[j]))
            print('>>> to sample cluster {0}: '.format(j + 1), tmp, tmp/cnt0)
            confusion_mat[i, j] = tmp
            confusion_mat_rate[i, j] = tmp / cnt0
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
                cbar_kws={'label': 'ratio from complete clusters to sample clusters', 'shrink': .6})

    ax1.set_title('clusters in sample set', loc='right')
    ax1.set_title('clusters in complete set', loc='left')
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
    plt.savefig('../images/measure_bipartite_cluster_flow.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
