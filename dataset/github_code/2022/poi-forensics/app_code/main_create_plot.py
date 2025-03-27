import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf
def gaus2prob(x):
    return (1-erf(x/np.sqrt(2.0))) / 2


def create_plot(dict_out, output_image, dist_normalization):
    embs_track = np.asarray(dict_out['embs_track'])
    embs_dists = np.asarray(dict_out['embs_dists'])
    while len(embs_dists.shape) > 1:
        embs_dists = embs_dists[..., -1]
    embs_range = np.asarray(dict_out['embs_range'])
    if 'global_score' in dict_out:
        global_score = dict_out['global_score']
    else:
        global_score = None

    xmin = np.PINF
    xmax = 0
    fig = plt.figure(figsize=(12, 6))
    for ids in np.unique(embs_track):
        inds = embs_track == ids
        dist = embs_dists[inds]
        rang = np.mean(embs_range[inds], -1) / 25.0
        xmin = min(xmin, np.min(rang))
        xmax = max(xmax, np.max(rang))
        if dist_normalization:
            plt.semilogy(rang, gaus2prob(dist), 'k', linewidth=2)
        else:
            plt.plot(rang, dist, 'k', linewidth=2)

    if global_score is not None:
        if dist_normalization:
            plt.hlines(gaus2prob(global_score), xmin, xmax, 'b', linestyles='dashdot', label='global_score')
        else:
            plt.hlines(global_score, xmin, xmax, 'b', linestyles='dashdot', label='global_score')

    plt.grid()
    plt.legend()
    plt.xlabel('time')
    if dist_normalization:
        plt.gca().invert_yaxis()
    else:
        plt.ylabel('distance')
    fig.savefig(output_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='create plot.')
    parser.add_argument('--file_npz', type=str,
                        help='numpy file.')
    parser.add_argument('--output_image', type=str,
                        help='output image (with extension .png).')
    parser.add_argument('--dist_normalization', type=int, default=0,
                        help="if True, the distances are normalized using on the values obtained on pristine videos.")
    argd = parser.parse_args()

    create_plot(np.load(argd.file_npz), argd.output_image)
