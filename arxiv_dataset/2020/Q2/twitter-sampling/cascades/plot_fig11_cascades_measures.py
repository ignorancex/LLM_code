import sys, os, platform
from collections import defaultdict
import numpy as np
from scipy.stats import percentileofscore

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend
from powerlaw import plot_ccdf

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, melt_snowflake
from utils.plot_conf import ColorPalette, hide_spines


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'

    sample_cascade_size = {}
    sample_inter_arrival_time = []
    sample_cascade_influence = {}
    sample_cascade_influence_10m = defaultdict(int)
    sample_cascade_influence_1h = defaultdict(int)
    with open('../data/{0}_out/sample_retweet_{0}.txt'.format(app_name), 'r') as fin:
        for line in fin:
            root_tweet, cascades = line.rstrip().split(':')
            cascades = cascades.split(',')
            root_tweet = root_tweet.split('-')[0]
            retweets = [x.split('-')[0] for x in cascades]
            influences = [int(x.split('-')[1]) for x in cascades]
            sample_cascade_size[root_tweet] = len(retweets)
            sample_cascade_influence[root_tweet] = sum(influences)
            root_timestamp = melt_snowflake(root_tweet)[0] / 1000
            retweet_timestamp_list = [root_timestamp]

            for i in range(len(retweets)):
                retweet_time = melt_snowflake(retweets[i])[0]/1000
                relative_retweet_time = retweet_time - root_timestamp
                retweet_timestamp_list.append(melt_snowflake(retweets[i])[0]/1000)
                if relative_retweet_time < 10 * 60:
                    sample_cascade_influence_10m[root_tweet] += influences[i]
                if relative_retweet_time < 60 * 60:
                    sample_cascade_influence_1h[root_tweet] += influences[i]

            for i in range(len(retweet_timestamp_list) - 1):
                sample_inter_arrival_time.append(retweet_timestamp_list[i+1] - retweet_timestamp_list[i])

    complete_cascade_size = {}
    complete_inter_arrival_time = []
    complete_cascade_influence = {}
    complete_cascade_influence_10m = defaultdict(int)
    complete_cascade_influence_1h = defaultdict(int)
    with open('../data/{0}_out/complete_retweet_{0}.txt'.format(app_name), 'r') as fin:
        for line in fin:
            root_tweet, cascades = line.rstrip().split(':')
            cascades = cascades.split(',')
            root_tweet = root_tweet.split('-')[0]
            retweets = [x.split('-')[0] for x in cascades]
            complete_cascade_size[root_tweet] = len(retweets)
            if len(retweets) >= 50:
                influences = [int(x.split('-')[1]) for x in cascades]
                complete_cascade_influence[root_tweet] = sum(influences)
                root_timestamp = melt_snowflake(root_tweet)[0] / 1000
                retweet_timestamp_list = [root_timestamp]

                for i in range(len(retweets)):
                    retweet_time = melt_snowflake(retweets[i])[0] / 1000
                    relative_retweet_time = retweet_time - root_timestamp
                    retweet_timestamp_list.append(melt_snowflake(retweets[i])[0] / 1000)
                    if relative_retweet_time < 10 * 60:
                        complete_cascade_influence_10m[root_tweet] += influences[i]
                    if relative_retweet_time < 60 * 60:
                        complete_cascade_influence_1h[root_tweet] += influences[i]

                for i in range(len(retweet_timestamp_list) - 1):
                    complete_inter_arrival_time.append(retweet_timestamp_list[i + 1] - retweet_timestamp_list[i])

    print('number of cascades in the complete set', len(complete_cascade_size))
    print('number of cascades in the sample set', len(sample_cascade_size))

    print('mean complete size', np.mean(list(complete_cascade_size.values())))
    print('mean sample size', np.mean(list(sample_cascade_size.values())))

    print('complete #cascades (≥50 retweets)', sum([1 for x in list(complete_cascade_size.values()) if x >= 50]))
    print('sample #cascades (≥50 retweets)', sum([1 for x in list(sample_cascade_size.values()) if x >= 50]))

    num_complete_cascades_in_sample = 0
    complete_cascades_in_sample_size_list = []
    num_complete_cascades_in_sample_50 = 0
    for root_tweet in sample_cascade_size:
        if sample_cascade_size[root_tweet] == complete_cascade_size[root_tweet]:
            num_complete_cascades_in_sample += 1
            complete_cascades_in_sample_size_list.append(complete_cascade_size[root_tweet])
            if complete_cascade_size[root_tweet] >= 50:
                num_complete_cascades_in_sample_50 += 1
    print('number of complete cascades in the sample set', num_complete_cascades_in_sample)
    print('number of complete cascades (>50 retweets) in the sample set', num_complete_cascades_in_sample_50)
    print('max: {0}, mean: {1}'.format(max(complete_cascades_in_sample_size_list), np.mean(complete_cascades_in_sample_size_list)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    cc4 = ColorPalette.CC4
    blue = cc4[0]
    red = cc4[3]

    sample_median = np.median(sample_inter_arrival_time)
    complete_median = np.median(complete_inter_arrival_time)

    plot_ccdf(sample_inter_arrival_time, ax=axes[0], color=blue, ls='-', label='sample')
    plot_ccdf(complete_inter_arrival_time, ax=axes[0], color='k', ls='-', label='complete')

    axes[0].plot([sample_median, sample_median], [0, 1], color=blue, ls='--', lw=1)
    axes[0].plot([complete_median, complete_median], [0, 1], color='k', ls='--', lw=1)

    print('\ninter_arrival_time sample median', sample_median)
    print('inter_arrival_time complete median', complete_median)

    axes[0].set_xscale('symlog')
    axes[0].set_xticks([0, 1, 100, 10000, 1000000])
    axes[0].set_yscale('linear')
    axes[0].set_xlabel('inter-arrival time (sec)', fontsize=16)
    axes[0].set_ylabel('$P(X \geq x)$', fontsize=16)
    axes[0].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True, loc='upper right')
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].set_title('(a)', fontsize=18, pad=-3*72, y=1.0001)

    influence_list = []
    influence_list_10m = []
    influence_list_1h = []
    for root_tweet in sample_cascade_size:
        if complete_cascade_size[root_tweet] >= 50:
            if complete_cascade_influence[root_tweet] > 0:
                influence_list.append(sample_cascade_influence[root_tweet] / complete_cascade_influence[root_tweet])
            if complete_cascade_influence_10m[root_tweet] > 0:
                influence_list_10m.append(sample_cascade_influence_10m[root_tweet] / complete_cascade_influence_10m[root_tweet])
            if complete_cascade_influence_1h[root_tweet] > 0:
                influence_list_1h.append(sample_cascade_influence_1h[root_tweet] / complete_cascade_influence_1h[root_tweet])

    plot_ccdf(influence_list_10m, ax=axes[1], color=red, ls='-', label='10m')
    plot_ccdf(influence_list_1h, ax=axes[1], color=blue, ls='-', label='1h')
    plot_ccdf(influence_list, ax=axes[1], color='k', ls='-', label='14d')

    print('influence_list median', np.median(influence_list))
    print('influence_list_1h median', np.median(influence_list_1h))
    print('influence_list_10m median', np.median(influence_list_10m))

    print('influence_list 0.25', percentileofscore(influence_list, 0.25))
    print('influence_list 0.25', percentileofscore(influence_list_1h, 0.25))
    print('influence_list 0.25', percentileofscore(influence_list_10m, 0.25))

    print('influence_list 0.75', percentileofscore(influence_list, 0.75))
    print('influence_list 0.75', percentileofscore(influence_list_1h, 0.75))
    print('influence_list 0.75', percentileofscore(influence_list_10m, 0.75))

    axes[1].set_xscale('linear')
    axes[1].set_yscale('linear')
    axes[1].set_xlabel('relative potential reach', fontsize=16)
    # axes[1].set_ylabel('$P(X \geq x)$', fontsize=16)
    axes[1].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True, loc='upper right')
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('(b)', fontsize=18, pad=-3*72, y=1.0001)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/cascades_measures.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
