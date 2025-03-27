from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging

from matplotlib import pyplot as plt

x_iter = [32,64,128,256 ]
y_fb_valence_3 = [0.6077,0.6163,0.626,0.6170]

# y_fb_valence_5 = [0.632, 0.659, 0.685, 0.654, 0.647]
# y_fb_valence_self = [0.634, 0.634, 0.634, 0.634, 0.634]
# y_fb_arousal_3 = [0.892, 0.908, 0.912, 0.903, 0.899]
# y_fb_arousal_5 = [0.902, 0.912, 0.918, 0.913, 0.902]
# y_fb_arousal_self = [0.905, 0.905, 0.905, 0.905, 0.905,]

font = {'family': 'Times New Roman', 'color': 'black',
        'weight': 'normal', 'size': 22}

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(x_iter, y_fb_valence_3, 'rs-',
                      linewidth=2.2, label='Val acc')

    # line2, = ax1.plot(x_iter, y_fb_valence_5, 'g^-',
    #                   linewidth=2.2, label='Valence 5 cap')
    # line3, = ax1.plot(x_iter, y_fb_valence_self, 'ch--',
    #                   linewidth=2.2, label='Valence self-attention')

    # ax1.set_xlim()
    # ax1.set_ylim(0.62, 0.695)
    ax1.set_ylabel('acc on dev set', fontdict=font)
    ax1.set_xlabel('Batch_size', fontdict=font)
    ax1.tick_params(labelsize=15)

    # ax2 = ax1.twinx()
    # line4, = ax2.plot(x_iter, y_fb_arousal_3, 'b*-',
    #                   linewidth=2.2, label='Arousal 3 cap')
    # line5, = ax2.plot(x_iter, y_fb_arousal_5, 'ko-',
    #                   linewidth=2.2, label='Arousal 5 cap')
    # line6, = ax2.plot(x_iter, y_fb_arousal_self, 'mp--',
    #                   linewidth=2.2, label='Arousal self-attention')

    # ax2.set_ylim(0.885, 0.925)
    # ax2.tick_params(labelsize=16)
    plt.legend(handles=[line1,  ], prop={
               'size': 16, 'family': 'Times New Roman'}, bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    # foo_fig = plt.gcf()
    plt.savefig('batch_size',format="eps")
    plt.show()
