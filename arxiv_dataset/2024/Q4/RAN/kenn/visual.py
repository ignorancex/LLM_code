import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, title, start=0, end=1250, bool_time_major=True):
    if bool_time_major:
        data = data.transpose(1, 0)
    else:
        pass
    for i in range(len(data)):
        plt.plot(data[i][start:end])
    plt.title(title)


def np_relu(arr):
    _arr = np.array([], dtype=arr[0].dtype)
    for i in arr:
        if i > 0:
            _arr = np.append(_arr, i)
        else:
            _arr = np.append(_arr, 0)
    return _arr


def compute_density(cpt, w_s):
    half_w_s = int(w_s / 2)
    _len = len(cpt)
    _score = np_relu(cpt)
    _score = np.array([], dtype=cpt[0].dtype)
    _density = np.array([], dtype=float)
    for i in range(_len):
        if i <= half_w_s:
            current_w_s = half_w_s + i
            _score = np.append(_score, np.sum(cpt[0:current_w_s]))
            _density = np.append(_density, np.sum(cpt[0:current_w_s]) / current_w_s)
        elif half_w_s < i < (_len - half_w_s):
            current_w_s = w_s
            _score = np.append(_score, np.sum(cpt[i - half_w_s:i + half_w_s]))
            _density = np.append(_density, np.sum(cpt[i - half_w_s:i + half_w_s]) / current_w_s)
        else:
            current_w_s = half_w_s + _len - i
            _score = np.append(_score, np.sum(cpt[i - half_w_s:_len]))
            _density = np.append(_density, np.sum(cpt[i - half_w_s:_len]) / current_w_s)
    _range = np.max(_density) - np.min(_density)
    return (_density - np.min(_density)) / _range



def plot_timegate(idx, labels, plot_cls=None):
    color_pool = ['gray', 'red', 'cyan', 'tan', 'orange', 'yellow', 'green', 'navy', 'olive',
                  'violet', 'yellowgreen', 'indigo', 'teal', 'tomato', 'gold', 'hotpink', 'palegreen', 'peru']
    labels_seg = labels
    total_cls = np.arange(18)
    plot_len = len(labels)
    plot_cls_gate = np.zeros([len(total_cls), plot_len], dtype=int)

    # print(np.unique(labels_seg))
    # print(np.unique(labels_seg == plot_cls))
    for n in range(len(total_cls)):
        for i in range(plot_len):
            if labels_seg[i] == n:
                plot_cls_gate[n][i] = 1
            else:
                plot_cls_gate[n][i] = 0
    # print(np.unique(plot_cls_gate[plot_cls]))
    # plt.figure(figsize=(5, 2))
    if plot_cls is None:
        for n in range(len(total_cls)):
            if n in np.unique(labels_seg):
                if n == 0:
                    plt.plot(plot_cls_gate[n], linestyle=':', color=color_pool[n], label=n)
                else:
                    plt.plot(plot_cls_gate[n], color=color_pool[n], label=n)

    else:
        if plot_cls in np.unique(labels_seg):
            plt.plot(plot_cls_gate[plot_cls])
        else:
            print('Dont exist')
    plt.yticks([])
    plt.ylim([0.1, 1.3])
    plt.legend(loc='best', ncol=5)
    # plt.title('Idx:[{}],\nInclude [{}]'.format(idx, np.unique(labels_seg)))
    # plt.show()
