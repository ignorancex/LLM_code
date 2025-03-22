import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Global vars

LOG_PATH_DATE = 'SET-TO-FOLDER-NAME-IN-LOGS'

MA_SMOOTH = 1*0.00025
MA_SKIP_THRESH = 1  # 1 --> a single data point is not allowed to more than double the current moving average
INCLUDE_MONTHWISE = False
SKIP_MONTHS = [4,5,9]
STEP_SIZE_DISPLAY = 25
SCALE_FACTOR = 2927.179633
SCALE_FACTOR = 191.4835771
X_AXIS_END = None
MAX_Y = 0.0015/3  # 0.0015/3, 0.0015 or 0.0055 -- only krycklan data
MAX_Y = 3.5*0.0015/3
STAT_NAME = 'L2_loss-other'

# LOCATIONS_TRAIN = ['Knislinge', 'Krycklan', 'Skivarp', 'Skovde', 'Tumba', 'Dalbergsan', 'Degea', 'Lillan_blekinge', 'Stavan']
# LOCATIONS_VAL = ['Jonkoping', 'Hassjaan', 'Lillan']
BL1_TRAIN = 4.1985e-06 * (2927.179633 / 191.4835771) ** 2
BL2_TRAIN = 0  # not used
BL3_TRAIN =2.1912e-07 * (2927.179633 / 191.4835771) ** 2
BL1_VAL = 4.8888e-07 * (2927.179633 / 191.4835771) ** 2
BL2_VAL = 0  # not used
BL3_VAL = 4.0730e-08 * (2927.179633 / 191.4835771) ** 2

def _custom_ma(data, ma_smooth=MA_SMOOTH):
    for idx, val in enumerate(data['values']):
        if idx < 25:
            data['mas_custom'][idx] = data['means'][idx]
        else:
            # Filter out "outlier" data so that it doesn't corrupt the MA
            if idx >= 50 and np.abs(((1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]) / data['mas_custom'][idx - 1] - 1) > MA_SKIP_THRESH:
                data['mas_custom'][idx] = data['mas_custom'][idx - 1]
            else:
                data['mas_custom'][idx] = (1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]

def _plot(datas, title='', xlabel='# training batches', ylabel='RMSE ($\mathregular{m}^3$ / sec)', start_it=0, max_x=None, max_y=None,
          force_axis=False, baseline1=None, baseline2=None, baseline3=None, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 2, 1)
    else:
        ax = fig.add_subplot(1, 2, 2)
    legend_entries = []
    for data in datas:
        x = data[0]['times']
        y = SCALE_FACTOR*np.sqrt(data[0]['mas_custom'])
        if X_AXIS_END is not None:
            x = x[:int(X_AXIS_END / (x[1] - x[0]))]
            y = y[:int(X_AXIS_END / (x[1] - x[0]))]
        id = data[1]
        if id == -1:
            legend_entries.append('ML-model')
        else:
            legend_entries.append(id + 1)
        plt.plot(x[start_it:], y[start_it:])
    print("nbr-data", len(y[start_it:]), "min-err", np.min(y[start_it:]))
    if baseline1 is not None:
        plt.plot([x[start_it], x[-1]], [SCALE_FACTOR*baseline1, SCALE_FACTOR*baseline1], linestyle='--', c='r')
        legend_entries.append('mean-per-site')
    if baseline2 is not None and not USE_NEW_DATA:
        plt.plot([x[start_it], x[-1]], [SCALE_FACTOR*baseline2, SCALE_FACTOR*baseline2], linestyle='--', c='c')
        legend_entries.append('mean-of-all-flows')
    if baseline3 is not None:
        plt.plot([x[start_it], x[-1]], [SCALE_FACTOR*baseline3, SCALE_FACTOR*baseline3], linestyle=':', c='g')
        legend_entries.append('prev-flow')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0, 50000, 100000, 150000, 200000, 250000])
    plt.grid(True)
    plt.legend(legend_entries)
    ax = plt.gca()
    if max_x is None:
        max_x = x[-1]
    if max_y is None:
        max_y = max(np.max(y['means'][start_it:]), np.max(y['mas'][start_it:]))
    if force_axis:
        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
    else:
        ax.set_xlim([0, min(max_x, x[-1])])
        ax.set_ylim([0, min(max_y, max(np.max(y['means'][start_it:]), np.max(y['mas'][start_it:])))])
    ax.set_aspect(max_x / max_y)
    return fig

# Read data from log path
if not isinstance(LOG_PATH_DATE, list):
    LOG_PATH_DATE = [LOG_PATH_DATE]
L2_losses_all = []
L2_losses_all_val = []
for log_path_date in LOG_PATH_DATE:
    log_path = os.path.join('log', log_path_date, 'train_stats', STAT_NAME + '.npz')
    L2_losses = np.load(log_path)
    L2_losses = {'means': L2_losses['means'], 'mas': L2_losses['mas'],
                 'values': L2_losses['values'], 'times': L2_losses['times'],
                 'mas_custom': np.zeros_like(L2_losses['mas'])}
    log_path = os.path.join('log', log_path_date, 'train_stats', STAT_NAME + '_val.npz')
    L2_losses_val = np.load(log_path)
    L2_losses_val = {'means': L2_losses_val['means'], 'mas': L2_losses_val['mas'],
                     'values': L2_losses_val['values'], 'times': [10 * vv for vv in L2_losses_val['times']],
                     'mas_custom': np.zeros_like(L2_losses_val['mas'])}

    # Create MA-smoothing of raw data
    _custom_ma(L2_losses)
    _custom_ma(L2_losses_val, ma_smooth=10*MA_SMOOTH)

    # Add to list of all
    L2_losses_all.append([L2_losses, -1])
    L2_losses_all_val.append([L2_losses_val, -1])

if False:
    L2_losses_all = [[L2_losses, -1]]
    L2_losses_all_val = [[L2_losses_val, -1]]
    if INCLUDE_MONTHWISE:
        L2_losses_month = []
        L2_losses_month_val = []
        for i in range(12):
            if i + 1 in SKIP_MONTHS:
                continue
            log_path = os.path.join('log', LOG_PATH_DATE, 'train_stats', 'L2_loss-month-' + str(i+1) + '.npz')
            losses_i = np.load(log_path)
            losses_i = {'means': losses_i['means'], 'mas': losses_i['mas'],
                        'values': losses_i['values'], 'times': losses_i['times'],
                        'mas_custom': np.zeros_like(losses_i['mas'])}
            _custom_ma(losses_i)
            L2_losses_month.append([losses_i, i])
        L2_losses_all = L2_losses_all + L2_losses_month
        for i in range(12):
            if i + 1 in SKIP_MONTHS:
                continue
            log_path = os.path.join('log', LOG_PATH_DATE, 'train_stats', 'L2_loss-month-' + str(i+1) + '_val.npz')
            losses_i = np.load(log_path)
            losses_i = {'means': losses_i['means'], 'mas': losses_i['mas'],
                        'values': losses_i['values'], 'times': [10 * vv for vv in losses_i['times']],
                        'mas_custom': np.zeros_like(losses_i['mas'])}
            _custom_ma(losses_i, ma_smooth=10*MA_SMOOTH)
            L2_losses_month_val.append([losses_i, i])
        L2_losses_all_val = L2_losses_all_val + L2_losses_month_val

# Plot results
fig_out = _plot(L2_losses_all, max_y=np.sqrt(MAX_Y) * SCALE_FACTOR, force_axis=True, baseline1=np.sqrt(BL1_TRAIN), baseline2=np.sqrt(BL2_TRAIN), baseline3=np.sqrt(BL3_TRAIN))
_plot(L2_losses_all_val, max_y=np.sqrt(MAX_Y) * SCALE_FACTOR, force_axis=True, baseline1=np.sqrt(BL1_VAL), baseline2=np.sqrt(BL2_VAL), baseline3=np.sqrt(BL3_VAL), fig=fig_out)

fig_out.savefig('result_plot.png')
fig_out.savefig('result_plot.eps')
plt.cla()
plt.clf()
plt.close('all')
print("Saved result plot!")
