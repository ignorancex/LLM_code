from kenn.visual import *
from kenn.evaluate import *
import numpy as np

label_name = ['<null>',
              '<Open_Door_1>',
              '<Open_Door_2>',
              '<Close_Door_1>',
              '<Close_Door_2>',
              '<Open_Fridge>',
              '<Close_Fridge>',
              '<Open_Dishwasher>',
              '<Close_Dishwasher>',
              '<Open_Drawer 1>',
              '<Close_Drawer 1>',
              '<Open_Drawer 2>',
              '<Close_Drawer 2>',
              '<Open_Drawer 3>',
              '<Close_Drawer 3>',
              '<Clean_Table>',
              '<Drink_from Cup>',
              '<Toggle_Switch>',
              '<start>', '<end>']

att_map = np.load('log/oppo/att_map.npy')
pred_cap = np.load('log/oppo/pred_cap_lb.npy')
true_data = np.load('opportunity\data\gestures\sequential_label/test/data.npy')
true_cap = np.load('opportunity\data\gestures\sequential_label/test/caps.npy')
true_labels = np.load('opportunity\data\gestures\sequential_label/test/labels.npy')


def visualization(n):
    print(extract_output(true_cap[n]), extract_output(pred_cap[n]))
    '''plot data'''
    plt.subplot(411)
    plot_data(true_data[n], "True: %s, Pred: %s" %
              (extract_output(true_cap[n]).astype(int), extract_output(pred_cap[n]).astype(int)))
    plt.xticks([])
    plt.yticks([])
    '''plot true label'''
    plt.subplot(412)
    plot_timegate(n, true_labels[n])
    plt.xticks([])
    plt.yticks([])
    '''plot attention curve of the first activity'''
    plt.subplot(413)
    alp_curr = att_map[n, 0, :]
    alp_curr = (alp_curr - np.min(alp_curr)) / (np.max(alp_curr) - np.min(alp_curr))
    plt.text(0, 0.8, '{}'.format(pred_cap[n][0]))
    plt.bar(np.arange(len(alp_curr)), alp_curr)
    plt.plot(compute_density(alp_curr, 12), color='g')
    plt.plot(compute_density(alp_curr, 12) > 0.7, color='r')
    plt.xticks([])
    plt.yticks([])
    '''plot attention curve of the second activity'''
    plt.subplot(414)
    alp_curr = att_map[n, 1, :]
    alp_curr = (alp_curr - np.min(alp_curr)) / (np.max(alp_curr) - np.min(alp_curr))
    plt.text(0, 0.8, '{}'.format(pred_cap[n][1]))
    plt.bar(np.arange(len(alp_curr)), alp_curr)
    plt.plot(compute_density(alp_curr, 12), color='g')
    plt.plot(compute_density(alp_curr, 12) > 0.7, color='r')
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == '__main__':
    visualization(155)

# 1   -   ML_Both_Arms   -   Open Door 1
# 2   -   ML_Both_Arms   -   Open Door 2
# 3   -   ML_Both_Arms   -   Close Door 1
# 4   -   ML_Both_Arms   -   Close Door 2
# 5   -   ML_Both_Arms   -   Open Fridge
# 6   -   ML_Both_Arms   -   Close Fridge
# 7   -   ML_Both_Arms   -   Open Dishwasher
# 8   -   ML_Both_Arms   -   Close Dishwasher
# 9   -   ML_Both_Arms   -   Open Drawer 1
# 10   -   ML_Both_Arms   -   Close Drawer 1
# 11   -   ML_Both_Arms   -   Open Drawer 2
# 12   -   ML_Both_Arms   -   Close Drawer 2
# 13   -   ML_Both_Arms   -   Open Drawer 3
# 14   -   ML_Both_Arms   -   Close Drawer 3
# 15   -   ML_Both_Arms   -   Clean Table
# 16   -   ML_Both_Arms   -   Drink from Cup
# 17   -   ML_Both_Arms   -   Toggle Switch
