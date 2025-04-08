import os
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

seg_len = 30
sigma = 40
plot_eer = True 
target_fnr = 0.1
traj_auc = 75.42
vae_auc = 75.11

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]

# specify the directory path
path1 = "/home/galinezh/TCVAE_modified/trajectory_results/CPCC_scores/CPCC6"
path2 = "/home/galinezh/TCVAE_modified/VAE_results/score6"
# save_dir = '/home/galinezh/anomaly_baseline/combined_results_final'
# order_path = "/home/galinezh/TCVAE_modified/results/shang_score_pose_no_vis/order.txt"
# initialize an empty dataframe
result = pd.DataFrame()


# with open(order_path, "r") as file:
#     # Read the lines of the file into a list
#     order = file.readlines()
clip_list = os.listdir(path1)
clip_list = sorted(fn for fn in clip_list if fn.endswith('.csv'))   
    
# loop through all files in the directory
for file in clip_list:
    file = file.split('.')[0] + '.csv'
    
    # read the csv file
    df1 = pd.read_csv(os.path.join(path1, file), header=None)
    df2 = pd.read_csv(os.path.join(path2, file), header=None)
    df1 = df1.rename(columns={0: 'gt', 1: 'traj_score'})
    df2 = df2.rename(columns={0: 'gt', 1: 'recon_score'})

    result = result.append(pd.concat([df1, df2['recon_score']], axis=1), ignore_index=True)
        
# check the result dataframe


recon_scores_shifted = np.zeros_like(result['recon_score']) 
shift = seg_len + (seg_len // 2) - 1
recon_scores_shifted[shift:] = result['recon_score'][:-shift]
recon_scores_smoothed = gaussian_filter1d(recon_scores_shifted, sigma)
result['recon_score'] = recon_scores_smoothed
scaler1 = StandardScaler()
scaler2 = StandardScaler()
result[['traj_score_s']] = scaler1.fit_transform(result[['traj_score']])
result[['recon_score_s']] = scaler2.fit_transform(result[['recon_score']])


print ("*********************Combined************************")
# combined_score = result['traj_score_s'] + result['recon_score_s']
# combined_score = []
# for s1, s2 in zip (result['traj_score_s'] , result['recon_score_s']):
#     combined_score.append(max(s1, s2))
combined_score = (traj_auc/(traj_auc+vae_auc))*result['traj_score_s'] + (vae_auc/(traj_auc+vae_auc))*result['recon_score_s']

auc_roc = roc_auc_score(result['gt'],combined_score)

precision, recall, thresholds = precision_recall_curve(result['gt'],combined_score)

auc_pr = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'],combined_score, pos_label=1)
fnr = 1 - tpr
eer_th = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc))
print('AUC PR: {}'.format(auc_pr))
print('EER: {}'.format(eer))
print('EER TH: {}'.format(eer_th))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
    
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))

if plot_eer:
    
    plt.plot(threshold, fpr, label='FPR', color='#64b19c')
    plt.plot(threshold, fnr, label='FNR', color='#f183d2')
    plt.plot([eer_th, eer_th], [0, 1], 'k--', label=f'EER ({eer:.3f})', linewidth=2, color='#e80c4b')
    plt.plot([threshold_at_target_fnr, threshold_at_target_fnr], [0, 1], 'k--', label=f'10ER ({fpr_at_target_fnr:.3f})', linewidth=2, color='#095c49')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('Error Rate on CHAD')
    plt.legend()
    plt.grid(True)
    plt.savefig('test.png')
    # plt.savefig('Shang_EER.pdf', format='pdf')
    
    

print ("*********************Pose VAE************************")
auc_roc_recon  = roc_auc_score(result['gt'],result['recon_score'])

precision, recall, thresholds = precision_recall_curve(result['gt'],result['recon_score'])

auc_pr_recon = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'],result['recon_score'], pos_label=1)
fnr = 1 - tpr
eer_th_recon = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer_recon = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc_recon))
print('AUC PR: {}'.format(auc_pr_recon))
print('EER: {}'.format(eer_recon))
print('EER TH: {}'.format(eer_th_recon))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))

print ("*********************Trajectory MSE************************")
auc_roc_traj  = roc_auc_score(result['gt'],result['traj_score'])

precision, recall, thresholds = precision_recall_curve(result['gt'],result['traj_score'])

auc_pr_traj = auc(recall, precision)

fpr, tpr, threshold = roc_curve(result['gt'],result['traj_score'], pos_label=1)
fnr = 1 - tpr
eer_th_traj = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
eer_traj = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print('AUC ROC: {}'.format(auc_roc_traj))
print('AUC PR: {}'.format(auc_pr_traj))
print('EER: {}'.format(eer_traj))
print('EER TH: {}'.format(eer_th_traj))

idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

# Get the corresponding threshold and FPR values
threshold_at_target_fnr = threshold[idx_closest_fnr]
fpr_at_target_fnr = fpr [idx_closest_fnr]
print('10ER: {}'.format(fpr_at_target_fnr))
print('10ER TH: {}'.format(threshold_at_target_fnr))
print("Done!")