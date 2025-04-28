import numpy as np

def get_max_preds(heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_keypoints, depth, height, width])
    preds: [batch_size, num_keypoints, [depth, height, width]]
    maxvals: [batch_size, num_keypoints, maxval]
    '''
    assert isinstance(heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 5, 'batch_images should be 4-ndim'
    batch_size = heatmaps.shape[0]
    num_keypoints = heatmaps.shape[1]
    hw = heatmaps.shape[3]*heatmaps.shape[4]
    w = heatmaps.shape[4]
    heatmaps_reshaped = heatmaps.reshape((batch_size, num_keypoints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_keypoints, 1))
    idx = idx.reshape((batch_size, num_keypoints, 1))

    preds = np.tile(idx, (1, 1, 3)).astype(np.int)
    preds[:, :, 0] = np.floor(preds[:, :, 0] / hw)
    preds[:, :, 1] = np.floor((preds[:, :, 1] % hw) / w)
    preds[:, :, 2] = np.floor((preds[:, :, 2] % hw) % w)

    return preds, maxvals

def cal_ap(oks,thresholds):
    size = oks.shape[0]
    acc = np.zeros(len(thresholds) + 1)
    if size == 0:
        pass
    else:
        for i in range(len(thresholds)):
            sum_val = np.sum(oks>thresholds[i])
            ap = sum_val / size
            acc[i + 1] = ap
        acc[0] = np.sum(acc[1:]) / len(thresholds) # mAP=acc[0], AP50=acc[1], AP75=acc[6]

    return acc


def accuracy(output, target, target_weight, edge_radius, edge_volume, mode):

    batch_size = output.shape[0]
    num_keypoints = output.shape[1]
    pred, _ = get_max_preds(output)
    target, _ = get_max_preds(target)
    distances = np.linalg.norm(pred - target, axis=-1) # (batch_size, num_keypoints)
    exp_vector = np.exp(-(distances**2) / (2*edge_volume*(0.2)**2)) # (batch_size, num_keypoints)
    exp_dis = np.exp(-(distances**2)) # (batch_size, num_keypoints)
    target_weight = target_weight.squeeze(axis=-1) # (batch_size, num_keypoints)
    b = (1 - target_weight).astype(bool)
    exp_vector[b] = -1
    exp_dis[b] = 0
    
    vis_num = np.sum(target_weight)
    mean_exp_dis = np.sum(exp_dis) / vis_num

    oks_list = []
    idxs = []

    for i in range(batch_size):
        not_neg = np.not_equal(exp_vector[i], -1)
        num = np.sum(not_neg)
        if num > 0:
            val = np.sum(exp_vector[i] * not_neg)
            oks_list.append(val/num)
            idxs.append(i)
            
        else:
            continue

    oks = np.array(oks_list)[:,np.newaxis] # [new_bs, 1]

    thresholds = np.arange(0.5,1,0.05)  # 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, ...
    acc = cal_ap(oks,thresholds)

    vis_num_k1 = np.sum(target_weight[:,0])
    mean_exp_dis_k1 = np.sum(exp_dis[:,0]) / vis_num_k1

    vis_num_k2 = np.sum(target_weight[:,1])
    mean_exp_dis_k2 = np.sum(exp_dis[:,1]) / vis_num_k2

    if mode == 'train':
        metric = {
        'ap': acc[0],
        'val_batch_size': oks.shape[0],
        'exp_dis': mean_exp_dis,
        'vis_keypoint': vis_num,
        'exp_dis_kp1': mean_exp_dis_k1,
        'vis_kp1': vis_num_k1,
        'exp_dis_kp2': mean_exp_dis_k2,
        'vis_kp2': vis_num_k2
        }

    else:
        oks_k1 = exp_vector[:,0][exp_vector[:,0]!=-1]
        oks_k2 = exp_vector[:,1][exp_vector[:,1]!=-1]

        acc_k1 = cal_ap(oks_k1,thresholds)
        acc_k2 = cal_ap(oks_k2,thresholds)

        edge_radius = edge_radius[idxs]

        # comupte ap scale
        oks_small = oks[edge_radius<=2]
        ap_small = cal_ap(oks_small,thresholds)

        oks_medium = oks[((edge_radius>2) * (edge_radius<=3)).astype(bool)]
        ap_medium = cal_ap(oks_medium,thresholds)

        oks_large = oks[edge_radius>3]
        ap_large = cal_ap(oks_large,thresholds)

        metric = {
        'ap': acc[0],
        'ap_50': acc[1],
        'ap_75': acc[6],
        'ap_s': ap_small[0],
        's_num': oks_small.shape[0],
        'ap_m': ap_medium[0],
        'm_num': oks_medium.shape[0],
        'ap_l': ap_large[0],
        'l_num': oks_large.shape[0],
        'val_batch_size': oks.shape[0],
        'exp_dis': mean_exp_dis,
        'vis_keypoint': vis_num,
        'ap_kp1': acc_k1[0],
        'num_kp1': oks_k1.shape[0],
        'ap_kp2': acc_k2[0],
        'num_kp2': oks_k2.shape[0],
        'exp_dis_kp1': mean_exp_dis_k1,
        'vis_kp1': vis_num_k1,
        'exp_dis_kp2': mean_exp_dis_k2,
        'vis_kp2': vis_num_k2
        }

    return metric