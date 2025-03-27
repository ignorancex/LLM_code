# utils.py

def metric_to_label(metric):
    if metric.lower() == 'ssim':
        return 'SSIM'
    elif metric.lower() == 'lpips':
        return 'LPIPS'
    elif 'auroc' in metric.lower():
        if 'unattacked' in metric.lower():
            return 'AUROC (Unattacked)'
        else:
            return 'AUROC'
    else:
        words = metric.replace('_', ' ').split()
        return ' '.join([word.capitalize() for word in words])


def target_to_label(target):
    target_map = {
        'hidden': 'HiDDeN',
        'mbrs': 'MBRS',
        'stega': 'StegaStamp'
    }
    return target_map.get(target.lower(), target)


def metric_to_label_custom(metric):
    if 'bitwise' in metric.lower():
        return 'Bit-wise Accuracy'
    else:
        return metric_to_label(metric)
