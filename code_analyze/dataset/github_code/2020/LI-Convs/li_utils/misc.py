import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import sys
import os
import tensorflow.compat.v1 as tf
import cv2

# plot one image or a list of images
def imshow(img,normalize_range=None,cmap='Oranges',clip=True):

    if type(img) is np.ndarray:
        img = [img]

    for this_img in img:
        if normalize_range is not None:
            norm=matplotlib.colors.Normalize(
                vmin=np.min(normalize_range),
                vmax=np.max(normalize_range),
                clip=clip)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            this_img = mapper.to_rgba(this_img)
        else:
            this_img = np.squeeze(this_img).astype(np.uint8)

        plt.figure()
        plt.imshow(this_img)
    plt.show()



# print on the same line (overwriting previous contents)
def print_overwrite(print_str):
   print(print_str, end='')
   sys.stdout.flush()
   print('\r', end='')


# add a simple value and let summary write records it
def add_simple_value_to_summary_writer(summary_writer, step, tag, value):
    new_sum = tf.Summary()
    new_sum.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(new_sum, step)


def _fast_hist(label_pred, label_true, num_classes,
               ignore_label=None,
               dtype=np.uint64):

    if ignore_label is not None:   # set ignore labels to be 255
        label_true[label_true==ignore_label] = 255

        if ignore_label==0:  # convert label from [1,class+1) to [0,class) if ignore label is 0
            label_true = label_true - 1    # decrease groudtruth by one
            label_true[label_true==244] = 255  # set ignore label to be 255
            label_pred = label_pred - 1   # decrease predictions by 1

    # ignore any pixel larger than num_classes
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes*label_true[mask].astype(int)+label_pred[mask],
        minlength=num_classes ** 2).reshape(num_classes, num_classes).astype(dtype)

    return hist


def get_metrics_from_hist(hist):

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc



def visual_label(label, col_set, num_of_classes):
    label_vis = np.zeros(label.shape[0:2]+(3,), dtype=np.uint8)
    for class_id in range(num_of_classes):
        if class_id > 0:  # no need to fill background class
            label_vis[label==class_id] = np.asarray(
                [c*255 for c in col_set[class_id]], dtype=np.uint8)
    return label_vis


def save_test_plot(image, pr, lb, num_of_classes, save_path):
    import seaborn as sns
    col_set = sns.color_palette("hls", num_of_classes)
    pr_plot = visual_label(pr, col_set, num_of_classes)
    lb_plot = visual_label(lb, col_set, num_of_classes)
    final_plot = np.concatenate((image, pr_plot, lb_plot), axis=1)
    _= cv2.imwrite(save_path, final_plot)


def save_test_plot_seperate(image, pr, lb, num_of_classes, save_dir, image_name):
    import seaborn as sns
    col_set = sns.color_palette("hls", num_of_classes)
    pr_plot = visual_label(pr, col_set, num_of_classes)
    lb_plot = visual_label(lb, col_set, num_of_classes)
    # final_plot = np.concatenate((image, pr_plot, lb_plot), axis=1)
    _= cv2.imwrite(
        os.path.join(save_dir, image_name+'_ori.png'),
        image)
    _= cv2.imwrite(
        os.path.join(save_dir, image_name+'_pred.png'),
        pr_plot)
    _= cv2.imwrite(
        os.path.join(save_dir, image_name+'_label.png'),
        lb_plot)


def save_test_result_to_txt(miou, acc, save_path):
    with open(save_path, 'w') as wrt:
        wrt.write('mIoU: {:.2f}%, acc: {:.2f}%\n'.format(
            miou*100.0, acc*100.0))


def save_model_flops_and_params_to_txt(
        save_path,
        model_flops,
        model_params,
        decimal='.2f'):
    with open(save_path, 'w') as wrt:
        wrt.write('FLOPs: {} or {}\n'.format(
            model_flops, human_format(model_flops, decimal=decimal)))
        wrt.write('Params: {} or {}\n'.format(
            model_params, human_format(model_params, decimal=decimal)))


def save_val_result_to_txt(checkpoint_path, miou_record, save_path):
    with open(save_path, 'w') as wrt:
        wrt.write('Checkpoint Name, Val mIoU\n')
        for miou, ck_path in zip(miou_record, checkpoint_path):
            ck_name = os.path.split(ck_path)[1]
            wrt.write('{}, {:.2f}%\n'.format(ck_name, miou*100.0))

# convert a large number to human readable string
def human_format(num, decimal='.2f'):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    format_str = '%'+decimal+'%s'
    return format_str % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])


# read the params and flops from the text
def get_params_flops_from_text(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    cmt = ', '
    for line in content:
        cmt = cmt+line+', '
    return cmt.strip()