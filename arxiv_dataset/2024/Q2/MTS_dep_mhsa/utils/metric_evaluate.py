from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
import torch 
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision
from einops import rearrange
import numpy as np



def calc_metrics(state, model, data_loader, device):
    f1_metrics = []
    auc_metrics = []
    acc_metrics = []
    model.load_state_dict(state)
    # print(model)
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    stream = tqdm(data_loader)
    # y_abnormal = 0
    # y_normal = 0
    for batch_idx, (videos, y_true, patient_id, _) in enumerate(stream):
        videos = videos.to(device=device).float()
        y_true_list.extend(y_true.flatten().cpu().tolist())
        y_true = y_true.to(device=device).long()
        # y_abnormal += y_true.sum().cpu()
        # y_normal += torch.Tensor([videos.size()[0] - y_true.sum()]).cpu()
        
        with torch.no_grad():
            reg_out = model(videos)
            y_pred_list.extend(torch.argmax(F.softmax(reg_out.cpu().float(), dim=1),dim=1).tolist())
            y_score_list.extend([x[1] for x in F.softmax(reg_out.cpu().float(), dim=1).detach().numpy()])
            # print(y_score_list)

    f1_metrics.append(f1_score(y_true_list, y_pred_list))
    auc_metrics.append(roc_auc_score(y_true_list, y_score_list))
    acc_metrics.append(accuracy_score(y_true_list, y_pred_list))
    return f1_metrics, auc_metrics, acc_metrics


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"value": 0, "count": 0, "avg": 0})

    def update(self, metric_name, value, cnt):
        metric = self.metrics[metric_name]
        metric["value"] += value * cnt
        metric["count"] += cnt
        metric["avg"] = metric["value"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
        

def ensure_no_patient_split(train_ids, valid_ids, dataset):
    overlapping_ids_in_test = []
    patient_ids = dataset.patient_id_by_chunk
    patient_ids_train = set([patient_ids[i] for i in train_ids])
    patient_ids_valid = set([patient_ids[i] for i in valid_ids])
    overlapping_ids = patient_ids_train.intersection(patient_ids_valid)
    if len(overlapping_ids) == 0:
        return train_ids, valid_ids

    for overlapping_id in overlapping_ids:
        if overlapping_id not in overlapping_ids_in_test:
            indices_to_move = [ind for ind in train_ids if patient_ids[ind] == overlapping_id]
            valid_ids = np.append(valid_ids, indices_to_move).flatten()
            train_ids = np.delete(train_ids, np.searchsorted(train_ids, indices_to_move))
            overlapping_ids_in_test.append(overlapping_id)
        else:
            indices_to_move = [ind for ind in valid_ids if patient_ids[ind] == overlapping_id]
            train_ids = np.append(train_ids, indices_to_move).flatten()
            valid_ids = np.delete(valid_ids, np.searchsorted(valid_ids, indices_to_move))
    return list(sorted(train_ids)), list(sorted(valid_ids))

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0)) # C H W -> H, W, C
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title("\n".join(title))
    plt.pause(0.001)  # Pau
    plt.show()

def plot_first_frame(inputs, patient_ids):
    # Grab some of the training data to visualize
    # Now we construct a grid from batch
    inputs = inputs[:, :, 0, :, :] # B C D H W -> B, C, H, W
    print(inputs.size()) # B, C, H, W
    out = torchvision.utils.make_grid(inputs)
    imshow(out, patient_ids)


def plot_first_person(inputs):
    # Grab some of the training data to visualize
    # Now we construct a grid from
    plt.ion()
    plt.figure()
    out = rearrange(inputs[0], "c d h w -> d h w c")
    for i in range(11):
        plt.imshow(out[i], cmap="gray")
        plt.pause(0.4)

    plt.ioff()
    plt.show()


def calculate_metrics(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred )
    f1_ = f1_score( y_true, y_pred)
    # print(f"F1: {f1_:.2f}")
    # print(f"Acc: {acc:.4f}")
    return {'f1':f1_, 'acc':acc}