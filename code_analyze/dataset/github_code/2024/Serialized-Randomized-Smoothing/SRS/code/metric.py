import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="", type=str)

args = parser.parse_args()

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        self.radius = np.array(df["radius"] * df["correct"])

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        self.radius = np.array(df["radius"] * df["correct"])
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


class Sample(object):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.labels = self.read_sample()

    def read_sample(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([df.iloc[:, 0]]).reshape(-1)


def metric1(baseline, line):
    metric = 0
    radii = np.arange(0, 1.5 + 0.1, 0.1)
    diff = baseline.quantity.at_radii(radii) - line.quantity.at_radii(radii)
    for idx in range(len(radii)):
        metric += max(diff[idx], 0)
    metric /= len(radii)

    return metric


# def metric2(baseline, line):
#     metric = 0
#     base_radii = baseline.quantity.radius
#     line_radii = line.quantity.radius
#     for base_radius, radius in zip(base_radii, line_radii):
#         # metric += abs(base_radius-radius) / base_radius if base_radius > 0 else 0
#         metric += max(base_radius-radius, 0) / base_radius if base_radius > 0 else 0
#     metric /= len(base_radii)
#
#     return metric

def metric2(baseline, line):
    metric = torch.zeros((500))
    base_radii = baseline.quantity.radius
    line_radii = line.quantity.radius
    for idx, (base_radius, radius) in enumerate(zip(base_radii, line_radii)):
        # metric += abs(base_radius-radius) / base_radius if base_radius > 0 else 0
        # metric[idx] = max(base_radius-radius, 0) / base_radius if base_radius > 0 else 0
        metric[idx] = abs(base_radius-radius) / 1.35 if base_radius > 0 else 0
    # interval = torch.linspace(0, 1, 16)
    # l_interval = interval[:-1]
    # u_interval = interval[1:]
    # for idx, (l, u) in enumerate(zip(l_interval, u_interval)):
    #     in_bin = metric.ge(l) * metric.le(u)
    #     bin_count = in_bin.sum()
    #     count[idx] = bin_count/500

    return metric


# def metric3(base_sample, sample):
#     base_labels = base_sample.labels
#     labels = sample.labels
#     metric = 0
#     for base_label, label in zip(base_labels, labels):
#         if base_label != label:
#             metric += 1
#     metric /= len(base_labels)
#     print(len(base_labels))
#     return metric

def metric3(base_sample, sample):
    base_labels = base_sample.labels
    labels = sample.labels
    metric = torch.zeros((500))
    for idx, (base_label, label) in enumerate(zip(base_labels, labels)):
        num = idx // 100
        if base_label != label:
            metric[num] += 1
    metric = metric/100
    metric[-1] = metric[-1] *100 /99
    return metric


if __name__ == '__main__':
    """For gap"""
    file_name = "analysis/plots/gap_LARGE_5.txt"
    gap = []
    with open(file_name, "r") as file:
        lines = file.readlines()
        for line in lines:
            gap.append(float(line))
    gap = np.array(gap)
    for idx in range(len(gap)):
        if gap[idx] > 0.2:
            gap[idx] = 0.1
    out_name = file_name[:-3] + "pdf"
    plt.figure()
    plt.xlabel('gap', fontsize=22)
    plt.ylabel('normalized frequency', fontsize=22)
    plt.tick_params(labelsize=20)
    plt.xlim((-0.1, 0.5))
    plt.ylim((0, 20))
    plt.hist(gap, range=(0, 0.3), bins=10, density=True, label=[f'SRS-{file_name.split("/")[2][4:9]}-A{int(file_name[-5])-2}'], edgecolor='k')
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()

    # path = args.folder
    # if path[-1] == '1':
    #     path1 = path[:-1]+'3'
    # else:
    #     path1 = path[:-1]+'1'
    # base_line = Line(ApproximateAccuracy(f"{path}/results1.txt"), "")
    # clean_line = Line(ApproximateAccuracy(f"{path}/results2.txt"), "")
    # step_line = Line(ApproximateAccuracy(f"{path1}/results3.txt"), "")
    #
    # base_sample = Sample(f"{path}/sample1.txt")
    # clean_sample = Sample(f"{path}/sample2.txt")
    # step_sample = Sample(f"{path}/sample3.txt")
    #
    # clean_metric2 = metric2(base_line, clean_line)
    # step_metric2 = metric2(base_line, step_line)

    # split histogram
    # print(clean_metric2)
    # print(step_metric2)
    # out_name = 'analysis/metric/'+path.split('/')[1]+path.split('/')[2][-2]+path.split('/')[3][-1]
    # plt.figure()
    # plt.xlabel('RRD')
    # plt.ylabel('normalized frequency')
    # plt.ylim((0, 20))
    # plt.hist(clean_metric2, bins=20, density=True, color='orange', edgecolor = "black")
    # plt.tight_layout()
    # plt.savefig(out_name+'mdeq.pdf')
    # plt.close()
    # plt.figure()
    # plt.xlabel('RRD')
    # plt.ylabel('normalized frequency')
    # plt.ylim((0,20))
    # plt.hist(step_metric2, bins=20, density=True, color='orange', edgecolor = "black")
    # plt.tight_layout()
    # plt.savefig(out_name + 'shallow.pdf')
    # plt.close()

    # joint histogram
    # out_name = 'analysis/metric2/' + path.split('/')[1] + path.split('/')[2][-2] + path.split('/')[3][-1]
    # data = [clean_metric2, step_metric2]
    # plt.figure()
    # plt.xlabel('RRD', fontsize=22)
    # plt.ylabel('normalized frequency', fontsize=22)
    # plt.tick_params(labelsize=20)
    # plt.xlim((0, 1))
    # plt.ylim((0, 20))
    # plt.hist(data, bins=20, density=True, label=["MDEQ-"+path[-8]+path[-1], "SRS-MDEQ-"+path[-8]+path[-1]])
    # plt.legend(loc='upper right', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(out_name + 'rrd.pdf')
    # plt.close()

    # split histogram
    # clean_metric3 = metric3(base_sample, clean_sample)
    # step_metric3 = metric3(base_sample, step_sample)
    # out_name = 'analysis/metric3/' + path.split('/')[1] + path.split('/')[2][-2] + path.split('/')[3][-1]
    # plt.figure()
    # plt.xlabel('LAD')
    # plt.ylabel('normalized frequency')
    # plt.ylim((0, 20))
    # plt.hist(clean_metric3, bins=20, density=True, color='orange', edgecolor = "black")
    # plt.tight_layout()
    # plt.savefig(out_name+'mdeq.pdf')
    # plt.close()
    # plt.figure()
    # plt.xlabel('LAD')
    # plt.ylabel('normalized frequency')
    # plt.ylim((0,20))
    # plt.hist(step_metric3, bins=20, density=True, color='orange', edgecolor = "black")
    # plt.tight_layout()
    # plt.savefig(out_name + 'shallow.pdf')
    # plt.close()

    # joint histogram
    # out_name = 'analysis/metric3/' + path.split('/')[1] + path.split('/')[2][-2] + path.split('/')[3][-1]
    # data = [clean_metric3, step_metric3]
    # plt.figure()
    # plt.xlabel('LAD', fontsize=22)
    # plt.ylabel('normalized frequency', fontsize=22)
    # plt.tick_params(labelsize=20)
    # plt.ylim((0, 20))
    # plt.hist(data, bins=20, density=True, label=["MDEQ-"+path[-8]+path[-1], "SRS-MDEQ-"+path[-8]+path[-1]])
    # plt.legend(loc='upper right', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(out_name + 'lad.pdf')
    # plt.close()

    # out_name = 'analysis/metric3/' + path.split('/')[1] + path.split('/')[2][-2] + path.split('/')[3][-1]
    # plt.figure()
    # plt.xlabel(r'$\overline{p_m}$', fontsize=22)
    # plt.ylabel('normalized frequency', fontsize=22)
    # plt.tick_params(labelsize=20)
    # plt.ylim((0, 20))
    # plt.xlim((0, 1))
    # plt.hist(step_metric3, bins=10, density=True, label=["SRS-MDEQ-" + path[-8] + str(int(path[-1])-2)], edgecolor='k')
    # plt.legend(loc='upper right', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(out_name + 'pm.pdf')
    # plt.close()

    # number of metrics
    # clean_metric1 = metric1(base_line, clean_line)
    # clean_metric2 = metric2(base_line, clean_line)
    # clean_metric3 = metric3(base_sample, clean_sample)
    #
    # step_metric1 = metric1(base_line, step_line)
    # step_metric2 = metric2(base_line, step_line)
    # step_metric3 = metric3(base_sample, step_sample)

    # print(f"clean_metric1: {clean_metric1:.4f}")
    # print(f"clean_metric2: {clean_metric2:.4f}")
    # print(f"clean_metric3: {clean_metric3:.4f}")
    # print(f"step_metric1: {step_metric1:.4f}")
    # print(f"step_metric2: {step_metric2:.4f}")
    # print(f"step_metric3: {step_metric3:.4f}")

    # radii = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    # acc1 = base_line.quantity.at_radii(radii)
    # acc2 = clean_line.quantity.at_radii(radii)
    # acc3 = step_line.quantity.at_radii(radii)
    #
    # print(f"acc1: {acc1}")
    # print(f"acc2: {acc2}")
    # print(f"acc3: {acc3}")
