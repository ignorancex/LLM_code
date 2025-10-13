import numpy as np
import pandas as pd
import json
import pywt
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [
    i for i in range(6, 32, CSI_VAID_SUBCARRIER_INTERVAL)
]  # 26  red
csi_vaid_subcarrier_index += [
    i for i in range(33, 59, CSI_VAID_SUBCARRIER_INTERVAL)
]  # 26  green
CSI_DATA_LLFT_COLUMNS = len(csi_vaid_subcarrier_index)
DATA_COLUMNS_NAMES = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]


def raw2csi(csi_raw_data):
    CSI_DATA_INDEX = len(csi_raw_data)
    csi_vaid_subcarrier_len = CSI_DATA_LLFT_COLUMNS
    csi_data_array = np.zeros(
        [CSI_DATA_INDEX, csi_vaid_subcarrier_len], dtype=np.complex64
    )

    # 将CSI_DATA整数两两组合恢复成复数形式
    for row in range(CSI_DATA_INDEX):
        for i in range(csi_vaid_subcarrier_len):
            csi_data_array[row][i] = complex(
                csi_raw_data[row, csi_vaid_subcarrier_index[i] * 2 + 1],
                csi_raw_data[row, csi_vaid_subcarrier_index[i] * 2],
            )

    return csi_data_array


def get_raw_csi_data(csifile):
    df = pd.read_csv(csifile)
    df.columns = DATA_COLUMNS_NAMES
    raw_csi_json = df["data"].values
    raw_timestamp = df["timestamp"].values

    raw_csi_data = []
    for i in range(len(raw_csi_json)):
        raw_csi_data.append(json.loads(raw_csi_json[i]))

    csi_data_array = raw2csi(np.array(raw_csi_data))
    return csi_data_array, raw_timestamp


def remove_outliers(data):
    """去除异常值"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    threshold = 3 * mad  # 可根据需求调整阈值

    cleaned_data = data.copy()

    for i in range(len(data)):
        if np.abs(data[i] - median) > threshold:
            cleaned_data[i] = median

    return cleaned_data


def denoise_with_dwt(data):
    # 执行小波去噪
    threshold = np.sqrt(2 * np.log(len(data)))  # 设置阈值
    coefficients = pywt.wavedec(data, "db4", level=6)  # 将信号分解为多级小波系数
    coefficients[1:] = (
        pywt.threshold(i, threshold, mode="soft") for i in coefficients[1:]
    )  # 对除最低频系数外的其他系数进行软阈值处理
    reconstructed_signal = pywt.waverec(coefficients, "db4")  # 重构信号
    if len(data) % 2 == 1:
        return reconstructed_signal[:-1]
    else:
        return reconstructed_signal


def preprocessing_along_subcarrier(data):
    """对每个子载波进行预处理"""
    data = np.apply_along_axis(remove_outliers, axis=0, arr=data)
    data = np.apply_along_axis(denoise_with_dwt, axis=0, arr=data)
    return data


# 计算变异系数
cv = lambda x: np.std(x) / np.mean(x)


if __name__ == "__main__":

    room_id = 2
    # 导入数据
    csi_ampl = np.loadtxt("data/room_{}_csi_ampl.txt".format(room_id))
    ground_truth = np.loadtxt("data/room_{}_truth.txt".format(room_id))
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.size'] = 16


    # 特征提取：对每个窗口数据计算变异系数
    WINDOW_SIZE = 50
    feature_vector = []
    gt_vector = []
    step = 0
    while step + WINDOW_SIZE < len(csi_ampl):
        window_csi = csi_ampl[step : step + WINDOW_SIZE, :]
        window_csi = preprocessing_along_subcarrier(window_csi)
        step += WINDOW_SIZE
        feature_vector.append(cv(window_csi))
        gt_vector.append(np.max(ground_truth[step : step + WINDOW_SIZE]))

    THRESHOLD = 0.065  # 动作检测阈值 硬件不同或者处理方法改变则需要调整
    MOTION_SIZE = 10  # 动作检测窗口
    MOTION_TIMES = 2  # 动作检测次数
    # 动作检测窗口内检测到的动作次数若小于2次判定无人，大于等于判定有人；防止出现异常误判有人

    svr_vector = []
    move_result = []
    detection_result = []
    for i in range(1, len(feature_vector)):
        svr = feature_vector[i] / feature_vector[i - 1]  # 特征值比
        svr_vector.append(svr)

        if abs(svr - 1) < THRESHOLD:
            move_result.append(0)  # 无运动
        else:
            move_result.append(1)  # 有运动

        if np.sum(move_result[-MOTION_SIZE:]) < 2:
            detection_result.append(0)
        else:
            detection_result.append(1)

    plt.figure(figsize=(8, 5), dpi=200)

    # 变异系数相邻窗口比值与标准阈值
    plt.subplot(211)
    plt.plot(svr_vector)  #
    plt.xlabel("Packet Index")
    plt.ylabel("Signal Value")
    plt.plot([1 + THRESHOLD] * len(svr_vector), color="r")
    plt.plot([1 - THRESHOLD] * len(svr_vector), color="r")
    plt.ylim(0.5, 1.5)


    # 人员存在状态检测结果与真值
    gt_vector = gt_vector[1:]
    gt_vector = [int(element) for element in gt_vector]
    plt.subplot(212)
    plt.plot(gt_vector, "-", label="Ground Truth", color="#2878b5")
    plt.plot(detection_result, "--", label="Detection Result", color="r")
    plt.xlabel("Packet Index")
    plt.ylabel("Status")
    plt.legend()
    plt.tight_layout()
    plt.savefig('room_{}_result.pdf'.format(room_id), format='pdf', bbox_inches='tight')
    plt.show()
    # 变异系数计算值
    # plt.subplot(221)
    # plt.plot(feature_vector)
    # plt.title("CV Feature")

    # # 变异系数相邻窗口比值与标准阈值
    # plt.subplot(222)
    # plt.plot(svr_vector)  #
    # plt.plot([1 + THRESHOLD] * len(svr_vector), color="r")
    # plt.plot([1 - THRESHOLD] * len(svr_vector), color="r")
    # plt.ylim(0.5, 1.5)
    # plt.title("The Ratio of Feature")

    # # 动作检测结果
    # plt.subplot(223)
    # plt.plot(move_result, ".")
    # plt.title("Motion Detection")

    # # 人员存在状态检测结果与真值
    # gt_vector = gt_vector[1:]
    # gt_vector = [int(element) for element in gt_vector]
    # plt.subplot(224)
    # plt.plot(gt_vector, "-", label="Ground Truth", color="#2878b5")
    # plt.plot(detection_result, "--", label="Detection Result", color="r")
    # plt.title("Human Detection")
    # plt.legend()
    # plt.tight_layout()
    # # plt.savefig('room_{}_result.pdf'.format(room_id), format='pdf', bbox_inches='tight')
    # plt.show()
