import json
import csv
import sys
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
class ModelComputeTemplate():
    '''
    template for dataset_result_customized.
    '''
    def __init__(self, model, empty_compute_dict, **kwargs):
        self.model = model
        self.compute_dict = empty_compute_dict
    def result(self, data_list):
        # decide your comutation here
        results = self.model(data_list, training=False)
        results = results[0].numpy()#get the softmax result in default
        return {"result": results}

def plt_bar(value_list, max_index, save_path, x_label=None, y_label=None, color="blue"):
    plt.clf()
    index_list = [i for i in range(max_index)]
    statistics_list = [0 for i in range(max_index)]

    for value in value_list:
        if value<max_index:
            statistics_list[value] += 1
    if not isinstance(x_label, type(None)):
        plt.xlabel(x_label)
    if not isinstance(y_label, type(None)):
        plt.ylabel(y_label)
    plt.bar(index_list, statistics_list, color=color)
    plt.savefig(save_path)
class PltScatterPlot():
    def __init__(self, x_range=None, y_range=None, x_label=None, y_label=None):
        self.fig, self.axs = plt.subplots()
        if not isinstance(x_range, type(None)):
            self.axs.set_xlim(*x_range)
        if not isinstance(y_range, type(None)):
            self.axs.set_ylim(*y_range)
        if not isinstance(x_label, type(None)):
            self.axs.set_xlabel(x_label)
        if not isinstance(y_label, type(None)):
            self.axs.set_ylabel(y_label)
    def add_data(self, x_value_list, y_value_list, color="blue", alpha_div=500):
        # alpha = alpha_mul/len(x_value_list)
        alpha = 1/alpha_div
        self.axs.scatter(x_value_list, y_value_list, color=color, alpha=alpha)
    def save_plot(self, save_path, add_legend=False):
        if add_legend:
            self.axs.legend()
        self.fig.savefig(save_path)
class PltQuantityPlot():
    def __init__(self, bin_width=None, x_range=None, y_range=None, x_label=None, y_label=None):
        self.fig, self.axs = plt.subplots()
        if not isinstance(x_range, type(None)):
            self.axs.set_xlim(*x_range)
        if not isinstance(y_range, type(None)):
            self.axs.set_ylim(*y_range)
        if not isinstance(x_label, type(None)):
            self.axs.set_xlabel(x_label)
        if not isinstance(y_label, type(None)):
            self.axs.set_ylabel(y_label)
        self.bin_width = bin_width
    def add_data(self, value_list, normalized=False, cumsum=False, color="blue", alpha=1, label=None):
        value_width = np.max(value_list)-np.min(value_list)
        if isinstance(self.bin_width, type(None)):
            self.bin_width = value_width/40
        bin_number = (int)(0.5+value_width/self.bin_width)
        values, base = np.histogram(value_list, bins=bin_number)
        if normalized:
            values = values/np.sum(values)
        if cumsum:#cumulative
            values = np.cumsum(values)
        self.axs.plot(base[:-1], values, color=color, label=label, alpha=alpha)
    def save(self, save_path, add_legend=False):
        if add_legend:
            self.axs.legend()
        self.fig.savefig(save_path)
class PltQuantityBar():#stack chart
    def __init__(self, bin_width, x_range, y_range=None, x_label=None, y_label=None):
        self.fig, self.axs = plt.subplots()
        if not isinstance(x_range, type(None)):
            self.axs.set_xlim(*x_range)
        if not isinstance(y_range, type(None)):
            self.axs.set_ylim(*y_range)
        if not isinstance(x_label, type(None)):
            self.axs.set_xlabel(x_label)
        if not isinstance(y_label, type(None)):
            self.axs.set_ylabel(y_label)
        self.x_range = x_range
        self.bin_width = bin_width
        self.bin_number = int((x_range[1]-x_range[0])/self.bin_width)
        self.y_base = list(np.zeros(self.bin_number))
    def add_data(self, value_list, normalized=False, cumsum=False, color="blue", label=None):
        values, x_base = np.histogram(value_list, bins=self.bin_number, range=self.x_range)
        if normalized:
            values = values/np.sum(values)
        if cumsum:#cumulative
            values = np.cumsum(values)
        plt.bar(x_base[:-1], values, width=self.bin_width, bottom=self.y_base, color=color, label=label)
        self.y_base = [base_value+values[base_index] for base_index, base_value in enumerate(self.y_base)]
    def save(self, save_path, add_legend=False):
        if add_legend:
            self.axs.legend()
        self.fig.savefig(save_path)
def show_image(img,name="test", wait=True):
    img = img.astype("uint8")
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey()
    else:
        cv2.waitKey(1)
def draw_label(image, point, label, label_color, font=cv2.FONT_ITALIC,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), label_color, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
def show_result(imgs, preds, key_list, show=True, save=False, path="./save/", name="test_", wait=True):
    for img_index, img in enumerate(imgs):
        test_img = cv2.resize(img, (160,160))
        label = key_list[(int)(preds[img_index])]
        draw_label(test_img, (0, test_img.shape[1]-10), label, (0,0,0))
        if show:
            show_image(test_img, name=name, wait=wait)
        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            cv2.imwrite(path+name+str(img_index)+".jpg", test_img)
    return test_img
def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except ValueError:
            print("error in ", data_path)
            sys.exit()
    return json_data
def json_write(json_data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        try:
            f.write(json.dumps(json_data))
        except ValueError:
            print("error in ", json_path)
            sys.exit()
def pickle_load(data_path):
    with open(data_path, 'rb') as f:
        try:
            pickle_data = pickle.load(f)
        except ValueError:
            print("error in ", data_path)
            sys.exit()
    return pickle_data
def pickle_write(python_data, pickle_path):
    with open(pickle_path, 'wb') as f:
        try:
            pickle.dump(python_data, f)
        except ValueError:
            print("error in ", json_path)
            sys.exit()
class MessageHandle():
    def __init__(self, data_path):
        self.data_path = data_path
        self.str_buffer = ""
    def add(self, message, interval=", "):
        self.str_buffer = self.str_buffer+message+interval
        print(message, end=interval)
    def refresh(self):
        self.str_buffer = ""
        print(end="\r")
    def save(self):
        self.str_buffer = self.str_buffer+"\n"
        with open(self.data_path, "a") as f:
            f.write(self.str_buffer)
        print()
        self.str_buffer = ""
class TxtWriter():
    def __init__(self, data_path):
        self.txt_file = open(data_path, "w")
    def __del__(self):
        self.txt_file.close()
    def write(self, data):#change line automatically
        self.txt_file.write(data)
class CsvReader():
    def __init__(self, data_path, delimiter=","):
        self.csvfile = open(data_path, newline='')
        self.reader = csv.reader(self.csvfile, delimiter=delimiter)
    def __del__(self):
        self.csvfile.close()
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.reader)

class CsvWriter():
    def __init__(self, data_path, delimiter=","):
        self.csvfile = open(data_path, 'w', newline='')
        self.writer = csv.writer(self.csvfile, delimiter=delimiter)
    def __del__(self):
        self.csvfile.close()
    def write(self, raw_data):#write a line
        self.writer.writerow(raw_data)