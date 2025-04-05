import json
import socket
import threading
import timeit
import datetime
import tensorflow as tf

from preprocessing.RingBuffer import RingBuffer
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.server import UDPServer
import preprocessing.init_buffer as buf

import plugin_interface as plugintypes

class MainThread(threading.Thread):
    def __init__(self, plugin_config, max_iteration=None):
        threading.Thread.__init__(self)
        self.is_run = True
        self.counter = 0
        self.lock = threading.Lock()
        self.plugin_config = plugin_config
        self.max_iteration = max_iteration
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.number_of_channels = int(self.plugin_config["number_of_channels"])
        self.buffer_capacity = int(self.plugin_config["buffer_capacity"])
        self.buffer_size = int(self.plugin_config["window_size"])
        buf.ring_buffers = [RingBuffer(self.buffer_size * self.buffer_capacity)
                             for i in range(0, self.number_of_channels)]
        self.tfrecords_filename = self.plugin_config["project_file_path"] \
                                  + str(self.plugin_config["processing"]["train"]["tfrecords_filename"])
        self.writer = tf.python_io.TFRecordWriter(self.tfrecords_filename)
        self.sampling_rate = int(self.plugin_config["sampling_rate"])
        self.sampling_time = 1.0 / self.sampling_rate * 1.0
        self.noisereducer_thread = NoiseReducer("main thread", self.server,
                                                self.writer, self.plugin_config)
        self.ip = str(self.plugin_config["ip"])

    def run(self):
        while self.is_run:
            if not self.noisereducer_thread.is_processing and \
                            buf.ring_buffers[0].__len__() > self.buffer_size:
                # if not plugin.noisereducer_thread.is_processing:
                print ("next process..." + str(buf.ring_buffers[0].__len__()))
                self.noisereducer_thread = NoiseReducer(str(self.counter), self.server,
                                                          self.writer, self.plugin_config)
                self.noisereducer_thread.start()
                self.noisereducer_thread.join()
                self.counter += 1
                if self.max_iteration is not None and self.counter == self.max_iteration:
                    self.is_run = False

        self.server.close()
        self.writer.close()
        print ("closing main thread...")


class PluginCSVCollectAndPublish(plugintypes.IPluginExtended):
    def init_plugin(self):
        config_file = self.project_file_path + "/config/config.json"
        with open(config_file) as config:
            self.plugin_config = json.load(config)
            self.now = datetime.datetime.now()
            self.time_stamp = '%d-%d-%d_%d-%d-%d' % (self.now.year, self.now.month, self.now.day, self.now.hour,
                                                     self.now.minute, self.now.second)
            self.train = eval(self.plugin_config["train"])
            self.number_of_channels = int(self.plugin_config["number_of_channels"])
            self.train_dir = self.project_file_path + str(self.plugin_config["train_dir"])
            self.plugin_config["project_file_path"] = self.project_file_path
            self.plugin_config["train_dir_abs_location"] = self.train_dir
            self.train_file = self.train_dir + self.time_stamp + "new_up.csv"
            self.start_time = timeit.default_timer()
            self.delim = ","
            self.verbose = eval(self.plugin_config["verbose"])
            self.test = eval(self.plugin_config["test"])
            self.kinect_angles = RingBuffer(20, dtype=list)
            self.channel_vector = list(self.plugin_config["channel_vector"])
            self.sampling_rate = int(self.plugin_config["sampling_rate"])
            self.sampling_time = 1.0 / self.sampling_rate * 1.0
            self.ip = str(self.plugin_config["ip"])
            self.port = int(self.plugin_config["port"])
            self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.main_thread = MainThread(self.plugin_config, buf.max_iteration)
            if self.train:
                self.secondary_ip = self.ip
                self.secondary_port = int(self.plugin_config["secondary_port"])
                self.receiver_port = int(self.plugin_config["receiver_port"])
                self.secondary_server = UDPServer("udp_server", self.kinect_angles, self.secondary_port
                                                  , self.receiver_port, ip=self.secondary_ip)

    def activate(self):
        self.project_file_path = "/home/runge/openbci/git/OpenBCI_Python"
        print ("stated initializing plugin...")
        self.init_plugin()
        if self.train:
            try:
                self.secondary_server.start()
                # self.secondary_server.isRun = False
                # self.secondary_server.join()
                # print ("Selecting raw UDP streaming. IP: ", self.secondary_ip, ", port: ",str(self.secondary_port))
            except:
                print ("Error while starting udp server...")
                self.secondary_server.socket.close()
        print ("plugin initialization is completed successfully.")

    def send_row_data(self, data):
        self.server.sendto(data, (self.ip, self.port))

    def deactivate(self):
        print ("Closing, CSV saved to:", self.train_file)
        self.main_thread.is_run = False
        self.server.close()
        self.main_thread.join()
        return


    def show_help(self):
        print ("Optional argument: [filename] (default: collect.csv)")
        print ("""Optional arguments: [ip [port]]
              \t ip: target IP address (default: 'localhost')
              \t port: target port (default: 12345)""")

    def get_updated_values(self):
        return self.secondary_server.get_next_point()

    def __call__(self, sample):
        t = timeit.default_timer() - self.start_time
        if self.verbose:
            print("CSV: %f | %d" % (t, sample.id))
        kinect_angles = self.get_updated_values()

        self.send_row_data(json.dumps(sample.channel_data))

        row = ''
        row += str(t)
        row += self.delim
        row += str(sample.id)
        row += self.delim
        index_buffer = 0
        for i in sample.channel_data:
            if self.test:
                if not (index_buffer >= self.number_of_channels):
                    buf.ring_buffers[index_buffer].append(float(str(i)))
            row += str(i)
            row += self.delim
            index_buffer += 1
        if self.train:
            for i in kinect_angles:
                row += str(i)
                row += self.delim
                if self.verbose:
                    print (kinect_angles)
        row[-1].replace(",", "")
        row += '\n'
        # buf.channel_data = row
        with open(self.train_file, 'a') as f:
            f.write(row)

        # self.main_thread.start()

