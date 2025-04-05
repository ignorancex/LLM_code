import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.interactive(False)
import threading
import os
import pandas as pd

from plugins.csv_collect_and_publish import PluginCSVCollectAndPublish
import preprocessing.init_buffer as buf


class DataFeeder(threading.Thread):
    def __init__(self, plugin):
        threading.Thread.__init__(self)
        self.is_run = True
        self.plugin = plugin

    def run(self):
        position = 0
        while self.is_run:
            index_buffer = 0
            for i in range(0, plugin.number_of_channels):
                buf.ring_buffers[index_buffer].append(float(str(df.ix[:, index_buffer][position])))
                index_buffer += 1
                position += 1
                if position == df.shape[0]:
                    self.is_run = False
                    break
                threading._sleep(self.plugin.sampling_time)


project_dir = "/home/runge/openbci/OpenBCI_Python"
plugin = PluginCSVCollectAndPublish()
plugin.activate()
df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
df = df[plugin.channel_vector].dropna(axis=0)

data_feeder = DataFeeder(plugin)
data_feeder.start()
plugin.main_thread.start()
print ("running as background process")

# print ("process is done")
