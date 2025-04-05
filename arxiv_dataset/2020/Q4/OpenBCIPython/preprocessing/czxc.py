# import time, random
# import math
# from collections import deque
#
# import librosa
# import matplotlib.animation as animation
# from matplotlib import pyplot as plt
# import numpy as np
# start = time.time()
#
#
# class RealtimePlot:
#     def __init__(self, axes, max_entries=100):
#         self.axis_x = deque(maxlen=max_entries)
#         self.axis_y = deque(maxlen=max_entries)
#         self.axes = axes
#         self.max_entries = max_entries
#
#         self.lineplot, = axes.plot([], [], "ro-")
#         self.axes.set_autoscaley_on(True)
#
#     def add(self, x, y):
#         self.axis_x.extend(x)
#         self.axis_y.extend(y)
#         self.lineplot.set_data(self.axis_x, self.axis_y)
#         self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
#         self.axes.relim()
#         self.axes.autoscale_view()  # rescale the y-axis
#
#     def animate(self, figure, callback, interval=50):
#         def wrapper(frame_index):
#             self.add(*callback(frame_index))
#             self.axes.relim()
#             self.axes.autoscale_view()  # rescale the y-axis
#             return self.lineplot
#         animation.FuncAnimation(figure, wrapper, interval=interval)
#
#
# def main():
#     fig, axes = plt.subplots()
#     display = RealtimePlot(axes)
#     display.animate(fig, lambda frame_index: (time.time() - start, random.random() * 100))
#     while True:
#         ydata = [random.randint(0, i) * i for i in range(0, 20)]
#         # ydata = librosa.amplitude_to_db(librosa.stft(ydata), ref=np.max)
#         xdata = [i for i in range(0, len(ydata))]
#         display.add(xdata, ydata)
#         plt.pause(0.001)
#
#
# if __name__ == "__main__": main()


import random
import time

from matplotlib import pyplot as plt
from matplotlib import animation


class RegrMagic(object):
    """Mock for function Regr_magic()
    """
    def __init__(self):
        self.x = 0
    def __call__(self):
        time.sleep(random.random())
        self.x += 1
        return self.x, random.random()

regr_magic = RegrMagic()

def frames():
    while True:
        yield regr_magic()

fig = plt.figure()

x = []
y = []
def animate(args):
    x.append(args[0])
    y.append(args[1])
    return plt.plot(x, y, color='g')


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000)
plt.show()
