import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import time
from matplotlib import dates
import datetime

class DynamicUpdate():
    min_x = 0
    max_x = 10

    def on_launch(self):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o')
        self.ax.set_autoscaley_on(True)
        # self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.grid()


    def on_running(self, xdata, ydata):
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.gcf().autofmt_xdate()
        librosa.display.specshow(ydata, sr=20, y_axis='linear', x_axis='time')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __call__(self):
        self.on_launch()
        xdata = []
        ydata = []
        for x in np.arange(0,10,0.5):
            ydata = np.array([np.exp(-i**2)+10*np.exp(-(i-7)**2) for i in range(0, 128)])
            if(x%2 == 0):
                ydata = np.abs(librosa.fmt(ydata, n_fmt=64))
            else:
                ydata = librosa.amplitude_to_db(librosa.stft(ydata), ref=np.max)
            xdata = np.array([i for i in range(0, ydata.size)])
            self.on_running(xdata, ydata)
            time.sleep(1)
        return xdata, ydata

d = DynamicUpdate()
d()

# import matplotlib.pyplot as plt
# y, sr = librosa.load(librosa.util.example_audio_file())
# plt.figure(figsize=(12, 8))
# D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
# plt.subplot(4, 2, 1)
# librosa.display.specshow(D, y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Linear-frequency power spectrogram')
print "---------------"
