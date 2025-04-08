############################################################################################################################################################
############################################################################################################################################################

# Program name: my_widgets.py
# Author:  Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 22 June 2022 (last modified)
# Objective: Widgets for the 'abakus_GUI.py' file rendering
# Description: This Python3 scripts creates the graphical environment for data visualization.
#              In particular, four windows are defined; each one is devoted to a specific kind of data:
#              - 'First Panel': the upper one is the size distribution as measured by the Abakus laser sensor second-by-second in the range [1.0-10.3 um], 
#                while the lower one is the time distribution of the total numer of counts (== total number of patricles detected).
#              - 'Second Panel': the upper one is the incremental size distribution as measured by the Abakus laser sensor (that is, the size distribution of all
#                 all the particles deteced by the instrument during the measurement) in the range [1.0-10.3 um], while the lower one is the time distribution 
#                 of the total numer of counts (== total number of patricles detected).
#              - 'Third Panel': here the user can see the laser-diode voltage and the RAM-buffer voltage of the instrument; the first one is about 5.3 V while the 
#                 second one is approximately 2.8 V. Regardless of the measurement, this curves are useful to check wether the Abakus is working properly.
#              - 'Fourth Panel': only active in non-LIVE mode, here the user can visualize multiple size distributions (appropriately normalized in order to
#                 make them comparable) loaded from the corresponding files and compare them with each other.

############################################################################################################################################################
############################################################################################################################################################


import pyqtgraph as pg, numpy as np, os                                                                 # Import the required libraries
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets, uic

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

_PATH = os.path.abspath(os.path.realpath(__file__))[2:-21].replace('\\', '/')


############################################################################################################################################################
############################################################################################################################################################
# GUI window for the visualization of the size distribution measured by the Abakus laser sensor second-by-second and the time distribution of the total number of particles
# detected.

class First_Panel(QtWidgets.QWidget, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Class contructor: defiens the GUI window layout, set the icons and creates the environments for the visualization of the size and time distributions. 

    def __init__(self, parent=None):

        super().__init__(parent)
        uic.loadUi(_PATH+'subGUIs/scd_panel.ui', self)                                                  # Load the graphical interface

        self.eurocold_logo = QPixmap(_PATH+'_icon/eurocold_pixmap.png')                                 # Load the icons (EuroCold logo, DISAT and
        self.unimib_logo = QPixmap(_PATH+'_icon/unimib_pixmap.png')                                     # Unniversity of Milano Bicocca icon)
        self.disat_logo = QPixmap(_PATH+'_icon/disat_pixmap.png')

        self.lineEdit_software.setFixedWidth(100)
        self.combobox_port.setFixedWidth(100)

        self.first_window = pg.GraphicsLayoutWidget(self.scd_widget, show=True)                         # Define the environment for data visualization
        self.first_window.resize(1170, 773)
        pg.setConfigOptions(antialias=True)

        self.single_d_plt = self.first_window.addPlot(0, 0)                                             # First plot: size distribution second-by-second
        self.single_d_plt.setLabel('bottom', 'd [\u03bc'+'m]')
        self.single_d_plt.setLabel('top', ' ')
        self.single_d_plt.setLabel('right', ' ')
        self.single_d_plt.setLabel('left', '# counts')

        self.time_plt = self.first_window.addPlot(1, 0)                                                 # Second plot: time evolution of the total number of particles
        self.time_plt.setLabel('bottom', 't [s]')                                                       # detected
        self.time_plt.setLabel('top', ' ')
        self.time_plt.setLabel('right', ' ')
        self.time_plt.setLabel('left', '# counts')

        self.btn_help.setStyleSheet("QPushButton { font: bold 12px; }")
        self.acknowledgments.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.btn_update.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_correct.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_calibration.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_voltage_noise.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_serial.setStyleSheet('QPushButton { font: bold 11px; }')
        
        self.show()

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def paintEvent(self, event):

        painter = QPainter(self)                                                                        # Set the environment for icon visualization
        
        painter.drawPixmap(QPoint(800,1000), self.unimib_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(600,1003), self.eurocold_logo.scaled(172, 172, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(520,1000), self.disat_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def single_d_and_time_plot(self, color, color1, color2, time_color, time_color_avg, width, style, width1, style1, brush_levels):

        return self.first_window, self.single_d_plt, self.single_d_plt.plot(pen=pg.mkPen(color, width=width, style=style), fillLevel=0, brush=(50,50,255,100)), self.single_d_plt.plot(pen=pg.mkPen(color1, width=width, style=style), fillLevel=0, brush=brush_levels), self.single_d_plt.plot(pen=pg.mkPen(color2, width=width, style=style), fillLevel=0, brush=(0, 255, 0, 100)), self.time_plt, self.time_plt.plot(pen=pg.mkPen(time_color, width=width, style=style), fillLevel=0, brush=(255,255,0,100)), self.time_plt.plot(pen=pg.mkPen(time_color_avg, width=width1, style=style1))

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def single_d_and_time_liveplot(self, counts_color, time_color, width, style):

        return self.first_window, self.single_d_plt, self.single_d_plt.plot(pen=pg.mkPen(counts_color, width=width, style=style), fillLevel=0, brush=(50,50,255,100)), self.time_plt, self.time_plt.plot(pen=pg.mkPen(time_color, width=width, style=style), fillLevel=0, brush=(255,255,0,100))


############################################################################################################################################################
############################################################################################################################################################
# GUI window for the visualization of the incremental size distribution measured by the Abakus laser sensor and the time distribution of the total number of particles
# detected.

class Second_Panel(QtWidgets.QWidget):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Class contructor: defiens the GUI window layout, set the icons and creates the environments for the visualization of the size and time distributions.

    def __init__(self, parent=None):

        super().__init__(parent)
        uic.loadUi(_PATH+'subGUIs/scd_panel.ui', self)                                                  # Load the graphical interface

        self.eurocold_logo = QPixmap(_PATH+'_icon/eurocold_pixmap.png')                                 # Load the icons (EuroCold logo, DISAT and
        self.unimib_logo = QPixmap(_PATH+'_icon/unimib_pixmap.png')                                     # Unniversity of Milano Bicocca icon)
        self.disat_logo = QPixmap(_PATH+'_icon/disat_pixmap.png')

        self.lineEdit_software.setFixedWidth(100)
        self.combobox_port.setFixedWidth(100)

        self.second_window = pg.GraphicsLayoutWidget(self.scd_widget, show=True)                        # Define the environment for data visualization
        self.second_window.resize(1170, 773)
        pg.setConfigOptions(antialias=True)
        
        self.incremental_d_plt = self.second_window.addPlot(0, 0)                                       # First plot: incremental size distribution
        self.incremental_d_plt.setLabel('bottom', 'd [\u03bc'+'m]')
        self.incremental_d_plt.setLabel('top', ' ')
        self.incremental_d_plt.setLabel('right', ' ')
        self.incremental_d_plt.setLabel('left', '# counts')
        
        self.time_plt = self.second_window.addPlot(1, 0)                                                # Second plot: time evolution of the total number of particles
        self.time_plt.setLabel('bottom', 't [s]')                                                       # detected
        self.time_plt.setLabel('top', ' ')
        self.time_plt.setLabel('right', ' ')
        self.time_plt.setLabel('left', '# counts')

        self.btn_help.setStyleSheet("QPushButton { font: bold 12px; }")
        self.acknowledgments.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.btn_update.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_correct.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_calibration.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_voltage_noise.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_serial.setStyleSheet('QPushButton { font: bold 11px; }')
        
        self.show()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def paintEvent(self, event):

        painter = QPainter(self)                                                                        # Set the environment for icon visualization
        
        painter.drawPixmap(QPoint(800,1000), self.unimib_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(600,1003), self.eurocold_logo.scaled(172, 172, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(520,1000), self.disat_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def incremental_d_and_time_plot(self, color, color1, color2, time_color, time_color_avg, width, style, width1, style1, brush_levels):

        return self.second_window, self.incremental_d_plt, self.incremental_d_plt.plot(pen=pg.mkPen(color, width=width, style=style), fillLevel=0, brush=(50,50,255,100)), self.incremental_d_plt.plot(pen=pg.mkPen(color1, width=width, style=style), fillLevel=0, brush=brush_levels), self.incremental_d_plt.plot(pen=pg.mkPen(color2, width=width, style=style), fillLevel=0, brush=(0, 255, 0, 100)), self.time_plt, self.time_plt.plot(pen=pg.mkPen(time_color, width=width, style=style), fillLevel=0, brush=(255,255,0,100)), self.time_plt.plot(pen=pg.mkPen(time_color_avg, width=width1, style=style1))
    

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def incremental_d_and_time_liveplot(self, counts_color, time_color, width, style):

        return self.second_window, self.incremental_d_plt, self.incremental_d_plt.plot(pen=pg.mkPen(counts_color, width=width, style=style), fillLevel=0, brush=(50,50,255,100)), self.time_plt, self.time_plt.plot(pen=pg.mkPen(time_color, width=width, style=style), fillLevel=0, brush=(255,255,0,100))


############################################################################################################################################################
############################################################################################################################################################
# GUI window for the visualization of the voltages (laser diode voltage and RAM-buffer voltage) of the Abakus laser sensor.

class Third_Panel(QtWidgets.QWidget):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def __init__(self, parent=None):

        super().__init__(parent)
        uic.loadUi(_PATH+'subGUIs/volt_panel.ui', self)                                                 # Load the graphical interface

        self.eurocold_logo = QPixmap(_PATH+'_icon/eurocold_pixmap.png')                                 # Load the icons (EuroCold logo, DISAT and
        self.unimib_logo = QPixmap(_PATH+'_icon/unimib_pixmap.png')                                     # Unniversity of Milano Bicocca icon)
        self.disat_logo = QPixmap(_PATH+'_icon/disat_pixmap.png')

        self.lineEdit_software.setFixedWidth(100)
        self.combobox_port.setFixedWidth(100)

        self.third_window = pg.GraphicsLayoutWidget(self.volt_widget, show=True)                        # Define the environment for data visualization
        self.third_window.resize(1170, 773)
        pg.setConfigOptions(antialias=True)

        self.volt_plt = self.third_window.addPlot()                                                     # Plot: voltages (laser-diode and RAM-buffer voltage)
        self.volt_plt.setLabel('bottom', 't [s]')                                                       # behaviour over time
        self.volt_plt.setLabel('top', ' ')
        self.volt_plt.setLabel('right', ' ')
        self.volt_plt.setLabel('left', 'voltage [mV]')

        self.btn_help.setStyleSheet("QPushButton { font: bold 12px; }")
        self.acknowledgments.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.btn_update.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_calibration.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_voltage_noise.setStyleSheet('QPushButton { font: bold 11px; }')
        self.btn_serial.setStyleSheet('QPushButton { font: bold 11px; }')
        
        self.show()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def paintEvent(self, event):

        painter = QPainter(self)                                                                        # Set the environment for icon visualization
        
        painter.drawPixmap(QPoint(800,1000), self.unimib_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(600,1003), self.eurocold_logo.scaled(172, 172, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))
        painter.drawPixmap(QPoint(520,1000), self.disat_logo.scaled(50, 50, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation))

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def volt_plot(self, color1, color2, width, style):

        return self.third_window, self.volt_plt, self.volt_plt.plot(pen=pg.mkPen(color1, width=width, style=style)), self.volt_plt.plot(pen=pg.mkPen(color2, width=width, style=style))
    

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def volt_liveplot(self, color1, color2, width, style):

        return self.third_window, self.volt_plt, self.volt_plt.plot(pen=pg.mkPen(color1, width=width, style=style)), self.volt_plt.plot(pen=pg.mkPen(color2, width=width, style=style))


############################################################################################################################################################
############################################################################################################################################################
