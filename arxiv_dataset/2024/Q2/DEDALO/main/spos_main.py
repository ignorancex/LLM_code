############################################################################################################################################################
############################################################################################################################################################

# Program name: abakus.pyw
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 22 June 2022 (last modified)
# Objective: Main program for Python3 Abakus GUI visualization
# Description: This Python3 script creates an intercative window for the user to analyze and process the content of Abakus files.
#              Linked with outher files in the directory, it defines a GUI panel divided into two halfes:
#              - on the left hand side, histograms are retrieved from the specified file(s); the distributions are referred both to the single measurement
#                ('Cumulative distribution'), which is taken by the first file in the specified list, and to the comparison of multiple measurements
#                ('Normalized cumulative distribution(s)') appropriately normalized.
#                Also, here is specified that the Abakus laser sensor works in spherical approximation, highlighting the relation betweene the extinction 
#                cross-section and the extinction diameter.
#              - on the right hand side, the user can first specifiy the input parameters (repository, file name(s), CFA flow rate, heading rows to skip in reading
#                the file, acquisition time, save path and name for the plots). Moreover, here are reported some fixed settings such as the Abakus range 
#                [1-9] microns and its resolution (100 nm).
#                In the lower part, an output window show the results from this analysis.

############################################################################################################################################################
############################################################################################################################################################


import sys, numpy as np, math as m, itertools, os, serial, serial.tools.list_ports, pyqtgraph as pg     # Import the required libraries
import matplotlib.pyplot as plt, traceback, matplotlib as mpl
import pyqtgraph.exporters
from datetime import datetime
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import warnings
warnings.filterwarnings("ignore")

_PATH = os.path.abspath(os.path.realpath(__file__))[2:-17].replace('\\', '/')

sys.path.insert(0, _PATH+'methods/')
from abakus_class import Abakus
from my_widgets import First_Panel, Second_Panel, Third_Panel
from data_correction import Data_corrector
from plot_correction import CData_Plotter

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.default'] = 'it'


############################################################################################################################################################
############################################################################################################################################################
# Method to list all the available serial ports on the user PC.

def find_USB_device(USB_DEV_NAME=None):

    myports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
    usb_device_list = [p[1] for p in myports]

    if USB_DEV_NAME is None: return myports                                                             # If the port name is not specified, the function returns
    else:                                                                                               # all the available serial ports, otherwise the selected 
        USB_DEV_NAME=str(USB_DEV_NAME).replace("'","").replace("b","")                                  # port ID is given
        for device in usb_device_list:
            if USB_DEV_NAME in device:
                usb_id = device[device.index("COM"):device.index("COM")+4]
        
                return usb_id


############################################################################################################################################################
############################################################################################################################################################


class Ui(QtWidgets.QMainWindow, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Class contructor: creates the Abakus GUI.
    
    def __init__(self, _PATH):
        
        super(Ui, self).__init__()
        uic.loadUi(_PATH+'subGUIs/abakus_GUI.ui', self)                                                 # Load the graphical interface
        self.setWindowIcon(QIcon(_PATH+'_icon/abakus_logo_tPk_5.ico'))
        self.setFixedWidth(1920)
        self.setFixedHeight(1040)

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        self.setCentralWidget(self.centralwidget)
        self.log_path = _PATH+'log_files/'

        self.print_on_terminal = False
                
        self.connect_widgets()                                                                          # Connect the buttons to the corresponding methods

        self.current_directory = os.getcwd()                                                            # Get the current directory where the script is stored
        _directory_path = _PATH[:-7]+'DEDALO/DATA/measurements/'                                        # Define the default path for saving the data measured by the Abakus
        _save_path = _PATH[:-7]+'DEDALO/DATA/results/'                                                  # laser sensor and the results derivedd from their analysis
        self.python_exec_path = 'C:/Users/EuroCold/AppData/Local/Programs/Python/Python39-32/python.exe'
        _files = "Type file name ( or list of file names [ , ] separated )"
        self.time_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f")[:-4]                            # Starting time of the measurement
        self.wavelength = 0.670                                                                         # Laser wavelength
        self.k = 2*m.pi/self.wavelength                                                                 # Wavenumber
        self.sizes = np.array([1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4, 6.7, 7.0, 7.3, 7.6, 7.9, 8.2, 8.5, 8.8, 9.1, 9.4, 9.7, 10.0, 10.3])
        self.error = self.sizes[1] - self.sizes[0]                                                      # Uncertainty in size distributon measurement (dafault value: 300 nm)
        self.portlist = find_USB_device()                                                               # List the available serial ports
        self.items = [p[0] for p in self.portlist]
        self.model = 'LDS 23/25'                                                                        # Abakus model
        self.cell_material = 'quartz'                                                                   # Abakus cell material
        self.ID_number = 'Abakus C Q1/02 --- Nr. AC1289'                                                # Abakus ID manufacturer number
        self.size_range = '1.0 - 120.0'                                                                 # Detectable particle full-range
        self.live_measurement = False                                                                   # Initialization of labels for live measurement, start/pause/stop the analysis
        self.stop = False                                                                               # and so on...
        self.pause = False
        self.update_label = False
        self.run_label = False
        self.calibration_label = False
        self.image_width = 900                                                                          # Default width for saving images
        self.repetition_time = 1                                                                        # Time interval for QTimer repetitio (1 second)
        self.xcell = 250                                                                                # Sizes {x, y, z} of the Abakus inner cell
        self.ycell = 230
        self.zcell_laser = 1.5
        
        self.lineEdit_directory_path.setText(_directory_path)
        self.lineEdit_save_path.setText(_save_path)
        self.lineEdit_file_name.setText(_files)
        self.lineEdit_date_and_time.setText(self.time_str)
        self.lineEdit_volt.setText('0')
        self.lineEdit_size.setText(' from 1.0 to 10.3; step of 300 nm')
        self.lineEdit_RAM.setText('0')
        self.lineEdit_iteration_time.setText('0')
        self.lineEdit_delay.setText('80')                                                               # Default delay time between two consecutive serial writing and reading operations
        self.lineEdit_skip.setText('38')                                                                # Default number of heading lines to skip for analyzinff Abakus output files

        self.first_panel.lineEdit_software.setText(' - - - - ')
        self.second_panel.lineEdit_software.setText(' - - - - ')
        self.third_panel.lineEdit_software.setText(' - - - - ')

        self.lineEdit_skip.setFixedWidth(80)
        self.lineEdit_delay.setFixedWidth(80)
        self.lineEdit_acq_time.setFixedWidth(80)
        self.lineEdit_flow_rate.setFixedWidth(80)
        self.lineEdit_date_and_time.setFixedWidth(200)
        self.lineEdit_file_name.setFixedWidth(480)
        self.lineEdit_abakus_run.setFixedWidth(20)
        self.lineEdit_volt_control.setFixedWidth(20)
        self.lineEdit_RAM_control.setFixedWidth(20)
        self.lineEdit_volt.setFixedWidth(119)
        self.lineEdit_RAM.setFixedWidth(119)
        self.lineEdit_iteration_time.setFixedWidth(119)
        self.lineEdit_size.setFixedWidth(170)
        self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: red; }")
        self.lineEdit_volt_control.setStyleSheet("QLineEdit { background: red; }")
        self.lineEdit_RAM_control.setStyleSheet("QLineEdit { background: red; }")
        self.label_separation.setFont(QFont('Arial', 1))
        self.label_separation1.setFont(QFont('Arial', 1))
        self.btn_live.setFixedWidth(130)
        self.btn_run.setFixedWidth(130)
        self.btn_save.setFixedWidth(100)
        self.btn_pause.setFixedWidth(100)
        self.btn_stop.setFixedWidth(100)
        self.btn_live.setFixedHeight(30)
        self.btn_run.setFixedHeight(30)
        self.btn_save.setFixedHeight(30)
        self.btn_pause.setFixedHeight(30)
        self.btn_stop.setFixedHeight(30)

        self.tab_widget_d.setTabText(0, 'Size (local) and time distributions')
        self.tab_widget_d.setTabText(1, 'Size (incremental) and time distributions')
        self.tab_widget_d.setTabText(2, 'Voltage monitor (laser diode and RAM)')

        self.label_flow_description = QtWidgets.QLabel(self.groupBox_data)
        self.label_flow_description.setText("flow rate set during continuous flow analysis measurement")
        self.label_flow_description.setStyleSheet("QLabel { font: bold 8px; }")
        self.label_flow_description.setGeometry(QtCore.QRect(245, 145, 250, 20))

        self.label_delay_description = QtWidgets.QLabel(self.groupBox_data)
        self.label_delay_description.setText("time delay between two consecutive writing and reading operations on serial port (in ACQUISITION mode)")
        self.label_delay_description.setStyleSheet("QLabel { font: bold 8px; }")
        self.label_delay_description.setGeometry(QtCore.QRect(245, 169, 390, 20))

        self.label_skip_description = QtWidgets.QLabel(self.groupBox_data)
        self.label_skip_description.setText("heading rows to skip at the beginning of the file (in ANALYSIS mode)")
        self.label_skip_description.setStyleSheet("QLabel { font: bold 8px; }")
        self.label_skip_description.setGeometry(QtCore.QRect(245, 195, 300, 20))        

        self.groupBox_data.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox_2.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox_err.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.fixed_settings.setStyleSheet('QGroupBox { font: bold 11px; }')
        self.groupBox_3.setStyleSheet('QGroupBox { font: bold 11px; }')

        self.btn_live.setText('START ACQUISITION')
        self.btn_run.setText('START ANALYSIS')
        self.btn_save.setText('SAVE')
        self.btn_pause.setText('PAUSE')
        self.btn_stop.setText('STOP')

        self.btn_live.setStyleSheet("QPushButton { font: bold 10px; }")
        self.btn_run.setStyleSheet("QPushButton { font: bold 10px; }")
        self.btn_save.setStyleSheet("QPushButton { font: bold 10px; }")
        self.btn_pause.setStyleSheet("QPushButton { font: bold 10px; }")
        self.btn_stop.setStyleSheet("QPushButton { font: bold 10px; }")

        self.output = QtWidgets.QTextBrowser(self.groupBox_2)                                           # Output window for results and data visualization
        self.output.setGeometry(QtCore.QRect(5, 25, 699, 200))
        self.output.setObjectName("output")
        self.output.setStyleSheet("QTextBrowser { background: white; color: green; }")
    
        self.output_err = QtWidgets.QTextBrowser(self.groupBox_err)                                     # Output window for errors and warnigns during the script eecution
        self.output_err.setGeometry(QtCore.QRect(5, 25, 699, 65))
        self.output_err.setObjectName("output_err")
        self.output_err.setStyleSheet("QTextBrowser { background: white; color: red; }")

        self.output_noise = QtWidgets.QTextBrowser(self.fixed_settings)                                 # Output window for the 32 Abakus channel voltages
        self.output_noise.setGeometry(QtCore.QRect(300, 55, 404, 94))
        self.output_noise.setObjectName("noise")
        self.output_noise.setStyleSheet("QTextBrowser { background: white; color: green; }")

        self.select_file = QtWidgets.QPushButton(self.groupBox_data)                                    # Button to select the desired file to analyse
        self.select_file.setText("Select")
        self.select_file.setStyleSheet("QPushButton { font: bold 11px; }")
        self.select_file.setGeometry(QtCore.QRect(608, 56, 80, 20))
        self.select_file.clicked.connect(self.getfiles)

        self.progressbar = QProgressBar(self.groupBox_data)                                             # Progress bar initialization
        self.progressbar.move(141, 326)
        self.progressbar.adjustSize()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setFixedWidth(565)
        self.progressbar.setFixedHeight(21)
        self.progressbar.setTextVisible(False)

        self.first_panel.combobox_port.addItems(self.items)                                             # Load the four panels for data rendering and visualization
        self.second_panel.combobox_port.addItems(self.items)                                            # described in 'my_widgets.py'
        self.third_panel.combobox_port.addItems(self.items)
        self.first_panel.combobox_port.setCurrentIndex(self.first_panel.combobox_port.count()-1)
        self.second_panel.combobox_port.setCurrentIndex(self.first_panel.combobox_port.count()-1)
        self.third_panel.combobox_port.setCurrentIndex(self.first_panel.combobox_port.count()-1)

        try:                                                                                            # Connect to Abakus serial port
            if self.items!=[]: self.output.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t '+self.first_panel.combobox_port.currentText()+' serial port correctly connected.\n\n########################################################\n')
            self.abakus = Abakus(self.first_panel.combobox_port.currentText())
        except: 
            self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t ERROR: No serial port detected. Please check USB and/or RS-232 connection.')
            self.abakus = Abakus('_default')

        self.correction_window = Data_corrector(self.wavelength)                                        # Load the class for further data correction and interpretation,
        self.correction_window.settings()                                                               # as described in 'data_correction.py'

        self.btn_save.setEnabled(False)                                                                 # Disable buttons at the beginning
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.first_panel.btn_update.setEnabled(False)
        self.first_panel.btn_correct.setEnabled(False)
        self.first_panel.btn_voltage_noise.setEnabled(False)
        self.second_panel.btn_update.setEnabled(False)
        self.second_panel.btn_correct.setEnabled(False)
        self.second_panel.btn_voltage_noise.setEnabled(False)
        self.third_panel.btn_update.setEnabled(False)
        self.third_panel.btn_voltage_noise.setEnabled(False)

        self.show()                                                                                     # Show the graphical interface


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #
        
    def connect_widgets(self):

        try:
        
            self.btn_live.clicked.connect(self.on_live_clicked)
            self.btn_run.clicked.connect(self.on_run_clicked)
            self.btn_save.clicked.connect(self.on_save_clicked)
            self.btn_pause.clicked.connect(self.on_pause_clicked)
            self.btn_stop.clicked.connect(self.on_stop_clicked)

            self.first_panel.btn_serial.clicked.connect(self.on_serial_clicked)
            self.first_panel.btn_help.clicked.connect(self.on_help_clicked)
            self.first_panel.btn_update.clicked.connect(self.on_plot_update_clicked)
            self.first_panel.btn_calibration.clicked.connect(self.on_calibration_clicked)
            self.first_panel.btn_correct.clicked.connect(self.on_data_correction_clicked)
            self.first_panel.btn_voltage_noise.clicked.connect(self.on_voltage_noise_plot_clicked)
            self.second_panel.btn_serial.clicked.connect(self.on_serial_clicked)
            self.second_panel.btn_help.clicked.connect(self.on_help_clicked)
            self.second_panel.btn_update.clicked.connect(self.on_plot_update_clicked)
            self.second_panel.btn_calibration.clicked.connect(self.on_calibration_clicked)
            self.second_panel.btn_correct.clicked.connect(self.on_data_correction_clicked)
            self.second_panel.btn_voltage_noise.clicked.connect(self.on_voltage_noise_plot_clicked)
            self.third_panel.btn_serial.clicked.connect(self.on_serial_clicked)
            self.third_panel.btn_help.clicked.connect(self.on_help_clicked)
            self.third_panel.btn_calibration.clicked.connect(self.on_calibration_clicked)
            self.third_panel.btn_voltage_noise.clicked.connect(self.on_voltage_noise_plot_clicked)

        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()


    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def getfiles(self):

        filename = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select one or more files to open', self.lineEdit_directory_path.text(), '*.txt')
        list = []
        for f in filename[0]: list.append(f[len(self.lineEdit_directory_path.text())+2:])
        self.lineEdit_file_name.setText(str(list).translate({ord(c): None for c in "'[]"}))


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #
    
    def print_gui_terminal(self):

        try:
            for f in self.filess:
                self.output.append('########################################################\n'+'FILE: '+"'"+f+"'"+'\n########################################################\n')
                self.output.append('1. Average laser diode voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.volt1))+' mV')  
                self.output.append('    Avergae RAM-buffer voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.RAM1))+' mV\n')  
                self.output.append('2. Flow rate:\t\t\t\t'+str(self.flow1)+' mL/min')  
                self.output.append('    Particles detected:\t\t\t'+'{:.2e}'.format(sum(self.data1.sum(axis=0)))+' pt')  
                self.output.append('    Total particles concentration:\t\t\t'+'{:.2e}'.format(self.ptc_conc1)+' pt/mL') 
                self.output.append('    Counts distribution peaked @:\t\t\t'+'{:.02f}'.format(self.sizes[np.where(self.h1[:-1]==np.amax(self.h1[:-1]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
                self.output.append('    Counts distribution average:\t\t\t'+'{:.02f}'.format(np.average(self.sizes[:-1], weights=self.h1[:-1]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.h1[:-1]**2))/sum(self.h1[:-1]))+' µm')
                self.output.append('    Counts distribution average (arithmetical):\t\t'+'{:.02f}'.format(np.mean(self.sizes[:-1]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.sizes[:-1])))+' µm')
                self.output.append('    Counts distribution std. deviation:\t\t'+'{:.02f}'.format(np.sqrt(np.var((self.sizes[:-1]))))+' µm\n')
                self.output.append('3. Time-average # counts:\t\t\t'+'{:.02f}'.format(np.mean(np.array(self.data1.sum(axis=1)))))
                self.output.append('    Time std. deviation # counts:\t\t\t'+'{:.02f}'.format(np.sqrt(np.var(np.array(self.data1.sum(axis=1))))))
                self.output.append('    Time-median # counts:\t\t\t'+'{:.02f}'.format(np.median(np.array(self.data1.sum(axis=1)))))
                self.output.append('     First quantile # counts (in time):\t\t'+'{:.02f}'.format(np.quantile(np.array(self.data1.sum(axis=1)), 0.25)))
                self.output.append('    Third quantile # counts (in time):\t\t'+'{:.02f}'.format(np.quantile(np.array(self.data1.sum(axis=1)), 0.75))+'\n')
                self.output.append('---------------------------------------------------------------------------------------------------------------\n')
                for i in range(0, len(self.sizes)): self.output.append('Particles concentration @ '+str(self.sizes[i])+' mm:\t\t'+'{:.2e}'.format(self.ptc_conc_channel1[i][1])+' pt/mL') 
                self.output.append('')
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_calibration_clicked(self):

        try: os.popen('python3 ../methods/calibration.py')
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python3 '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()
        
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_help_clicked(self):

        try: os.system('start ../manual/manual.pdf')
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_serial_clicked(self):

        try: 
            self.abakus.close()
            os.popen('python '+_PATH+'methods/serial_monitor.py '+self.first_panel.combobox_port.currentText()+' '+_PATH)
            self.abakus = Abakus(self.first_panel.combobox_port.currentText())
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()
        
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_live_clicked(self):

        self.btn_save.setEnabled(True)                                                                  # Enable the disabled buttons
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_run.setEnabled(False)                                                                  # Disabel the button for running offline analyses
        self.first_panel.btn_update.setEnabled(False)
        self.first_panel.btn_correct.setEnabled(False)
        self.first_panel.btn_voltage_noise.setEnabled(True)
        self.second_panel.btn_update.setEnabled(False)
        self.second_panel.btn_correct.setEnabled(False)
        self.second_panel.btn_voltage_noise.setEnabled(True)
        self.third_panel.btn_update.setEnabled(False)
        self.third_panel.btn_voltage_noise.setEnabled(True)

        self.live_measurement = True

        self.btn_live.setStyleSheet("QPushButton { color: green; font: bold 10px; }")
        self.btn_run.setStyleSheet("QPushButton { color: gray; font: bold 10px; }")
        self.btn_stop.setStyleSheet("QPushButton { color: red; font: bold 10px; }")

        self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: green; font: bold 10px; }")
        self.lineEdit_volt_control.setStyleSheet("QLineEdit { background: green; font: bold 10px; }")
        self.lineEdit_RAM_control.setStyleSheet("QLineEdit { background: green; font: bold 10px; }")
        self.live_run()

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_pause_clicked(self):

        if self.btn_pause.isChecked(): 
            if self.live_measurement == True:
                self.pause = True
                self.timer.stop()
                self.abakus.close()
                self.output.append('\nMeasurement paused, release the button to restart.\n\n########################################################\n')
        if not self.btn_pause.isChecked(): 
            if self.live_measurement == True:
                self.pause = False
                self.abakus = Abakus(self.first_panel.combobox_port.currentText())
                directory_path = self.lineEdit_directory_path.text()
                files = [self.lineEdit_file_name.text()]
                save_path = self.lineEdit_save_path.text()
                txt_save_name = self.lineEdit_save_name.text()
                self.abakus.abakus_parameters(self.ID_number, directory_path, files, 0, self.flow_rate, self.sizes, 0, self.time_delay, self.time_str, save_path, txt_save_name, self.model, self.cell_material, self.size_range, self.print_on_terminal, self.output, self.output_err)

                self.channels, self.software, self.noise = self.abakus.initialization(b'C0001\n', b'X0003\n', b'C0013\n')
                self.abakus.sendCommand(b'C0005\n')
                self.counts_sum = 0
                self.prev_time = datetime.now()
                self.data_bkp = np.zeros(len(self.channels[1]))
                self.timer.start(1000*self.repetition_time) 
        
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_stop_clicked(self):

        self.update_label = False
        self.btn_pause.setEnabled(False)
        self.btn_live.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.first_panel.btn_update.setEnabled(False)
        self.first_panel.btn_correct.setEnabled(False)
        self.first_panel.btn_voltage_noise.setEnabled(False)
        self.second_panel.btn_update.setEnabled(False)
        self.second_panel.btn_correct.setEnabled(False)
        self.second_panel.btn_voltage_noise.setEnabled(False)
        self.third_panel.btn_update.setEnabled(False)
        self.third_panel.btn_voltage_noise.setEnabled(False)

        self.btn_live.setStyleSheet("QPushButton { color: black; font: bold 10px; }")
        self.btn_run.setStyleSheet("QPushButton { color: black; font: bold 10px; }")
        self.btn_stop.setStyleSheet("QPushButton { color: gray; font: bold 10px; }")
    
        if self.live_measurement == True:
            self.stop = True
            self.timer.stop() 
            self.abakus.close()
            self.work_book.save(self.txt_full_path+self.output_txt_name+'_'+self.temp+'.xlsx')
            self.saving_txtfile.close()
            self.output.append('\nMeasurement and serial communication ended.\n\n########################################################\n')
        
        if self.live_measurement == False:
            self.stop = True
            self.abakus.close()
            self.output.append('\nMeasurement and serial communication ended.\n\n########################################################\n')


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_voltage_noise_plot_clicked(self):

        try:

            if self.live_measurement==False:

                try:
                    x_values, y_values = [], []

                    figure = plt.figure(figsize=(8, 6))
                    plt.suptitle('Abakus voltage calibration curve', size=16, y=0.97) 
                    figure.subplots_adjust(left=0.13, right=0.975, top=0.93, bottom=0.105)
                    a = figure.add_subplot(111)
                    a.tick_params(axis='both', which='major', labelsize=20)
                    a.set_ylabel('Diameter [$\mathrm{\mu}$m]', fontsize=20, labelpad=20)
                    a.set_xlabel('Voltage [V]', fontsize=20, labelpad=20)
                    for j in range(0, len(self.abakus_noises)-1): 
                        x_values.append(float(self.abakus_noises[j].split()[4]))
                        y_values.append(float(self.abakus_noises[j].split()[1]))
                    x_values, y_values = np.array(x_values)/1000, np.array(y_values)
                    
                    poly_coefficients, cov_matrix = np.polyfit(x_values, y_values, 2, full=False, cov=True)
                    voltage_fit = np.poly1d(poly_coefficients)
                    voltage_fit_lower, voltage_fit_upper = np.poly1d(poly_coefficients-2*np.diag(cov_matrix)), np.poly1d(poly_coefficients+2*np.diag(cov_matrix))
                    a.plot(x_values, voltage_fit(x_values), 'k', linewidth=2, label='fit')
                    a.plot(x_values, voltage_fit_lower(x_values), 'k', linewidth=0.2); a.plot(x_values, voltage_fit_upper(x_values), 'k', linewidth=0.2)
                    a.fill_between(x_values, voltage_fit_upper(x_values), voltage_fit_lower(x_values), color='y', label='2σ deviation')
                    a.plot(x_values, y_values, '^', markersize=8, markerfacecolor='orange', markeredgecolor='r', markeredgewidth=3, label='voltage calibration')
                    a.legend(loc='lower right', prop={'size': 18})
                    figure.tight_layout()
                    plt.show()
                except: print('')

            if self.live_measurement==True:

                x_values, y_values = [], []

                figure = plt.figure(figsize=(8, 6))
                plt.suptitle('Abakus voltage calibration curve', size=16, y=0.97) 
                figure.subplots_adjust(left=0.085, right=0.960, top=0.93, bottom=0.100)
                a = figure.add_subplot(111)
                a.set_ylabel('Diameter [$\mathrm{\mu}$m]', fontsize=12)
                a.set_xlabel('Voltage [mV]', fontsize=12)
                for j in range(0, len(self.noise[1])-2, 2): 
                    x_values.append(float(10*self.noise[1][j]))
                    y_values.append(float(self.noise[1][j+1]))
                x_values, y_values = np.array(x_values), np.array(y_values)
                
                poly_coefficients, cov_matrix = np.polyfit(x_values, y_values, 2, full=False, cov=True)
                voltage_fit = np.poly1d(poly_coefficients)
                voltage_fit_lower, voltage_fit_upper = np.poly1d(poly_coefficients-3*np.diag(cov_matrix)), np.poly1d(poly_coefficients+3*np.diag(cov_matrix))
                a.plot(x_values, voltage_fit(x_values), 'k', linewidth=2, label='fit')
                a.plot(x_values, voltage_fit_lower(x_values), 'k', linewidth=0.2); a.plot(x_values, voltage_fit_upper(x_values), 'k', linewidth=0.2)
                a.fill_between(x_values, voltage_fit_upper(x_values), voltage_fit_lower(x_values), color='y', label='3σ deviation')
                a.plot(x_values, y_values, '^', markersize=8, markerfacecolor='orange', markeredgecolor='r', markeredgewidth=3, label='voltage calibration')
                a.legend(loc='lower right')
                plt.show()
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_plot_update_clicked(self):

        if self.live_measurement == False:
            
            self.update_label = True

            self.indexes = len(self.sizes) - 1

            self.new_single_d_range = self.single_d_lr.getRegion()
            self.new_time1_range = self.time1_lr.getRegion()
            self.new_incremental_d_range = self.incremental_d_lr.getRegion()
            self.new_time2_range = self.time2_lr.getRegion()

            if self.new_single_d_range!=self.single_d_range: 
                self.indexes = np.where((self.sizes >= self.new_single_d_range[0]) & (self.sizes <= self.new_single_d_range[1]))[0]
                self.single_d_lr.setRegion([self.new_single_d_range[0], self.new_single_d_range[1]])
                self.curve_single_d_upd.setData(self.sizes[self.indexes], self.h1[self.indexes], stepMode='right')
            
                self.update_window()

            if self.new_incremental_d_range!=self.incremental_d_range: 
                self.indexes = np.where((self.sizes >= self.new_incremental_d_range[0]) & (self.sizes <= self.new_incremental_d_range[1]))[0]
                self.incremental_d_lr.setRegion([self.new_incremental_d_range[0], self.new_incremental_d_range[1]])
                self.curve_incremental_d_upd.setData(self.sizes[self.indexes], self.h1[self.indexes], stepMode='right')

                self.update_window()

            if self.new_time1_range!=self.time1_range:
                self.time_indexes = np.where((np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]) >= self.new_time1_range[0]) & (np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]) <= self.new_time1_range[1]))[0]
                self.time1_lr.setRegion([self.new_time1_range[0], self.new_time1_range[1]])
                self.curve_time1_avg.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), np.mean(self.time_data[self.time_indexes])*np.ones(self.data1.shape[0]))
                if len(self.time_indexes)==1: self.single_histogram = np.array(self.data1.loc[self.time_indexes[0]]) - np.array(self.data1.loc[self.time_indexes[0]-1])
                elif len(self.time_indexes)>1: 
                    self.single_histogram = np.array(self.data1.loc[self.time_indexes[0]]) - np.array(self.data1.loc[self.time_indexes[0]-1])
                    for kk in range(1, len(self.time_indexes)):
                        self.single_histogram += np.array(self.data1.loc[self.time_indexes[kk]]) - np.array(self.data1.loc[self.time_indexes[kk]-1])
                self.curve_single_d.setData(self.sizes[:-1], self.single_histogram[:-1], stepMode='right')

            if self.new_time2_range!=self.time2_range:
                self.time_indexes = np.where((np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]) >= self.new_time2_range[0]) & (np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]) <= self.new_time2_range[1]))[0]
                self.time2_lr.setRegion([self.new_time2_range[0], self.new_time2_range[1]])
                self.curve_time2_avg.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), np.mean(self.time_data[self.time_indexes])*np.ones(self.data1.shape[0]))

            self.single_d_range = self.single_d_lr.getRegion()
            self.time1_range = self.time1_lr.getRegion()
            self.incremental_d_range = self.incremental_d_lr.getRegion()
            self.time2_range = self.time2_lr.getRegion()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #
    
    def update_window(self):

        self.left, self.top, self.width, self.height = 570, 450, 460, 300
        self.window = QWidget() 
        self.window.setWindowTitle("Updated Abakus results")
        self.window.setGeometry(self.left, self.top, self.width, self.height)

        self.output_update = QtWidgets.QTextBrowser(self.window)
        self.output_update.setGeometry(QtCore.QRect(2, 2, 456, 296))
        self.output_update.setObjectName("update")
        self.output_update.setStyleSheet("QTextBrowser { background: white; color: green; }")

        for f in self.filess:
            self.output_update.append('########################################################\n'+'FILE: '+"'"+f+"'"+'\n########################################################\n')
            self.output_update.append('1. Range: d = ['+'{:.02f}'.format(self.sizes[self.indexes][0])+', '+'{:.02f}'.format(self.sizes[self.indexes][-1])+'] um\n')
            self.output_update.append('2. Average laser diode voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.volt1))+' mV')  
            self.output_update.append('    Avergae RAM-buffer voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.RAM1))+' mV\n')  
            self.output_update.append('3. Flow rate:\t\t\t\t'+str(self.flow1)+' mL/min')  
            self.output_update.append('    Particles detected:\t\t\t'+'{:.2e}'.format(sum(self.h1[self.indexes]))+' pt')  
            self.output_update.append('    Counts distribution peaked @:\t\t\t'+'{:.02f}'.format(self.sizes[np.where(self.h1==np.amax(self.h1[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            try: self.output_update.append('    Counts distribution average:\t\t\t'+'{:.02f}'.format(np.average(self.sizes[self.indexes], weights=self.h1[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.h1[self.indexes]**2))/sum(self.h1[self.indexes]))+' µm')
            except: self.output_update.append('    Counts distribution average:\t\t\t'+'nan')
            self.output_update.append('    Counts distribution average (arithmetical):\t\t'+'{:.02f}'.format(np.mean(self.sizes[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.indexes)))+' µm\n')
            self.output_update.append('    Counts distribution std. deviation:\t\t'+'{:.02f}'.format(np.sqrt(np.var((self.sizes[self.indexes]))))+' µm')
            self.output_update.append('    First quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.25))+' µm')
            self.output_update.append('    Second quantile # counts:\t\t\t'+'{:.02f}'.format(np.median(self.sizes[self.indexes]))+' µm')
            self.output_update.append('    Third quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.75))+' µm')
            self.output_update.append('')

        self.window.show()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_data_correction_clicked(self):

        try:

            self.correction_window.progressbar.setValue(0)

            self.calibration_label = True

            if self.run_label==True:

                if self.first_panel.btn_correct.isChecked() or self.second_panel.btn_correct.isChecked(): self.correction_window.show()

                self.correction_window.btn_run.clicked.connect(self.on_data_correction_execute)
        
        except:

            if os.path.isdir(self.log_path): print('')										                # If the log path does not exists, it is created
            else: os.makedirs(self.log_path)
            with open(self.log_path+'report.log', 'a+') as fh:
                fh.write('\n'+datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")[:-7]+'\n')
                e_type, e_val, e_tb = sys.exc_info()
                traceback.print_exception(e_type, e_val, e_tb, file=fh)

            os.popen('python '+_PATH+'methods/error_handling.py log_files/report.log')
            sys.exit()

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_data_correction_execute(self):

        self.sizes_RI_cal, self.sizes_ar_cal = np.zeros(len(self.sizes)), np.zeros(len(self.sizes))
        self.ref_index_Re, self.ref_index_Im = 0, 0
        self.diameters_Cext, self.Cext_polystirene, self.selected_Cext, self.poly_fit = np.zeros(len(self.sizes)), np.zeros(len(self.sizes)), np.zeros(len(self.sizes)), np.poly1d(1)
        
        if self.correction_window.ref_index_correction_label==True: 
            self.sizes_RI_cal, self.ref_index_Re, self.ref_index_Im, self.diameters_Cext, self.Cext_polystirene, self.selected_Cext, self.poly_fit = self.correction_window.refractive_index_calibration_correction(self.sizes)
        
        if self.correction_window.aspect_ratio_correction_label==True: self.sizes_ar_cal = self.correction_window.aspect_ratio_calibration_correction(self.sizes)

        self.correction_labels = [self.correction_window.ref_index_correction_label, self.correction_window.aspect_ratio_correction_label]
        self.x_data = [self.sizes[:-1], self.sizes_RI_cal[:-1], self.sizes_ar_cal[:-1]]

        self.correction_window.close()
        self.first_panel.btn_correct.setChecked(False)

        self.correction_plot = CData_Plotter(self.x_data, self.h1[:-1], self.time_data, self.data1, self.ref_index_Re, self.ref_index_Im, self.diameters_Cext, self.Cext_polystirene, self.selected_Cext, self.poly_fit, f"{self.lineEdit_save_path.text()}/{self.time_str[:-12]}/", self.lineEdit_save_name.text(), self.correction_labels)
        self.correction_plot.show()

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_run_clicked(self):

        self.time_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S.%f")[:-4]
        self.lineEdit_date_and_time.setText(self.time_str)

        self.btn_save.setEnabled(True)                                                                  # Enable the disabled buttons
        self.btn_pause.setEnabled(False)
        self.btn_live.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.first_panel.btn_update.setEnabled(True)
        self.first_panel.btn_correct.setEnabled(True)
        self.first_panel.btn_voltage_noise.setEnabled(True)
        self.second_panel.btn_update.setEnabled(True)
        self.second_panel.btn_correct.setEnabled(True)
        self.second_panel.btn_voltage_noise.setEnabled(True)
        self.third_panel.btn_update.setEnabled(True)
        self.third_panel.btn_voltage_noise.setEnabled(True)

        self.btn_run.setStyleSheet("QPushButton { color: green; font: bold 10px; }")
        self.btn_live.setStyleSheet("QPushButton { color: gray; font: bold 10px; }")
        self.btn_stop.setStyleSheet("QPushButton { color: red; font: bold 10px; }")

        self.lineEdit_iteration_time.setText('0')
        self.lineEdit_volt.setText('0')
        self.lineEdit_RAM.setText('0')
        self.lineEdit_volt_control.setStyleSheet("QLineEdit { background: red; }")
        self.lineEdit_RAM_control.setStyleSheet("QLineEdit { background: red; }")
        self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: red; }")

        self.live_measurement = False

        if self.stop == True: 
            self.abakus = Abakus(self.first_panel.combobox_port.currentText())
            self.single_d_plt.clear()
            self.incremental_d_plt.clear()
            self.time1_plt.clear()
            self.time2_plt.clear()
            self.volt_plt.clear()
            self.output_err.clear()
            self.output.clear()
            self.output_noise.clear()

        self.run_label = True

        if self.lineEdit_skip.text()=='': self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t ERROR: Number of heading lines to skip missing. Please specify it.')

        if self.lineEdit_acq_time.text()=='': self.acq_time = 0
        else: self.acq_time = int(self.lineEdit_acq_time.text())

        skip_rows = int(self.lineEdit_skip.text())
        directory_path = self.lineEdit_directory_path.text()
        files = [self.lineEdit_file_name.text()]
        save_path = self.lineEdit_save_path.text()
        txt_save_name = self.lineEdit_save_name.text()+'.txt'

        splitting, self.filess = [], []
        for i in range(0, len(files[0])):
            if files[0][i]!=',': splitting.append(files[0][i])
            if files[0][i]==',' or i==len(files[0])-1: 
                self.filess.append(''.join(splitting[0:len(splitting)]))
                splitting = []
        for j in range(0, len(self.filess)):
            if self.filess[j][0] == ' ': self.filess[j] = self.filess[j][1:]
            if os.path.exists(directory_path+self.filess[j]): self.output.append(self.filess[j]+' found in directory')
            else: self.output_err.append(self.filess[j]+' not found, check it out!')
        self.output.append('')

        self.abakus.abakus_parameters(self.ID_number, directory_path, self.filess, skip_rows, 0, self.sizes, self.acq_time, 0, self.time_str, save_path, txt_save_name, self.model, self.cell_material, self.size_range, self.print_on_terminal, self.output, self.output_err)

        self.abakus_noises, self.flow1, self.vol, self.data1, self.ptc_conc1, self.ptc_conc_channel1, self.h1, self.volt1, self.RAM1 = self.abakus.scd_analysis()

        for j in range(0, len(self.abakus_noises)): self.output_noise.append(self.abakus_noises[j][12:-1])

        self.time_data = [0, np.array(self.data1.sum(axis=1))[1] - np.array(self.data1.sum(axis=1))[0], np.array(self.data1.sum(axis=1))[2] - np.array(self.data1.sum(axis=1))[1]]
        for k in range(3, self.data1.shape[0]):
            time_increment_1 = np.array(self.data1.sum(axis=1))[k-2] - np.array(self.data1.sum(axis=1))[k-3]
            time_increment_2 = np.array(self.data1.sum(axis=1))[k-1] - np.array(self.data1.sum(axis=1))[k-2]
            time_increment_3 = np.array(self.data1.sum(axis=1))[k] - np.array(self.data1.sum(axis=1))[k-1]

            if abs(time_increment_2-time_increment_1) > 4000 and abs(time_increment_3-time_increment_2) > 4000: 
                self.time_data.append(time_increment_2/2)
                self.time_data.append(time_increment_2/2)
            else: self.time_data.append(time_increment_3)

        k = 0
        for k in range(2, len(self.time_data)): 
            if abs(self.time_data[k-2] - self.time_data[k-1]) > 4000 and abs(self.time_data[k-1] - self.time_data[k]) > 1000: self.time_data[k-1] = self.time_data[k-1]/2
            if abs(self.time_data[k-2] - self.time_data[k-1]) > 3000 and abs(self.time_data[k-1] - self.time_data[k]) > 3000: self.time_data[k-1] = self.time_data[k-1]/2

        self.time_data = np.array(self.time_data)

        self.single_d_and_time_win, self.single_d_plt, self.curve_single_d, self.curve_single_d_upd, self.curve_single_d_cal, self.time1_plt, self.curve_time1, self.curve_time1_avg = self.first_panel.single_d_and_time_plot('b', 'r', '#028a0f', 'r', 'k', 4, QtCore.Qt.SolidLine, 4, QtCore.Qt.DashLine, (255,0,0,100))
        self.incremental_d_and_time_win, self.incremental_d_plt, self.curve_incremental_d, self.curve_incremental_d_upd, self.curve_incremental_d_cal, self.time2_plt, self.curve_time2, self.curve_time2_avg = self.second_panel.incremental_d_and_time_plot('b', 'r', '#028a0f', 'r', 'k', 4, QtCore.Qt.SolidLine, 4, QtCore.Qt.DashLine, (255,0,0,100))
        self.volt_win, self.volt_plt, self.curve_volt, self.curve_ram = self.third_panel.volt_plot('b', 'r', 4, QtCore.Qt.SolidLine)

        self.curve_single_d.setData(self.sizes[1:-1], np.array(self.data1.loc[0])[1:-1], stepMode='right')
        self.curve_time1.setData(np.linspace(0, len(self.time_data)-1, len(self.time_data)), self.time_data, stepMode='right')
        self.curve_time1_avg.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), np.mean(self.time_data)*np.ones(self.data1.shape[0]))
        self.curve_incremental_d.setData(self.sizes[1:-1], self.h1[1:-1], stepMode='right')
        self.curve_time2.setData(np.linspace(0, len(self.time_data)-1, len(self.time_data)), self.time_data, stepMode='right')
        self.curve_time2_avg.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), np.mean(self.time_data)*np.ones(self.data1.shape[0]))
        self.curve_volt.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), self.volt1)
        self.curve_ram.setData(np.linspace(0, self.data1.shape[0]-1, self.data1.shape[0]), self.RAM1)

        legend_single_d = pg.LegendItem((0,0), offset=(910,35))
        legend_single_d.setParentItem(self.single_d_plt.graphicsItem())
        legend_single_d.addItem(self.curve_single_d, '# counts')

        legend_time1 = pg.LegendItem((0,0), offset=(910,35))
        legend_time1.setParentItem(self.time1_plt.graphicsItem())
        legend_time1.addItem(self.curve_time1, '# counts')

        legend_incremental_d = pg.LegendItem((0,0), offset=(910,35))
        legend_incremental_d.setParentItem(self.incremental_d_plt.graphicsItem())
        legend_incremental_d.addItem(self.curve_incremental_d, '# counts')

        legend_time2 = pg.LegendItem((0,0), offset=(910,35))
        legend_time2.setParentItem(self.time2_plt.graphicsItem())
        legend_time2.addItem(self.curve_time2, '# counts')
        
        legend_volt = pg.LegendItem((0,0), offset=(820,300))
        legend_volt.setParentItem(self.volt_plt.graphicsItem())
        legend_volt.addItem(self.curve_volt, 'LASER diode voltage')
        legend_volt.addItem(self.curve_ram, 'RAM-buffer voltage')
            
        self.single_d_lr = pg.LinearRegionItem([self.sizes[-2], self.sizes[-2]+0.04], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.single_d_lr.setZValue(-10)
        self.single_d_plt.addItem(self.single_d_lr)
        self.single_d_range = self.single_d_lr.getRegion()

        self.time1_lr = pg.LinearRegionItem([self.data1.shape[0]-2, self.data1.shape[0]-1], pen=pg.mkPen(width=2.5), brush=(255,255,255,100), swapMode='push')
        self.time1_lr.setZValue(-10)
        self.time1_plt.addItem(self.time1_lr)
        self.time1_range = self.time1_lr.getRegion()

        self.incremental_d_lr = pg.LinearRegionItem([self.sizes[-2], self.sizes[-2]+0.04], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.incremental_d_lr.setZValue(-10)
        self.incremental_d_plt.addItem(self.incremental_d_lr)
        self.incremental_d_range = self.incremental_d_lr.getRegion()

        self.time2_lr = pg.LinearRegionItem([0, self.data1.shape[0]-1], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.time2_lr.setZValue(-10)
        self.time2_plt.addItem(self.time2_lr)
        self.time2_range = self.time2_lr.getRegion()

        self.print_gui_terminal()
        self.output.append('########################################################\n\n')


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def live_run(self):

        if self.lineEdit_flow_rate.text()=='': self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t WARNING: Flow rate missing. Please specify it.')
        
        self.flow_rate = float(self.lineEdit_flow_rate.text())
        directory_path = self.lineEdit_directory_path.text()
        files = [self.lineEdit_file_name.text()]
        self.time_delay = self.lineEdit_delay.text()
        save_path = self.lineEdit_save_path.text()
        txt_save_name = self.lineEdit_save_name.text()

        self.abakus.abakus_parameters(self.ID_number, directory_path, files, 0, self.flow_rate, self.sizes, 0, self.time_delay, self.time_str, save_path, txt_save_name, self.model, self.cell_material, self.size_range, self.print_on_terminal, self.output, self.output_err)

        self.channels, self.software, self.noise = self.abakus.initialization(b'C0001\n', b'X0003\n', b'C0013\n')
        self.abakus.sendCommand(b'C0005\n')
        self.output.append('Command '+b'C0005\n'.decode('utf-8')[:-1]+' sent to Abakus: measurement starts.\n')
        self.output.append('########################################################\n')

        self.saving_txtfile, self.worksheet, self.work_book, self.txt_full_path, self.output_txt_name, self.temp = self.abakus.starting_files(self.flow_rate, self.first_panel.combobox_port.currentText(), self.software[1][0][:-1], self.noise[1], self.channels[1], self.xcell, self.ycell, self.zcell_laser, self.wavelength)
        self.index = 0
        self.counts_sum = 0

        self.first_panel.lineEdit_software.setText(str(self.software[1][0][:-1]))
        self.second_panel.lineEdit_software.setText(str(self.software[1][0][:-1]))
        self.third_panel.lineEdit_software.setText(str(self.software[1][0][:-1]))

        for j in range(0, len(self.noise[1]), 2):
            if j<10: 
                if j+2!=10: self.output_noise.append(str((j+1)//2 + 1)+') '+str(self.noise[1][j+1])+' um\t--->\t'+str(10*self.noise[1][j])+' mV')
                else: self.output_noise.append(str((j+1)//2 + 1)+') '+str(self.noise[1][j+1])+' um\t--->\t'+str(10*self.noise[1][j])+' mV')
            elif j>=10: self.output_noise.append(str((j+1)//2 + 1)+') '+str(self.noise[1][j+1])+' um\t--->\t'+str(10*self.noise[1][j])+' mV')

        self.prev_time = datetime.now()
        self.time_data, self.time_volt, self.time_ram = [0], [], []
        self.data_bkp = np.zeros(len(self.channels[1]))

        self.output.append('TOTAL NUMBER OF PARTICLES DETECTED:\n')
        self.output.append('Time\t\t# counts\t\t# counts (incremental)')

        self.single_d_and_time_win, self.single_d_plt, self.curve_single_d, self.time1_plt, self.curve_time1 = self.first_panel.single_d_and_time_liveplot('b', 'r', 4, QtCore.Qt.SolidLine)
        self.incremental_d_and_time_win, self.incremental_d_plt, self.curve_incremental_d, self.time2_plt, self.curve_time2 = self.second_panel.incremental_d_and_time_liveplot('b', 'r', 4, QtCore.Qt.SolidLine)
        self.volt_win, self.volt_plt, self.curve_volt, self.curve_ram = self.third_panel.volt_liveplot('b', 'r', 4, QtCore.Qt.SolidLine)

        self.time1_plt.setYRange(0, 700)
        self.time2_plt.setYRange(0, 700)

        self.flow_rate = (10**11)*self.flow_rate/6
        self.volume = self.flow_rate*self.repetition_time
        self.speed = self.flow_rate/(self.xcell*self.ycell)
        self.z_pumped = self.speed*self.repetition_time
        self.counts_treshold = self.z_pumped/self.zcell_laser
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.cfa_measurement)
        self.timer.start(1000*self.repetition_time)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def cfa_measurement(self):

        if self.pause==False:

            self.incremental_data = np.zeros(len(self.channels[1]))
            self.volt, self.buffer, self.meas_data, self.init_time, self.end_time, self.running_label = self.abakus.single_measurement(b'C0012\n', b'U0004\n', b'U0003\n')
            self.lineEdit_iteration_time.setText(str((self.end_time-self.init_time).total_seconds()))
            self.lineEdit_volt.setText(str(self.volt))
            self.lineEdit_RAM.setText(str(self.buffer))
            
            for i in range(1, len(self.meas_data), 2): 
                self.counts_sum += self.meas_data[i]
                self.incremental_data[(i-1)//2] = self.meas_data[i]
            self.counts_per_cycle = sum(self.incremental_data - self.data_bkp)
            if abs(self.counts_per_cycle) >= 2300: self.counts_per_cycle = self.counts_per_cycle_bkp
            self.particle_density = self.counts_per_cycle/self.volume

            if (datetime.now()-self.prev_time).total_seconds() >= 1.7: 
                self.time_data.append(self.counts_per_cycle//2)
                self.time_data.append(self.counts_per_cycle//2)
                self.index += 2
            elif (datetime.now()-self.prev_time).total_seconds() <= 1.5: 
                self.time_data.append(self.counts_per_cycle)
                self.index += 1
            self.time_volt.append(self.volt)
            self.time_ram.append(self.buffer)

            if self.counts_per_cycle >= self.counts_treshold:
                self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: red; }")
                self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t WARNING: Concentration is too high, far from single-particle regime.')

            if self.running_label==False: 
                self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: red; }")
                self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t ERROR: Something is wrong with the serial reading from Abakus.')
            if self.volt>=7000: 
                self.lineEdit_volt.setStyleSheet("QLineEdit { background: red; }")
                self.lineEdit_volt_control.setStyleSheet("QLineEdit { background: red; }")
                self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t WARNING: Laser diode voltage is close to 8.0 V, turn off the system and check it.')
            if self.buffer<=2400: 
                self.lineEdit_RAM.setStyleSheet("QLineEdit { background: red; }")
                self.lineEdit_RAM_control.setStyleSheet("QLineEdit { background: red; }")
                self.output_err.append(datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t WARNING: RAM-buffer voltage is lower than 2.4 V, turn off the system and check it.')

            if self.running_label==True and self.counts_per_cycle < self.counts_treshold: self.lineEdit_abakus_run.setStyleSheet("QLineEdit { background: green; }")
            if self.volt<7000:
                self.lineEdit_volt_control.setStyleSheet("QLineEdit { background: green; }")
                self.lineEdit_volt.setStyleSheet("QLineEdit { background: white; }")
            if self.buffer>2400: 
                self.lineEdit_RAM_control.setStyleSheet("QLineEdit { background: green; }")
                self.lineEdit_RAM.setStyleSheet("QLineEdit { background: white; }")

            self.progressbar_readcounts()

            self.saving_txtfile.write('\n'+str(self.index-1)+'\t\t'+'{:.06f}'.format((self.end_time-self.init_time).total_seconds())+'\t\t\t\t'+str(self.volt)+'\t\t\t\t\t\t'+str(self.buffer)+'\t\t\t\t')
            for j in range(1, len(self.meas_data), 2): self.saving_txtfile.write(str(self.meas_data[j])+'\t\t')
            
            self.xlsx_meas_data_list = []
            for i in range(1, len(self.meas_data), 2): self.xlsx_meas_data_list.append(str(self.meas_data[i]))
            self.worksheet.append([str(self.index-1), '{:.06f}'.format((self.end_time-self.init_time).total_seconds()), str(self.volt), str(self.buffer)]+self.xlsx_meas_data_list)

            self.prev_time = datetime.now()
            self.output.append(self.prev_time.strftime("%d-%m-%Y_%H:%M:%S.%f")[11:-7]+'\t\t'+str(self.counts_per_cycle)+' pt\t\t'+str(sum(self.time_data))+' pt') 

            self.curve_single_d.setData(self.channels[1][1:-1], (self.incremental_data - self.data_bkp)[1:-1], stepMode='right')
            self.curve_incremental_d.setData(self.channels[1][1:-1], self.incremental_data[1:-1], stepMode='right')
            self.curve_time1.setData(np.linspace(0, len(self.time_data), len(self.time_data)+1)[:-1], np.array(self.time_data), stepMode='left')
            self.curve_time2.setData(np.linspace(0, len(self.time_data), len(self.time_data)+1)[:-1], np.array(self.time_data), stepMode='left')
            self.curve_volt.setData(np.linspace(0, len(self.time_volt), len(self.time_volt)+1)[:-1], np.array(self.time_volt))
            self.curve_ram.setData(np.linspace(0, len(self.time_ram), len(self.time_ram)+1)[:-1], np.array(self.time_ram))

            legend_single_d = pg.LegendItem((0,0), offset=(910,35))
            legend_single_d.setParentItem(self.single_d_plt.graphicsItem())
            legend_single_d.addItem(self.curve_single_d, '# counts')

            legend_time1 = pg.LegendItem((0,0), offset=(910,35))
            legend_time1.setParentItem(self.time1_plt.graphicsItem())
            legend_time1.addItem(self.curve_time1, '# counts')

            legend_incremental_d = pg.LegendItem((0,0), offset=(910,35))
            legend_incremental_d.setParentItem(self.incremental_d_plt.graphicsItem())
            legend_incremental_d.addItem(self.curve_incremental_d, '# counts')

            legend_time2 = pg.LegendItem((0,0), offset=(910,35))
            legend_time2.setParentItem(self.time2_plt.graphicsItem())
            legend_time2.addItem(self.curve_time2, '# counts')
            
            legend_volt = pg.LegendItem((0,0), offset=(820,300))
            legend_volt.setParentItem(self.volt_plt.graphicsItem())
            legend_volt.addItem(self.curve_volt, 'LASER diode voltage')
            legend_volt.addItem(self.curve_ram, 'RAM-buffer voltage')

            if self.print_on_terminal==True: print('\n\n\n\n', self.index, '\t', self.volt, '\t', self.buffer, '\t', (self.end_time-self.init_time).total_seconds(), '\t',  self.meas_data, '\n')
            if self.print_on_terminal==True: print(self.counts_sum) 

            self.data_bkp = self.incremental_data
            self.counts_per_cycle_bkp = self.counts_per_cycle
            self.volt_bkp, self.buffer_bkp, self.meas_data_bkp, self.init_time_bkp, self.end_time_bkp, self.running_label_bkp = self.volt, self.buffer, self.meas_data, self.init_time, self.end_time, self.running_label
            self.counts_per_cycle, self.particle_density = 0, 0


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def progressbar_readcounts(self):

        self.progressbar.setValue(int(round(1000*self.counts_per_cycle//self.counts_treshold)))
        if self.progressbar.value() >= 0 and self.progressbar.value() < 10: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #00de60; }")
        if self.progressbar.value() >= 10 and self.progressbar.value() < 20: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #00de25; }")
        if self.progressbar.value() >= 20 and self.progressbar.value() < 30: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #55de00; }")
        if self.progressbar.value() >= 30 and self.progressbar.value() < 40: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #94de00; }")
        if self.progressbar.value() >= 40 and self.progressbar.value() < 50: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #d3de00; }")
        if self.progressbar.value() >= 50 and self.progressbar.value() < 60: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #dec400; }")
        if self.progressbar.value() >= 60 and self.progressbar.value() < 70: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #dea600; }")
        if self.progressbar.value() >= 70 and self.progressbar.value() < 80: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #de8500; }")
        if self.progressbar.value() >= 80 and self.progressbar.value() < 90: self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #de6800; }")
        if self.progressbar.value() >= 90 and self.progressbar.value() < self.progressbar.maximum(): self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #de2c00; }")
        if self.progressbar.value() >= self.progressbar.maximum(): self.progressbar.setStyleSheet("QProgressBar::chunk { background-color: #990000; }")
    

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def on_save_clicked(self):

        if self.live_measurement==False: save_path = self.lineEdit_save_path.text()
        if self.live_measurement==True: save_path = self.lineEdit_directory_path.text()
        save_name = self.lineEdit_save_name.text()

        self.save_file(save_path, save_name)
        self.save_image(save_path, save_name)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def save_file(self, save_path, save_name):

        if self.live_measurement == False:

            if self.update_label == False:

                self.full_path = f"{save_path}/{self.time_str[:-12]}/"
                if os.path.isdir(self.full_path): print("")
                else: os.makedirs(self.full_path)

                save_name = save_name+'_'+self.time_str[11:-3]
                file = open(self.full_path+save_name+'.txt', 'w')

                file.write('%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%\nSamples and results from Abakus laser sensor --- CFA YETI, Continuous FLow Analysis measurement\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')
                file.write('\n\n\nAcquisition time:\t\t\t\t\t\t'+str(self.acq_time)+' s\n')
                for k in range(0, len(self.filess)):
                    file.write('\n\n%--------------------------------------------------------------------------------------------------------------%\nFile: '+"'"+self.filess[k]+"'"+'\n%--------------------------------------------------------------------------------------------------------------%')
                    file.write('\n\nAverage laser diode voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.volt1))+' mV')  
                    file.write('\nAvergae RAM-buffer voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.RAM1))+' mV\n')  
                    file.write('\n\nFlow rate:\t\t\t\t\t'+str(self.flow1)+' mL/min')  
                    file.write('\nParticles detected:\t\t\t\t'+'{:.2e}'.format(sum(self.data1.sum(axis=0)))+' pt')
                    file.write('\nTotal particles concentration:\t\t\t'+'{:.2e}'.format(self.ptc_conc1)+' pt/mL')
                    file.write('\nCounts distribution peaked @:\t\t\t'+'{:.02f}'.format(self.sizes[np.where(self.h1[:-1]==np.amax(self.h1[:-1]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' um (as a function of particle diameters)')
                    file.write('\nCounts distribution average:\t\t\t'+'{:.02f}'.format(np.average(self.sizes[:-1], weights=self.h1[:-1]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.h1[:-1]**2))/sum(self.h1[:-1]))+' um')
                    file.write('\nCounts distribution average (arithmetical):\t'+'{:.02f}'.format(np.mean(self.sizes[:-1]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.sizes[:-1])))+' um')
                    file.write('\nTime-average # counts:\t\t\t\t'+'{:.02f}'.format(np.mean(np.array(self.data1.sum(axis=1)))))
                    file.write('\nTime std. deviation # counts:\t\t\t'+'{:.02f}'.format(np.sqrt(np.var(np.array(self.data1.sum(axis=1))))))
                    file.write('\nTime-median # counts:\t\t\t\t'+'{:.02f}'.format(np.median(np.array(self.data1.sum(axis=1)))))
                    file.write('\nFirst quantile # counts (in time):\t\t'+'{:.02f}'.format(np.quantile(np.array(self.data1.sum(axis=1)), 0.25)))
                    file.write('\nThird quantile # counts (in time):\t\t'+'{:.02f}'.format(np.quantile(np.array(self.data1.sum(axis=1)), 0.75))+'\n\n')
                    for i in range(0, len(self.sizes)): file.write('Particles concentration @ '+str(self.sizes[i])+'\t[mm]:\t'+'{:.2e}'.format(self.ptc_conc_channel1[i][1])+' pt/mL\n')
                file.write('\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')

                file.close()
                
                if self.print_on_terminal==True: print(f"\nFile saved as '{self.full_path}{save_name}.txt'\n")
                self.output.append(f"\nFile saved as '{self.full_path}{save_name}.txt'\n")

            if self.update_label == True:

                self.full_path = f"{save_path}/{self.time_str[:-12]}/"
                if os.path.isdir(self.full_path): print("")
                else: os.makedirs(self.full_path)

                save_name = save_name+'_d_'+'{:.01f}'.format(self.d_range[0])+'_'+'{:.01f}'.format(self.d_range[1])+'µm'+'_'+self.time_str[11:-3]
                file = open(self.full_path+save_name+'.txt', 'w')

                file.write('%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%\nSamples and results from Abakus laser sensor --- CFA YETI, Continuous FLow Analysis measurement\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')
                file.write('\n\n\nAcquisition time:\t\t\t\t\t\t'+str(self.acq_time)+' s\n')
                for k in range(0, len(self.filess)):
                    file.write('\n\n%--------------------------------------------------------------------------------------------------------------%\nFile: '+"'"+self.filess[k]+"'"+'  ------  restricted on x axis (diameter, extinction cross-section and scattering amplitude)\n%--------------------------------------------------------------------------------------------------------------%')
                    file.write('\n\nAverage laser diode voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.volt1))+' mV')  
                    file.write('\nAvergae RAM-buffer voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.RAM1))+' mV')  
                    file.write('\n\nFlow rate:\t\t\t\t\t'+str(self.flow1)+' mL/min')  
                    file.write('\nParticles detected:\t\t\t\t'+'{:.2e}'.format(sum(self.h1[self.indexes]))+' pt')
                    file.write('\n\nCounts distribution peaked @:\t\t\t'+'{:.02f}'.format(self.sizes[np.where(self.h1==np.amax(self.h1[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' um')
                    file.write('\nCounts distribution average:\t\t\t'+'{:.02f}'.format(np.average(self.sizes[self.indexes], weights=self.h1[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.h1[:-1]**2))/sum(self.h1[:-1]))+' um')
                    file.write('\nCounts distribution average (arithmetical):\t'+'{:.02f}'.format(np.mean(self.sizes[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.sizes[:-1])))+' um')
                    file.write('\nCounts distribution std. deviation:\t\t'+'{:.02f}'.format(np.sqrt(np.var((self.sizes[self.indexes]))))+' um')
                    file.write('\nFirst quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.25))+' um')
                    file.write('\nSecond quantile # counts:\t\t\t'+'{:.02f}'.format(np.median(self.sizes[self.indexes]))+' um')
                    file.write('\nThird quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.75))+' um\n')
                file.write('\n--> Complete results are saved in:\n    '+self.full_path+save_name[:-12]+'.txt\n')
                file.write('\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')

                file.close()

                if self.print_on_terminal==True: print(f"\nUpdated data saved as '{self.full_path}{save_name}.txt'\n")
                self.output.append(f"\nUpdated data saved as '{self.full_path}{save_name}.txt'\n")

            if self.calibration_label == True and self.update_label==False:

                self.full_path = f"{save_path}/{self.time_str[:-12]}/"
                if os.path.isdir(self.full_path): print("")
                else: os.makedirs(self.full_path)

                save_name_cal = save_name+'_extinction_corrected'+'_'+self.time_str[11:-3]
                file_cal = open(self.full_path+save_name_cal+'.txt', 'w')

                file_cal.write('%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%\nSamples and results from Abakus laser sensor --- CFA YETI, Continuous FLow Analysis measurement\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')
                file_cal.write('\n\n\nAcquisition time:\t\t\t\t\t\t'+str(self.acq_time)+' s\n')
                for k in range(0, len(self.filess)):
                    file_cal.write('\n\n%--------------------------------------------------------------------------------------------------------------%\nFile: '+"'"+self.filess[k]+"'"+'  ------  restricted on x axis (diameter, extinction cross-section and scattering amplitude)\n%--------------------------------------------------------------------------------------------------------------%')
                    file_cal.write('\n\nAverage laser diode voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.volt1))+' mV')  
                    file_cal.write('\nAvergae RAM-buffer voltage:\t\t\t'+'{:.01f}'.format(np.mean(self.RAM1))+' mV')  
                    file_cal.write('\n\nFlow rate:\t\t\t\t\t'+str(self.flow1)+' mL/min')  
                    file_cal.write('\nParticles detected:\t\t\t\t'+'{:.2e}'.format(sum(self.h1[self.indexes]))+' pt')
                    file_cal.write('\n\nCounts distribution peaked @:\t\t\t'+'{:.02f}'.format(self.sizes[np.where(self.h1==np.amax(self.h1[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' um')
                    file_cal.write('\nCounts distribution average:\t\t\t'+'{:.02f}'.format(np.average(self.sizes[self.indexes], weights=self.h1[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.h1[:-1]**2))/sum(self.h1[:-1]))+' um')
                    file_cal.write('\nCounts distribution average (arithmetical):\t\t'+'{:.02f}'.format(np.mean(self.sizes[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.sizes[:-1])))+' um')
                    file_cal.write('\nCounts distribution std. deviation:\t'+'{:.02f}'.format(np.sqrt(np.var((self.sizes[self.indexes]))))+' um')
                    file_cal.write('\nFirst quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.25))+' um')
                    file_cal.write('\nSecond quantile # counts:\t\t\t'+'{:.02f}'.format(np.median(self.sizes[self.indexes]))+' um')
                    file_cal.write('\nThird quantile # counts:\t\t\t'+'{:.02f}'.format(np.quantile(self.sizes[self.indexes], 0.75))+' um\n')
                file_cal.write('\n--> Complete results are saved in:\n    '+self.full_path+save_name[:-12]+'.txt\n')
                file_cal.write('\n%--------------------------------------------------------------------------------------------------------------%\n%--------------------------------------------------------------------------------------------------------------%')

                file_cal.close()

                if self.print_on_terminal==True: print(f"\nExtinction-corrected data saved as '{self.full_path}{save_name_cal}.txt'\n")
                self.output.append(f"\nExtinction-corrected data saved as '{self.full_path}{save_name_cal}.txt'\n")


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def save_image(self, save_path, save_name):

        self.full_path = f"{save_path}/{self.time_str[:-12]}/Images/"
        if os.path.isdir(self.full_path): print("")
        else: os.makedirs(self.full_path)

        exporter_d = pg.exporters.ImageExporter(self.single_d_plt.scene())
        exporter_d.parameters()['width'] = self.image_width
        exporter_d.export(self.full_path+save_name+'_hist_d_'+self.time_str[11:-3]+'.png')

        exporter_time = pg.exporters.ImageExporter(self.incremental_d_plt.scene())
        exporter_time.parameters()['width'] = self.image_width
        exporter_time.export(self.full_path+save_name+'_hist_time_'+self.time_str[11:-3]+'.png')

        exporter_volt = pg.exporters.ImageExporter(self.volt_plt.scene())
        exporter_volt.parameters()['width'] = self.image_width
        exporter_volt.export(self.full_path+save_name+'_hist_volt_'+self.time_str[11:-3]+'.png')

        if self.print_on_terminal==True: print(f"\nImages saved\n")
        self.output.append(f"\nImages saved\n")


############################################################################################################################################################
############################################################################################################################################################


app = QtWidgets.QApplication(sys.argv)                                                                  # Run the application
style = """
        QTabWidget {
            font: bold 11px;
        }
"""
app.setStyle('Fusion')
app.setStyleSheet(style)
window = Ui(_PATH)                                                                                      # Definition of a 'CFA_mainpanel' element
app.exec_()                                                                                             # Python script execution


############################################################################################################################################################
############################################################################################################################################################
