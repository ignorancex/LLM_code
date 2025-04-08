############################################################################################################################################################
############################################################################################################################################################

# Program name: serial_monitor.py
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 01 May 2022 (last modified)
# Objective: Embedded serial monitor
# Description: This script defines a class for monitoring the serial communication between the Abakus laser sensor and the user pc.
#              The serial monitor has a user friendly GUI interface to facilitate this task.
#              After setting automatically the serial communication paraneters (baudrate, bytesize, parity, stopbits and timeout), the user can both
#              monitor passively the serial communication and sendig specific commands as well checking the instrument output.

############################################################################################################################################################
############################################################################################################################################################


import sys, serial, serial.tools.list_ports, os, time                                                   # Import the required libraries
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import *


############################################################################################################################################################
############################################################################################################################################################


class SerialMonitor(QtWidgets.QMainWindow, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Innitialization of the serial commmunication and the graphical user interface (GUI) of the serial terminal.
    
    def __init__(self, serial_port, path, baudrate = 38400, bytesize = 8, parity = 'N', stopbits = 1, timeout = 0.1, debug = True):
        
        super(SerialMonitor, self).__init__()
        uic.loadUi(path+'/subGUIs/serial_monitor.ui', self)                                             # Graphical user interface loading
        self.setWindowIcon(QIcon(path+'/_icon/serial.png'))

        self.serial_port = serial_port                                                                  # Setting the Anakus serial port
        self.debug = debug                                                                              # Debug option
        self._dev = serial.Serial(self.serial_port, baudrate, bytesize, parity, stopbits, timeout)      # Initializing the serial communication with the specificed serial
                                                                                                        # communication parameters: baudrate, timeout, parity, stopbits and bytesize

        self.btn_send.clicked.connect(self.on_send_clicked)                                             # If the "Send" button is clicked, the user-specificed serial 
                                                                                                        # command is sent
        self.btn_close.clicked.connect(self.on_close_clicked)                                           # If the "Send" button is clicked, the serial mponitor
                                                                                                        # stops and the COM port is closed
        self.lineEdit_port.setText(self.serial_port)

        self.output = QtWidgets.QTextBrowser(self.console)
        self.output.setGeometry(QtCore.QRect(10, 20, 680, 545))
        self.output.setObjectName("output")                                                             # Defining the GUI output window
        self.output.setStyleSheet("QTextBrowser { background: black; color: green; }")

        self.output.append("<b>Connected to "+self.serial_port+" serial port, Abakus laser sensor reading continuously.\n")

        self.btn_send.setStyleSheet("QPushButton { color: green; font: bold 11px; }")
        self.btn_close.setStyleSheet("QPushButton { color: red; font: bold 11px; }")
        self.groupBox.setStyleSheet("QGroupBox { font: bold 11px; }")
        self.console.setStyleSheet("QGroupBox { font: bold 11px; }")

        self.timer = QTimer()                                                                           # Starting the continuous (passive) serial monitoring

        if self.btn_send.isChecked(): self.on_send_clicked()
        else:
            self.timer.timeout.connect(self.readData)
            self.timer.start(1000)

        self.show()
 
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method for sendig to the Abakus laser sensor the serial command set by the user in the GUI interface when the "Send" button is clicked.

    def on_send_clicked(self):
    
        self.cmd = self.lineEdit_cmd.text()                                                             # Retrieving the serial command
        self.output.append('<b>'+QDateTime.currentDateTime().toString('hh:mm:ss')+'\t'+' >>  '+self.cmd)
        self.sendData()                                                                                 # Sending the command to the Abakus laser sensor


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # 

    def sendData(self):

        if self.cmd != "":                                                                              # If a commmand is sent by the user, it is appropriately
            self._dev.write(b'U0003\n')                                                                 # encoded for serial commmunication
            time.sleep(1)
            answer = self._dev.readline().decode('utf-8')

            self.output.append('<b>'+QDateTime.currentDateTime().toString('hh:mm:ss')+'\t <<  '+answer)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method for reading the serial output from the Abakus laser sensor and visualize it in the GUI iutput window. This task is exectued each second.

    def readData(self):

        try: 
            answer = self._dev.readline().decode('utf-8')                                               # Print output on the serial mmonitor
            self.output.append('<b>'+QDateTime.currentDateTime().toString('hh:mm:ss')+'\t<b> <<  '+answer)
        
        except: print('')


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method for closing both the serial monitor and the connected serial port at the end of the control.
    
    def on_close_clicked(self):

        self._dev.close()


############################################################################################################################################################
############################################################################################################################################################


app = QtWidgets.QApplication(sys.argv)                                                                  # Run the application
window = SerialMonitor(sys.argv[1], sys.argv[2])                                                        # Definition of a 'SerialMonitor' object
app.exec_()                                                                                             # Python script execution


############################################################################################################################################################
############################################################################################################################################################
