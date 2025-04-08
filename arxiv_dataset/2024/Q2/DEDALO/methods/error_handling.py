############################################################################################################################################################
############################################################################################################################################################

# Program name: error_handling.py
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 30 November 2022 (last modified)
# Objective: Error handling module
# Description: This Python3 script defines a class for handling errors and warnings which may occur during the software running.

############################################################################################################################################################
############################################################################################################################################################


import sys, os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

_PATH = os.path.abspath(os.path.realpath(__file__))[2:-25].replace('\\', '/')


############################################################################################################################################################
############################################################################################################################################################


class Error_Handling(QMainWindow, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#

    def __init__(self, log_file):

        self.file = log_file
        self.btn_height = 50
        self.btn_width = 250
        self.window_height = 200
        self.window_width = 710
        result = []

        super(Error_Handling, self).__init__()
        self.setWindowTitle("DEDALO Error handling window")
        self.setWindowIcon(QIcon(_PATH+'_icon/error.png'))
        self.setFixedSize(QSize(self.window_width, self.window_height))

        self.widget = QTabWidget()
        self.setCentralWidget(self.widget)

        self.centralbox = QGroupBox(self.widget)
        self.centralbox.setGeometry(QRect(0, 0, self.window_width, self.window_height))

        file = open(_PATH+self.file, 'r')
        lines = file.readlines()                                                                        # Open text file and read rows
        for x in lines: result.append(x.split('\n'))                                                    # Split values according to the separator ','
        file.close()
        
        self.line1 = QLabel('<b>An error occurred! Help! Help!', self.centralbox)
        self.line1.move(30, 30)
        try:
            self.line2 = QLabel(result[-4][0], self.centralbox)
            self.line2.move(30, 60)
        except: pass
        try:
            self.line3 = QLabel(result[-3][0][2:], self.centralbox)
            self.line3.move(30, 90)
        except: pass
        try:
            self.line4 = QLabel(result[-2][0][4:], self.centralbox)
            self.line4.move(30, 120)
        except: pass
        try:
            self.line5 = QLabel(result[-1][0], self.centralbox)
            self.line5.setStyleSheet('QLabel { color: red; }')
            self.line5.move(30, 150)
        except: pass


############################################################################################################################################################
############################################################################################################################################################


app = QApplication(sys.argv)
app.setStyle('Fusion')
ex = Error_Handling(sys.argv[1])
ex.show()
sys.exit(app.exec_())


############################################################################################################################################################
############################################################################################################################################################
