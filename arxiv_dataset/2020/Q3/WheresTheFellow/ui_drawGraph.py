# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'drawGraph.ui',
# licensing of 'drawGraph.ui' applies.
#
# Created: Fri Feb 28 15:14:07 2020
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_CalibrationWidget(object):
    def setupUi(self, CalibrationWidget):
        CalibrationWidget.setObjectName("Graph2ImageWidget")
        CalibrationWidget.resize(640, 557)
        self.horizontalLayout = QtWidgets.QHBoxLayout(CalibrationWidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget = QtWidgets.QWidget(CalibrationWidget)
        self.widget.setMinimumSize(QtCore.QSize(200, 0))
        self.widget.setObjectName("widget")
        self.horizontalLayout.addWidget(self.widget)
        self.tableWidget = QtWidgets.QTableWidget(CalibrationWidget)
        self.tableWidget.setMinimumSize(QtCore.QSize(400, 0))
        self.tableWidget.setStyleSheet("font: 15pt \"Ubuntu\";")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.horizontalLayout.addWidget(self.tableWidget)

        self.retranslateUi(CalibrationWidget)
        QtCore.QMetaObject.connectSlotsByName(CalibrationWidget)

    def retranslateUi(self, CalibrationWidget):
        CalibrationWidget.setWindowTitle(QtWidgets.QApplication.translate("CalibrationWidget", "Calibration graph inspector", None, -1))

