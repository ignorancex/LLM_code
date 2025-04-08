############################################################################################################################################################
############################################################################################################################################################

# Program name: data_correction.py
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 22 June 2022 (last modified)
# Objective: Class definition: Abakus data correction and interpretation
# Description: This Python3 script defines a class for Abakus size distribution calibration, correction and, generally speaking, interpretation.
#              In particular, three main kind of data analysis have been implemented:
#              - size distribution calibration: the size distribution is corrected over the instrumental extinction calibration curve that can be obtained from the file
#                calibration.py.
#                This calibration function have been computed from several measurements made with suspensions of well known polystyrene particle with fixed refractive index 
#                and calibrated diameters (from 1.0 um to 5.45 um approximately). 
#              - refractive index correction: this is the key point of the whole analysis software and allows the user to appropriately 
#                correct the measured size distribution considering different possible refractive index values.
#                The fact is that the Abakus laser sensor physical calibration is based on spherical polystirene nanoparticles with fixed refractive index (about 1.58 @
#                670 nm) and this leads to uncertainties and alterations in the measured particle size distribution. This method allows the user to set the 
#                refractive index value arbitrarily between 1.3311 and 2.1000 according to the sample effective composition, thus improving the reliability of
#                the Abakus results.
#                The most suitable refractive index to operate this correction may be retrieved independently with other analyses and techniques.
#              - aspect ratio correction: this correction will take into account also the effect of particles non-sphericity, considering aspect ratios different from 1.
#                It is still under development...

############################################################################################################################################################
############################################################################################################################################################


import os, pyqtgraph as pg, math as m, numpy as np, time, sys                                           # Import required libraries
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qtwidgets import *

_PATH = os.path.abspath(os.path.realpath(__file__))[2:-26].replace('\\', '/')


############################################################################################################################################################
############################################################################################################################################################


class Data_corrector(QtWidgets.QMainWindow, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Class constructor: creates a Data Corrector object
    
    def __init__(self, wavelength):
        
        super(Data_corrector, self).__init__()
        uic.loadUi(_PATH+'subGUIs/data_correction.ui', self)                                            # Load the graphical interface
        self.setWindowIcon(QIcon(_PATH+'_icon/data_correction.png'))
        
        self.wavelength = wavelength                                                                    # Set the laser wavelength value

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Physical and graphical parameters settings.
    # The three main options for Abakus data correction are set: calibration based on instrumental calibration curve (the 'extinction' one, calculated by analyzing polystyrene spheres
    # with calibrated diameter and known refractive index), interpretation based on a user-defined refractive index and correction considering the effect of the particle aspcet ratio
    # (in case of non-spherical particles, still work in progress). 
    
    def settings(self):

        self.connect_widgets()

        self.print_on_terminal = False
        self.ref_index_correction_label = False
        self.aspect_ratio_correction_label = False
        self.run_correction_label = False

        self.k = 2*m.pi/self.wavelength                                                                 # Wavenumber

        self.abakus_logo = QPixmap(_PATH+'_icon/abakus_pixmap.png')                                     # Abakus icon

        self.label_progbar.setText("<b>Progress Bar") 

        self.toggle_refractive_index = AnimatedToggle(checked_color="#028a0f", pulse_checked_color="#03c04a")
        self.toggle_refractive_index.setFixedWidth(90)                                                  # Button for activate/inactivate data correction based on user-defined refrative index
        self.toggle_refractive_index.setFixedHeight(40)
        self.grid_ref_index.addWidget(self.toggle_refractive_index)
        self.toggle_refractive_index.clicked.connect(self.on_ref_index_clicked)

        self.toggle_aspect_ratio = AnimatedToggle(checked_color="#028a0f", pulse_checked_color="#03c04a")
        self.toggle_aspect_ratio.setFixedWidth(90)                                                      # Button for activate/inactivate data correction considering the particle aspect ratio
        self.toggle_aspect_ratio.setFixedHeight(40)
        self.grid_aspect_ratio.addWidget(self.toggle_aspect_ratio)
        self.toggle_aspect_ratio.clicked.connect(self.on_aspect_ratio_clicked)

        self.label_aspectratio_description.setFont(QFont('Arial', 8))
        self.combobox_ref_index_RE.setFixedWidth(70)
        self.combobox_ref_index_IM.setFixedWidth(70)
        self.lineEdit_aspect_ratio.setFixedWidth(70)

        self.combobox_ref_index_RE.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.combobox_ref_index_IM.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.combobox_ref_index_RE.addItems(['1.5848']+['{:.4f}'.format(0.0001*i) for i in range(13311, 26000)]) 
        self.combobox_ref_index_IM.addItems(['{:.4f}'.format(0.1*i) for i in range(0, 11)])             # Real and imaginary parts of the refractive index set by the user
        self.lineEdit_aspect_ratio.setText('1.0000')                                                    # Aspect ratio default value (spherical approximation)

        self.progressbar = QProgressBar(self.groupBox_data)                                             # Progress bar creation
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.move(10, 190)
        self.progressbar.setFixedHeight(50)
        self.progressbar.setFixedWidth(250)
        self.progressbar.setTextVisible(True)

        self.label_ref_index.setText("Refractive index correction:"+" <b>OFF") 
        self.label_aspect_ratio.setText("Aspect-ratio correction  [0, 1]:"+" <b>OFF")

        self.btn_run.setStyleSheet("QPushButton { font: bold 11px; }")
        self.btn_stop.setStyleSheet("QPushButton { font: bold 11px; }")


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method to connect button(s) with the relative function(s).
        
    def connect_widgets(self):

        self.btn_stop.clicked.connect(self.on_stop_correction)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method to set the Abakus icon inside the GUI software.

    def paintEvent(self, event):

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(QPoint(387, 190), self.abakus_logo.scaled(108, 50, Qt.IgnoreAspectRatio, transformMode = Qt.SmoothTransformation))


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Function to set the data compensation based on the user-defined refractive index according to the corresponding button.
        
    def on_ref_index_clicked(self):

        if self.toggle_refractive_index.isChecked(): 
            self.ref_index_correction_label = True                                                      # If the button is 'ON', the label is set to TRUE and 
            self.label_ref_index.setText("Refractive index correction:"+" <b>ON")                       # the correction is performed
        else: 
            self.ref_index_correction_label = False                                                     # If the button is 'OFF', the label is set to FALSE and 
            self.label_ref_index.setText("Refractive index correction:"+" <b>OFF")                      # the corrction is not performed


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Function to set the data correction considering the effect of non-sperical particles (-> aspect ratio different from 1) according to the corresponding button.
    # ATTENTION: the procedure for aspect ratio correction is currently under construction, hence this option is not working yet.
        
    def on_aspect_ratio_clicked(self):

        if self.toggle_aspect_ratio.isChecked(): 
            self.aspect_ratio_correction_label = True                                                   # If the button is 'ON', the label is set to TRUE and 
            self.label_aspect_ratio.setText("Aspect-ratio correction  [0, 1]:"+" <b>ON")                # the correction is performed
        else: 
            self.aspect_ratio_correction_label = False                                                  # If the button is 'OFF', the label is set to FALSE and 
            self.label_aspect_ratio.setText("Aspect-ratio correction  [0, 1]:"+" <b>OFF")               # the correction is not performed


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Method to stop the program execution.
        
    def on_stop_correction(self):

        if self.btn_stop.isChecked(): self.run_correction_label = False
        time.sleep(2)
        sys.exit()


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # This method defines the Abakus data correction and interpretation on the basis of any refractive index set by the user, different from the polystyrene one
    # with which the instrument is calibrated.
    # Data for this kind of analysis have been computed previously and independently for multipe refractive index values (ranging from 1.3311 to 2.1000) and loaded 
    # from a LUT (look-up table). Particles diameters range from 0.2 um to 20 um.
    # According to the refractive index real and imaginary part set by the user, assuming that the particles are suspended in water with refractive index 1.3310 @ 670 nm, 
    # the corresponding extinction cross-section curve is selected from the LUT and used for data inversion.
        
    
    def refractive_index_calibration_correction(self, sizes):

        def f(x, *p): return np.poly1d(p)(x)

        t = datetime.now()

        self.ref_index_Re = float(self.combobox_ref_index_RE.currentText())                                 # Set the refractive index real and imaginary part 
        self.ref_index_Im = float(self.combobox_ref_index_IM.currentText())
        self.n_med = 1.3310                                                                                 # Set the water refractive index
        h = 0

        if self.ref_index_Im != 0: file = open(_PATH+'/LUT_Cext/LUT_Cext_l='+'{:.02f}'.format(self.wavelength)+'um_nmed='+'{:.04f}'.format(self.n_med)+'_m=[1.0001+'+'{:.04f}'.format(self.ref_index_Im)+'j-1.9534+'+'{:.04f}'.format(self.ref_index_Im)+'j].txt', 'r')
        else: file = open(_PATH+'/LUT_Cext/LUT_Cext_l='+'{:.02f}'.format(self.wavelength)+'um_nmed='+'{:.04f}'.format(self.n_med)+'_m=[1.0001-1.9534].txt', 'r')

        self.m_polystirene = np.round(1.5848/self.n_med, 4)                                                 # Polystirene relative refractive index, rounded to the 4th decimal value
        
        self.m = np.round(self.ref_index_Re/self.n_med, 4)                                                  # Relative refractive index, rounded to the 4th decimal value
        
        self.diameters_Cext, self.m_Cext, lines_Cext, self.Cext = [], [], file.readlines(), []              # The first row is taken apart since it contains
        self.diameters_Cext.append(np.array([float(i) for i in lines_Cext[0].split('\t')[2:] if i.strip()]))# the particle diameters; the other ones are converted to float and 
        self.diameters_Cext = self.diameters_Cext[0]                                                        # appended to the corresponding list
        for x in lines_Cext[1:]: 
            if self.btn_stop.isChecked(): break
            self.Cext.append(np.array([complex(i) for i in x.split('\t') if i.strip()]))
            self.progressbar.setValue((h*100+1)/len(lines_Cext))
            h += 1
        
        for j in range(0, len(self.Cext)): 
            self.m_Cext.append(self.Cext[j][0])
            self.Cext[j] = np.real(self.Cext[j][1:])

        diameters_idx = []
        for i in range(0, len(sizes)): diameters_idx.append(np.where(self.diameters_Cext==round(sizes[i]-0.70, 2))[0][0])

        polystirene_idx = np.where(np.real(self.m_Cext)==self.m_polystirene.real)[0]                        # Find when the row corresponding to polystirene refractive index 
 
        idx = np.where(np.real(self.m_Cext)==self.m.real)[0]                                                # Find when the experimental refractive index is equal to some 
        if len(idx) > 1:                                                                                    # value ammong the LUT ones
            self.selected_Cext = 0                                                                          # If more than one is found, the average Cext is computed
            for i in range(0, len(idx)): 
                if self.btn_stop.isChecked(): break
                self.selected_Cext += self.Cext[idx[i]]  
            self.selected_Cext = self.selected_Cext/len(idx)
        else: 
            self.selected_Cext = self.Cext[idx[0]]
        idx = 0

        self.selected_Cext_cfr = self.selected_Cext[diameters_idx]

        self.n_range = np.array([1.42, 1.46, 1.47, 1.48, 1.50, 1.51, 1.52, 1.53, 1.56, 1.58, 1.64])
        self.s_range = np.array([100, 125, 150, 150, 150, 150, 150, 125, 125, 100, 100])
        self.poly_coefficients = np.polyfit(self.n_range, self.s_range, 3)
        self._poly_fit = np.poly1d(self.poly_coefficients)

        self.size_avg = self._poly_fit(1.58)

        self.poly_fit = interp1d(uniform_filter1d(self.selected_Cext, size=int(self.size_avg)), self.diameters_Cext, kind='linear', fill_value='extrapolate') 

        true_pos = np.array([1.0, 1.8, 2.9, 3.7, 5, 10])
        false_pos = np.array([1.05, 2.5, 3.7, 4.1, 5.8, 10])
        false_pos_dev = np.array([0.1, 0.3, 0.2, 0.3, 0.3, 0.2])
        selected_Cext_interp = interp1d(self.poly_fit(self.selected_Cext), self.selected_Cext, kind='linear', fill_value='extrapolate')

        sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        p1, _ = curve_fit(f, false_pos, selected_Cext_interp(true_pos), (0, 0, 0, 0, 0), sigma=sigma)

        self.cal_curve = interp1d(self.diameters_Cext, f(self.diameters_Cext, *p1), kind='linear', fill_value='extrapolate')

        self.Cext_polystirene = self.Cext[polystirene_idx[0]]
        self.Cext_polystirene_cfr = self.cal_curve(sizes)
        self.Cext_polystirene_cfr[self.Cext_polystirene_cfr<=0.0001] = 0.0001
        self.sizes_RI_idx, self.sizes_RI = [], np.zeros(len(sizes))
        self.interp_xaxis = np.arange(self.selected_Cext[0], self.selected_Cext[-1], 0.001)
        for j in range(0, len(self.Cext_polystirene_cfr)):                                                  # Data correction
            if self.btn_stop.isChecked(): break
            idx = np.where(abs(self.interp_xaxis - self.Cext_polystirene_cfr[j]) < 0.001)[0]
            if len(idx) > 1:
                for i in range(0, len(idx)): 
                    a = self.poly_fit(self.interp_xaxis[idx[i]])
                    self.sizes_RI[j] += a
                self.sizes_RI[j] = self.sizes_RI[j]/len(idx)
            else: 
                aa =  self.poly_fit(self.interp_xaxis[idx[0]])
                self.sizes_RI[j] = aa

        self.sizes_RI[self.sizes_RI<=0.] = 0.5

        return self.sizes_RI, self.ref_index_Re, self.ref_index_Im, self.diameters_Cext, self.Cext_polystirene, self.selected_Cext, self.poly_fit


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # ---- STILL WORK IN PROGESS ----
        
    def aspect_ratio_calibration_correction(self, sizes):

        return


############################################################################################################################################################
############################################################################################################################################################
