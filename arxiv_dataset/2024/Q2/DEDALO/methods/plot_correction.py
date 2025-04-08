############################################################################################################################################################
############################################################################################################################################################

# Program name: plot_correction.py
# Author: Luca Teruzzi, PhD in Physics, Astrophysics and Applied Physics, University of Milan, Milan (IT)
#         EuroCold Lab, Department of Earth and Environmental Sciences, University of Milano-Bicocca, Milan (IT)
# Date: 01 May 2022 (last modified)
# Objective: Embedded serial monitor
# Description: This script defines a graphical environment for size distribution visualization and corrections.
#              All the required input parameters and data are provided by the main programme.
#              In particular, the user can see four different kind of size distribution:
#              - 'raw' size distribution: the size distribution as measured by the Abakus laser sensor, without any kind of correction or calibration.
#              - 'calibrated' size distribution: the size distribution corrected over the instrumental calibration curve that can be obtained from the file
#                calibration.py.
#                The importance of the calibration curve is better discussed in the python file mentioned above.
#              - '(refractive index)-corrected' size distribution: this is the key point of the whole analysis software and allows the user to appropriately 
#                correct the measured size distribution considering different possible refractive index values.
#                The fact is that the Abakus laser sensor physical calibration is based on spherical polystirene nanoparticles with fixed refractive index (about 1.58 @
#                670 nm) and this leads to uncertainties and alterations in the measured particle size distribution. This method allows the user to set the 
#                refractive index value arbitrarily between 1.3311 and 2.1000 according to the sample effective composition, thus improving the reliability of
#                the Abakus results.
#                The most suitable refractive index to operate this correction may be retrieved independently with other analyses and techniques.
#              - '(aspect ratio)-corrected' size distribution: still under development...

############################################################################################################################################################
############################################################################################################################################################


import numpy as np, math as m, pyqtgraph as pg, os                                                      # Import the required libraries
import pyqtgraph.exporters
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

pg.setConfigOption('background', 'w')                                                                   # Set the background color (white) and the text color (black)
pg.setConfigOption('foreground', 'k')

_PATH = os.path.abspath(os.path.realpath(__file__))[2:-26].replace('\\', '/')


############################################################################################################################################################
############################################################################################################################################################


class CData_Plotter(QtWidgets.QMainWindow, object):

    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    # Class initialization and size distributions visualization, according to the provided input values.
    # The main window is divided into four sections referring to the possibile size distribution corrections mentioned above.
    
    def __init__(self, x_data, y_data, y_time_data, full_dataset, ref_index_Re, ref_index_Im, diameters_Cext, Cext_polystirene, selected_Cext, poly_fit, save_path, save_name, labels_correct):
        
        super(CData_Plotter, self).__init__()
        self.setWindowTitle('DEDALO - Device for Enhanced Dust Analysis with Light Obscuration sensors')# GUI interface title
        self.setFixedSize(QSize(1920, 1040))                                                            # Interface sizes
        self.setWindowIcon(QIcon(_PATH+'_icon/abakus_logo_tPk_5.ico'))

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        self.widget = QWidget()
        self.setCentralWidget(self.widget)

        self.print_on_terminal = False                                                                  # Option for printing results and messages in terminal
        self.x_data = x_data                                                                            # Setting the plot x axis = particle diameters (raw and corrected ones)
        self.ref_index_RE = ref_index_Re                                                                # Setting the real and imaginary part of the user defined refractive index
        self.ref_index_IM = ref_index_Im
        self.diameters_Cext = diameters_Cext
        self.Cext_polystirene = Cext_polystirene                                                        # Theoretical extinction cross section from Mie scattering theory
        self.selected_Cext = selected_Cext
        self.poly_fit = poly_fit                                                                        # Polynomial fit coefficients for extinction cross section <-> diameter inversion
        self.save_path = save_path                                                                      # Path where to save the results
        self.save_name = save_name                                                                      # Output file name
        self.correction_labels = labels_correct                                                         # Labels for size distribution correction,  referring both to extinction correction, 
        self.RI_correction = self.correction_labels[0]                                                 # refractive index correction and aspect-ratio correction
        self.ar_correction = self.correction_labels[1]
        self.linewidth = 4                                                                              # Plot linewidth
        self.linestyle = QtCore.Qt.SolidLine                                                            # Plot linestyle

        if os.path.isdir(self.save_path): print("")                                                     # If the path does not yet exists, it is created
        else: os.makedirs(self.save_path)

        self.output_file = open(self.save_path+self.save_name+'_summary.txt', 'w')                      # Output file initialization
        self.output_file.write('Raw d [$\mu$m]\t\tRI corrected d [$\mu$m]\t\tAR corrected d [$\mu$m]\t\t# counts\n')

        self.y_data = y_data                                                                            # Number of counts for each Abakus channel (= particle diameter)
        self.time_data = y_time_data
        self.time_xrange = np.linspace(0, len(self.time_data)-1, len(self.time_data))
        self.dataset = full_dataset

        self.raw_x_data = self.x_data[0]
        self.RI_x_data = self.x_data[1]
        self.ar_x_data = self.x_data[2]
        self.error = self.raw_x_data[1] - self.raw_x_data[0]

        if not all(self.diameters_Cext)==0:
            self.start = np.where(self.diameters_Cext==self.raw_x_data[0])[0][0]
            self.stop = np.where(self.diameters_Cext==self.raw_x_data[-1])[0][0]

            self.diameters_Cext = self.diameters_Cext[self.start:self.stop]
            self.Cext_polystirene = self.Cext_polystirene[self.start:self.stop]
            self.selected_Cext = self.selected_Cext[self.start:self.stop]

        self.RI_tab_win = QTabWidget(self.widget)                                                       # STEP 3: refractive index corrected size distribution visualizaton
        self.RI_tab_win.setGeometry(QtCore.QRect(960, 0, 960, 600))
        self.RI_tab_win.setTabPosition(QTabWidget.West)
        self.RI_win = pg.GraphicsLayoutWidget()
        self.cext_win = pg.GraphicsLayoutWidget()
        self.cext_inv_win = pg.GraphicsLayoutWidget()
        self.RI_tab_win.addTab(self.RI_win, 'Tab 1')
        self.RI_tab_win.addTab(self.cext_win, 'Tab 2')
        self.RI_tab_win.addTab(self.cext_inv_win, 'Tab 3')
        self.RI_tab_win.setTabText(0, 'Size distribution')
        self.RI_tab_win.setTabText(1, 'Ext. cross-section')
        self.RI_tab_win.setTabText(2, 'Cross-section inversion')
        pg.setConfigOptions(antialias=True)
        self.RI_plt = self.RI_win.addPlot()
        self.RI_plt.setTitle('<b>Refractive-index corrected size distribution')
        self.RI_plt.setLabel('bottom', 'd [\u03bc'+'m]')
        self.RI_plt.setLabel('left', '# counts')
        self.RI_lr = pg.LinearRegionItem([self.RI_x_data[-1], self.RI_x_data[-1]], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.RI_lr.setZValue(-10)
        self.RI_plt.addItem(self.RI_lr)
        self.RI_range = self.RI_lr.getRegion()
        self.RI_curve = self.RI_plt.plot(pen=pg.mkPen('r', width=self.linewidth, style=self.linestyle), fillLevel=0, brush=(255,100,0,100))
        self.RI_curve_upd = self.RI_plt.plot(pen=pg.mkPen('y', width=self.linewidth, style=self.linestyle), fillLevel=0, brush=(255,100,0,100))
        self.RI_legend = QtWidgets.QTextBrowser(self.widget)
        self.RI_legend.setGeometry(QtCore.QRect(1700, 70, 150, 120))
        self.RI_legend.setFontPointSize(8)
        if self.RI_correction==True:
            self.RI_curve.setData(self.RI_x_data[1:], self.y_data[1:], stepMode='right')
            self.RI_legend.append('<b>Particles:</b>  '+'{:.2e}'.format(sum(self.y_data))+' pts')  
            self.RI_legend.append('<b>Peak:</b>        '+'{:.02f}'.format(self.RI_x_data[np.where(self.y_data==np.amax(self.y_data))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.RI_legend.append('<b>M. avg:</b>     '+'{:.02f}'.format(np.mean(self.RI_x_data))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.RI_x_data)))+' µm')
            try: self.RI_legend.append('<b>W. avg.:</b>   '+'{:.02f}'.format(np.average(self.RI_x_data, weights=self.y_data))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.y_data**2))/sum(self.y_data))+' µm')
            except: self.RI_legend.append('<b>W. avg.:</b> nan')
            self.RI_legend.append('<b>S.D.:</b>         '+'{:.02f}'.format(np.sqrt(np.var((self.RI_x_data))))+' µm')
            self.RI_legend.append('<b>1st qnt.:</b>   '+'{:.02f}'.format(np.quantile(self.RI_x_data, 0.25))+' µm')
            self.RI_legend.append('<b>2nd qnt.:</b>  '+'{:.02f}'.format(np.median(self.RI_x_data))+' µm')
            self.RI_legend.append('<b>3rd qnt.:</b>   '+'{:.02f}'.format(np.quantile(self.RI_x_data, 0.75))+' µm')
        else:
            self.RI_legend.append('<b>Particles:</b>   '+'-.--e- pts')  
            self.RI_legend.append('<b>Peak:</b>        '+' -.-- ± -.-- µm')
            self.RI_legend.append('<b>M. avg:</b>     '+' -.-- ± -.-- µm')
            self.RI_legend.append('<b>W. avg.:</b>   '+' -.-- ± -.-- µm')
            self.RI_legend.append('<b>S.D.:</b>         '+' -.-- µm')
            self.RI_legend.append('<b>1st qnt.:</b>   '+' -.-- µm')
            self.RI_legend.append('<b>2nd qnt.:</b>  '+' -.-- µm')
            self.RI_legend.append('<b>3rd qnt.:</b>   '+' -.-- µm')
        
        self.cext_plt = self.cext_win.addPlot()
        self.cext_plt.setTitle('<b>Extinction cross-sections comparison')
        self.cext_plt.setLabel('bottom', 'd [\u03bc'+'m]')
        self.cext_plt.setLabel('left', 'σ ext [\u03bc'+'m^2]')
        self.cext_polystirene_curve = self.cext_plt.plot(pen=pg.mkPen('b', width=self.linewidth, style=self.linestyle))
        self.cext_curve = self.cext_plt.plot(pen=pg.mkPen('r', width=self.linewidth, style=self.linestyle))
        self.cext_polystirene_curve.setData(self.diameters_Cext, self.Cext_polystirene)
        self.cext_curve.setData(self.diameters_Cext, self.selected_Cext)
        legend_cext = pg.LegendItem((0,0), offset=(50,50))
        legend_cext.setParentItem(self.cext_plt.graphicsItem())
        legend_cext.addItem(self.cext_polystirene_curve, 'n = 1.5848')
        if self.ref_index_IM==0: legend_cext.addItem(self.cext_curve, 'n = '+'{:.04f}'.format(self.ref_index_RE))
        else: legend_cext.addItem(self.cext_curve, 'n = '+'{:.04f}'.format(self.ref_index_RE)+' + i'+'{:.04f}'.format(self.ref_index_IM))

        self.cext_inv_plt = self.cext_inv_win.addPlot()
        self.cext_inv_plt.setTitle('<b>Selected extinction cross-section inversion')
        self.cext_inv_plt.setLabel('left', 'd [\u03bc'+'m]')
        self.cext_inv_plt.setLabel('bottom', 'σ ext [\u03bc'+'m^2]')
        self.cext_inv_curve = self.cext_inv_plt.plot(pen=pg.mkPen('r', width=self.linewidth, style=self.linestyle))
        self.cext_inv_curve.setData(self.selected_Cext, self.diameters_Cext)
        self.fit_curve = self.cext_inv_plt.plot(pen=pg.mkPen('k', width=self.linewidth, style=self.linestyle))
        self.fit_curve.setData(self.selected_Cext, self.poly_fit(self.selected_Cext))
        legend_cext = pg.LegendItem((0,0), offset=(50,50))
        legend_cext.setParentItem(self.cext_inv_plt.graphicsItem())
        legend_cext.addItem(self.cext_inv_curve, 'd vs σ ext')
        legend_cext.addItem(self.fit_curve, 'fit')
        

        self.raw_win = pg.GraphicsLayoutWidget(self.widget, show=True)                                  # STEP 1: raw size distribution visualizaton
        self.raw_win.setGeometry(QtCore.QRect(0, 0, 960, 600))
        pg.setConfigOptions(antialias=True)
        self.raw_plt = self.raw_win.addPlot()
        self.raw_plt.setTitle('<b>Raw size distribution')
        self.raw_plt.setLabel('bottom', 'd [\u03bc'+'m]')
        self.raw_plt.setLabel('left', '# counts')
        self.raw_lr = pg.LinearRegionItem([self.raw_x_data[-1], self.raw_x_data[-1]], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.raw_lr.setZValue(-10)
        self.raw_plt.addItem(self.raw_lr)
        self.raw_range = self.raw_lr.getRegion()
        self.raw_curve = self.raw_plt.plot(pen=pg.mkPen('b', width=self.linewidth, style=self.linestyle), fillLevel=0, brush=(50,50,255,100))
        self.raw_curve_upd = self.raw_plt.plot(pen=pg.mkPen('#1f456e', width=self.linewidth, style=self.linestyle), fillLevel=0, brush='#82eefd')
        self.raw_curve.setData(self.raw_x_data[1:], self.y_data[1:], stepMode='right')
        self.raw_legend = QtWidgets.QTextBrowser(self.widget)
        self.raw_legend.setGeometry(QtCore.QRect(750, 70, 150, 120))
        self.raw_legend.setFontPointSize(8)
        self.raw_legend.append('<b>Particles:</b>  '+'{:.2e}'.format(sum(self.y_data))+' pts')  
        self.raw_legend.append('<b>Peak:</b>        '+'{:.02f}'.format(self.raw_x_data[np.where(self.y_data==np.amax(self.y_data))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
        self.raw_legend.append('<b>M. avg:</b>     '+'{:.02f}'.format(np.mean(self.raw_x_data))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.raw_x_data)))+' µm')
        try: self.raw_legend.append('<b>W. avg.:</b>   '+'{:.02f}'.format(np.average(self.raw_x_data, weights=self.y_data))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.y_data**2))/sum(self.y_data))+' µm')
        except: self.raw_legend.append('W. avg.:</b> nan')
        self.raw_legend.append('<b>S.D.:</b>         '+'{:.02f}'.format(np.sqrt(np.var((self.raw_x_data))))+' µm')
        self.raw_legend.append('<b>1st qnt.:</b>   '+'{:.02f}'.format(np.quantile(self.raw_x_data, 0.25))+' µm')
        self.raw_legend.append('<b>2nd qnt.:</b>  '+'{:.02f}'.format(np.median(self.raw_x_data))+' µm')
        self.raw_legend.append('<b>3rd qnt.:</b>   '+'{:.02f}'.format(np.quantile(self.raw_x_data, 0.75))+' µm')
        

        self.time_win = pg.GraphicsLayoutWidget(self.widget, show=True)
        self.time_win.setGeometry(QtCore.QRect(0, 605, 1920, 405))
        pg.setConfigOptions(antialias=True)
        self.time_plt = self.time_win.addPlot()
        self.time_plt.setTitle('<b>Time distribution')
        self.time_plt.setLabel('left', '# counts')
        self.time_plt.setLabel('bottom', 't [s]')
        self.time_lr = pg.LinearRegionItem([self.time_xrange[0], self.time_xrange[-1]], pen=pg.mkPen(width=2.5), brush=(255,255,255,100))
        self.time_lr.setZValue(-10)
        self.time_plt.addItem(self.time_lr)
        self.time_range = self.time_lr.getRegion()
        self.time_curve = self.time_plt.plot(pen=pg.mkPen('#ff7c2e', width=self.linewidth, style=self.linestyle), fillLevel=0, brush=(255,255,0,100))
        self.time_curve_upd = self.time_plt.plot(pen=pg.mkPen('r', width=self.linewidth, style=self.linestyle), fillLevel=0, brush=(255,255,0,100))
        self.time_curve.setData(self.time_xrange, self.time_data, stepMode='right')


        self.update_raw = QPushButton(self.widget)
        self.update_raw.setGeometry(QtCore.QRect(750, 13, 85, 25))
        self.update_raw.setText('Get stats')
        self.update_raw.clicked.connect(self.update_plot)
        self.update_raw.setStyleSheet("QPushButton { color: green; font: bold 11px; }")

        self.update_RI = QPushButton(self.widget)
        self.update_RI.setGeometry(QtCore.QRect(1700, 13, 85, 25))
        self.update_RI.setText('Get stats')
        self.update_RI.clicked.connect(self.update_plot)
        self.update_RI.setStyleSheet("QPushButton { color: green; font: bold 11px; }")

        self.update_time = QPushButton(self.widget)
        self.update_time.setGeometry(QtCore.QRect(1190, 812, 85, 25))
        self.update_time.setText('Update')
        self.update_time.clicked.connect(self.update_time_plot)
        self.update_time.setStyleSheet("QPushButton { color: green; font: bold 11px; }")

        for i in range(0, len(self.raw_x_data)): self.output_file.write('{:.02f}'.format(self.raw_x_data[i])+'\t\t\t'+'{:.02f}'.format(self.RI_x_data[i])+'\t\t\t\t'+'{:.02f}'.format(self.ar_x_data[i])+'\t\t\t\t'+str(self.y_data[i])+'\n')


    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def update_plot(self):

        self.indexes = len(self.raw_x_data)

        self.new_raw_range = self.raw_lr.getRegion()
        self.new_RI_range = self.RI_lr.getRegion()

        if self.new_raw_range!=self.raw_range: 
            self.indexes = np.where((self.raw_x_data >= self.new_raw_range[0]) & (self.raw_x_data <= self.new_raw_range[1]))[0]
            self.raw_lr.setRegion([self.new_raw_range[0], self.new_raw_range[1]])
            self.raw_curve_upd.setData(self.raw_x_data[self.indexes], self.y_data[self.indexes], stepMode='right')
            self.raw_legend.clear()
            self.raw_legend.append('Particles:  '+'{:.2e}'.format(sum(self.y_data[self.indexes]))+' pts')  
            self.raw_legend.append('Peak:        '+'{:.02f}'.format(self.raw_x_data[np.where(self.y_data==np.amax(self.y_data[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.raw_legend.append('M. avg:     '+'{:.02f}'.format(np.mean(self.raw_x_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.raw_x_data[self.indexes])))+' µm')
            try: self.raw_legend.append('W. avg.:   '+'{:.02f}'.format(np.average(self.raw_x_data[self.indexes], weights=self.y_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.y_data[self.indexes]**2))/sum(self.y_data[self.indexes]))+' µm')
            except: self.raw_legend.append('W. avg.: nan')
            self.raw_legend.append('S.D.:         '+'{:.02f}'.format(np.sqrt(np.var((self.raw_x_data[self.indexes]))))+' µm')
            self.raw_legend.append('1st qnt.:   '+'{:.02f}'.format(np.quantile(self.raw_x_data[self.indexes], 0.25))+' µm')
            self.raw_legend.append('2nd qnt.:  '+'{:.02f}'.format(np.median(self.raw_x_data[self.indexes]))+' µm')
            self.raw_legend.append('3rd qnt.:   '+'{:.02f}'.format(np.quantile(self.raw_x_data[self.indexes], 0.75))+' µm')

        if self.new_RI_range!=self.RI_range: 
            self.indexes = np.where((self.RI_x_data >= self.new_RI_range[0]) & (self.RI_x_data <= self.new_RI_range[1]))[0]
            self.RI_lr.setRegion([self.new_RI_range[0], self.new_RI_range[1]])
            self.RI_curve_upd.setData(self.RI_x_data[self.indexes], self.y_data[self.indexes], stepMode='right')
            self.RI_legend.clear()
            self.RI_legend.append('Particles:  '+'{:.2e}'.format(sum(self.y_data[self.indexes]))+' pts')  
            self.RI_legend.append('Peak:        '+'{:.02f}'.format(self.RI_x_data[np.where(self.y_data==np.amax(self.y_data[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.RI_legend.append('M. avg:     '+'{:.02f}'.format(np.mean(self.RI_x_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.RI_x_data[self.indexes])))+' µm')
            try: self.RI_legend.append('W. avg.:   '+'{:.02f}'.format(np.average(self.RI_x_data[self.indexes], weights=self.y_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.y_data[self.indexes]**2))/sum(self.y_data[self.indexes]))+' µm')
            except: self.RI_legend.append('W. avg.: nan')
            self.RI_legend.append('S.D.:         '+'{:.02f}'.format(np.sqrt(np.var((self.RI_x_data[self.indexes]))))+' µm')
            self.RI_legend.append('1st qnt.:   '+'{:.02f}'.format(np.quantile(self.RI_x_data[self.indexes], 0.25))+' µm')
            self.RI_legend.append('2nd qnt.:  '+'{:.02f}'.format(np.median(self.RI_x_data[self.indexes]))+' µm')
            self.RI_legend.append('3rd qnt.:   '+'{:.02f}'.format(np.quantile(self.RI_x_data[self.indexes], 0.75))+' µm')

        self.raw_range = self.raw_lr.getRegion()
        self.RI_range = self.RI_lr.getRegion()

    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------#
    #

    def update_time_plot(self):

        self.indexes = len(self.raw_x_data)-1

        self.new_time_range = self.time_lr.getRegion()

        if self.new_time_range!=self.time_range:
            self.time_indexes = np.where((np.linspace(0, self.dataset.shape[0]-1, self.dataset.shape[0]) >= self.new_time_range[0]) & (np.linspace(0, self.dataset.shape[0]-1, self.dataset.shape[0]) <= self.new_time_range[1]))[0]
            self.time_lr.setRegion([self.new_time_range[0], self.new_time_range[1]])
            if len(self.time_indexes)==1: self.single_histogram = np.array(self.dataset.loc[self.time_indexes[0]]) - np.array(self.dataset.loc[self.time_indexes[0]-1])
            elif len(self.time_indexes)>1: 
                self.single_histogram = np.array(self.dataset.loc[self.time_indexes[0]]) - np.array(self.dataset.loc[self.time_indexes[0]-1])
                for kk in range(1, len(self.time_indexes)):
                    self.single_histogram += np.array(self.dataset.loc[self.time_indexes[kk]]) - np.array(self.dataset.loc[self.time_indexes[kk]-1])
        
            self.raw_curve_upd.setData(self.raw_x_data[self.indexes], self.single_histogram[self.indexes], stepMode='right')
            self.raw_legend.clear()
            self.raw_legend.append('Particles:  '+'{:.2e}'.format(sum(self.single_histogram[self.indexes]))+' pts')  
            self.raw_legend.append('Peak:        '+'{:.02f}'.format(self.raw_x_data[np.where(self.single_histogram==np.amax(self.single_histogram[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.raw_legend.append('M. avg:     '+'{:.02f}'.format(np.mean(self.raw_x_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.raw_x_data[self.indexes])))+' µm')
            try: self.raw_legend.append('W. avg.:   '+'{:.02f}'.format(np.average(self.raw_x_data[self.indexes], weights=self.single_histogram[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.single_histogram[self.indexes]**2))/sum(self.single_histogram[self.indexes]))+' µm')
            except: self.raw_legend.append('W. avg.: nan')
            self.raw_legend.append('S.D.:         '+'{:.02f}'.format(np.sqrt(np.var((self.raw_x_data[self.indexes]))))+' µm')
            self.raw_legend.append('1st qnt.:   '+'{:.02f}'.format(np.quantile(self.raw_x_data[self.indexes], 0.25))+' µm')
            self.raw_legend.append('2nd qnt.:  '+'{:.02f}'.format(np.median(self.raw_x_data[self.indexes]))+' µm')
            self.raw_legend.append('3rd qnt.:   '+'{:.02f}'.format(np.quantile(self.raw_x_data[self.indexes], 0.75))+' µm')

        
            self.RI_curve_upd.setData(self.RI_x_data[self.indexes], self.single_histogram[self.indexes], stepMode='right')
            self.RI_legend.clear()
            self.RI_legend.append('Particles:  '+'{:.2e}'.format(sum(self.single_histogram[self.indexes]))+' pts')  
            self.RI_legend.append('Peak:        '+'{:.02f}'.format(self.RI_x_data[np.where(self.single_histogram==np.amax(self.single_histogram[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.RI_legend.append('M. avg:     '+'{:.02f}'.format(np.mean(self.RI_x_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.RI_x_data[self.indexes])))+' µm')
            try: self.RI_legend.append('W. avg.:   '+'{:.02f}'.format(np.average(self.RI_x_data[self.indexes], weights=self.single_histogram[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.single_histogram[self.indexes]**2))/sum(self.single_histogram[self.indexes]))+' µm')
            except: self.RI_legend.append('W. avg.: nan')
            self.RI_legend.append('S.D.:         '+'{:.02f}'.format(np.sqrt(np.var((self.RI_x_data[self.indexes]))))+' µm')
            self.RI_legend.append('1st qnt.:   '+'{:.02f}'.format(np.quantile(self.RI_x_data[self.indexes], 0.25))+' µm')
            self.RI_legend.append('2nd qnt.:  '+'{:.02f}'.format(np.median(self.RI_x_data[self.indexes]))+' µm')
            self.RI_legend.append('3rd qnt.:   '+'{:.02f}'.format(np.quantile(self.RI_x_data[self.indexes], 0.75))+' µm')

        
            self.ar_curve_upd.setData(self.ar_x_data[self.indexes], self.single_histogram[self.indexes], stepMode='right')
            self.ar_legend.clear()
            self.ar_legend.append('Particles:  '+'{:.2e}'.format(sum(self.single_histogram[self.indexes]))+' pts')  
            self.ar_legend.append('Peak:        '+'{:.02f}'.format(self.ar_x_data[np.where(self.single_histogram==np.amax(self.single_histogram[self.indexes]))[0]][0])+' ± '+'{:.02f}'.format(self.error)+' µm')
            self.ar_legend.append('M. avg:     '+'{:.02f}'.format(np.mean(self.ar_x_data[self.indexes]))+' ± '+'{:.02f}'.format(self.error/np.sqrt(len(self.ar_x_data[self.indexes])))+' µm')
            try: self.ar_legend.append('W. avg.:   '+'{:.02f}'.format(np.average(self.ar_x_data[self.indexes], weights=self.single_histogram[self.indexes]))+' ± '+'{:.02f}'.format(self.error*np.sqrt(sum(self.single_histogram[self.indexes]**2))/sum(self.single_histogram[self.indexes]))+' µm')
            except: self.ar_legend.append('W. avg.: nan')
            self.ar_legend.append('S.D.:         '+'{:.02f}'.format(np.sqrt(np.var((self.ar_x_data[self.indexes]))))+' µm')
            self.ar_legend.append('1st qnt.:   '+'{:.02f}'.format(np.quantile(self.ar_x_data[self.indexes], 0.25))+' µm')
            self.ar_legend.append('2nd qnt.:  '+'{:.02f}'.format(np.median(self.ar_x_data[self.indexes]))+' µm')
            self.ar_legend.append('3rd qnt.:   '+'{:.02f}'.format(np.quantile(self.ar_x_data[self.indexes], 0.75))+' µm')

        self.time_range = self.time_lr.getRegion()


############################################################################################################################################################
############################################################################################################################################################
