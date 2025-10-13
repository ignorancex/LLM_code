#! /usr/bin/env python3
"""

This file contains a GUI application for interactive calculating and plotting the acceptance probability of a quantum token,
the angle dependence of the fraction of qubits measured in the |0> state and the total noise for a given theta.
The desidered parameters can be input and the plots will be updated in real time.
This is part of 'Ensemble-Based Quantum-Token Protocol Benchmarked on IBM Quantum Processors' arXiv:2412.08530 (2024).

Version: 1.0
Date: 12.12.2024
License: GPL-3.0
"""

import sys

from scipy.stats import norm, skewnorm
from scipy.integrate import quad

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Tries to import either PySide6 or PyQt6 depending on the user's environment
try:
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QLabel,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTabWidget,
        QSlider,
        QLineEdit
    )
    from PySide6.QtCore import Qt
except ImportError:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QLabel,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QSlider,
        QLineEdit
    )
    from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

matplotlib.use("QtAgg")

plt.rcParams.update({"font.size": 3})

def n_analytical(theta_b, theta_a, phi_b, phi_a, c):
    """
    Analytical expression of the probability of the faker to measure the token in the |0> state, as defined in the paper.

    Parameters:
    theta_b (float): banl polar angle in degrees
    theta_a (float): attacker polar angle in degrees
    phi_b (float): bank azimuthal angle in degrees
    phi_a (float): attacker azimuthal angle in degrees
    c (float): normalized contrast between the two states

    Returns:
    float: The fraction of qubits measured in |0> state
    """
    theta_a *= np.pi/180
    theta_b *= np.pi/180
    phi_a *= np.pi/180
    phi_b *= np.pi/180        
    return 1/2 + c/2*(np.cos(theta_a)*np.cos(theta_b) + np.sin(theta_a)*np.sin(theta_b)*np.cos(phi_b - phi_a))

def sigman(theta, N0, N1, sigma_exp):
    """ 
    Calculate the total noise for a given theta, N0, N1 and sigma_exp.
    
    Parameters:
    theta (float): The angle in degrees
    N0 (float): The number of counts for the |0> state
    N1 (loat): The number of counts for the |1> state
    sigma_exp (float): The experimental uncertainty component of the total noise

    Returns:
    float: The total noise
    """
    theta *= np.pi/180
    sigma_s2 = np.cos(theta/2)**2*N0 + np.sin(theta/2)**2*N1
    sigma_q2 = N0**2*np.cos(theta/2)**2 + N1**2*np.sin(theta/2)**2 - sigma_s2**2

    return (sigma_s2 + sigma_q2 + sigma_exp**2)**.5

class InteractivePlotApp(QMainWindow):
    """
    Class for the application window.

    Class Methods
    -------------
    __init__():
        Initializes the main window and the layout.
    set_p_layout():
        Initializes the layout for the p tab with the sliders and input fields.
    set_n_layout():
        Initializes the layout for the n tab with the sliders and input fields.
    set_sigman_layout():
        Initializes the layout for the sigman tab with the input fields.
    update_nf_value(value):
        Updates the value of the nf slider label.
    update_nb_value(value):
        Updates the value of the nb slider label.
    update_nT_value(value):
        Updates the value of the nT slider label.
    update_c_value(value):
        Updates the value of the c slider label.
    update_thetaa_value(value):
        Updates the value of the thetaa slider label.
    update_phia_value(value):
        Updates the value of the phia slider label.
    set_right_layout():
        Intializes the plot on the right side of the grid with an empty plot.
    calculate_p(nb_mean, sigmab, nf_shape, nf_loc, nf_scale, nT):
        Calculates the acceptance probability of the bank and the forger by integrating the distributions.
    update_n_plot():
        Updates the plot for the n tab with the new values of the sliders.
    update_p_plot():
        Updates the plot for the p tab with the new values of the sliders.
    update_sigman_plot():
        Updates the plot for the sigman tab with the new values of the input fields.
    """
    def __init__(self):
        """
        Initializes the main window and the layout.
        There are 3 tabs, one for each plot and a right side for the plot.
        """
        super().__init__()

        self.main_layout = QHBoxLayout()
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("Ensemble-Based Quantum-Token Protocol Benchmarked on IBMQ")
        self.showMaximized()

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        self.tab_p = QWidget()
        self.tab_p_layout = QVBoxLayout()
        self.tab_p.setLayout(self.tab_p_layout)

        self.tab_n = QWidget()
        self.tab_n_layout = QVBoxLayout(self.tab_n)

        self.tab_sigman = QWidget()
        self.tab_sigman_layout = QVBoxLayout(self.tab_sigman)

        self.set_p_layout()
        self.set_n_layout()
        self.set_sigman_layout()

        self.tab_widget.addTab(self.tab_p, "p")
        self.tab_widget.addTab(self.tab_n, "n")
        self.tab_widget.addTab(self.tab_sigman, "σₙ")
        self.left_layout.addWidget(self.tab_widget)
        self.left_layout.addStretch()

        self.set_right_layout()

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

    def set_p_layout(self):
        """
        Initializes the layout for the p tab with the sliders and input fields.        
        """
        self.tab_p_layout.addWidget(QLabel("<b>Forger Distribution Parameters</b>"))

        self.tab_p_layout.addWidget(QLabel("Location:"))
        self.nf_slider = QSlider(Qt.Horizontal)
        self.nf_slider.setMinimum(0)
        self.nf_slider.setMaximum(100)
        self.nf_slider.setValue(60) 
        self.nf_slider.valueChanged.connect(self.update_nf_value)
        self.tab_p_layout.addWidget(self.nf_slider)
        self.nf_value_label = QLabel("0.6")
        self.tab_p_layout.addWidget(self.nf_value_label)

        self.tab_p_layout.addWidget(QLabel("Scale:"))
        self.nf_scale_input = QLineEdit()
        self.nf_scale_input.setText('0.4')
        self.tab_p_layout.addWidget(self.nf_scale_input)

        self.tab_p_layout.addWidget(QLabel("Shape:"))
        self.nf_shape_input = QLineEdit()
        self.nf_shape_input.setText('-30')
        self.tab_p_layout.addWidget(self.nf_shape_input)

        self.tab_p_layout.addWidget(QLabel("<b>Bank Distribution Parameters</b>"))

        self.tab_p_layout.addWidget(QLabel("Mean value ⟨n<sub>b</sub>⟩:"))
        self.nb_slider = QSlider(Qt.Horizontal)
        self.nb_slider.setMinimum(0)
        self.nb_slider.setMaximum(100)
        self.nb_slider.setValue(80)  # initial value
        self.nb_slider.valueChanged.connect(self.update_nb_value)
        self.tab_p_layout.addWidget(self.nb_slider)
        self.nb_value_label = QLabel("0.8")
        self.tab_p_layout.addWidget(self.nb_value_label)

        self.tab_p_layout.addWidget(QLabel("Standard deviation σ<sub>b</sub>:"))
        self.sigmab_input = QLineEdit()
        self.sigmab_input.setText('0.05')
        self.tab_p_layout.addWidget(self.sigmab_input)

        self.tab_p_layout.addWidget(QLabel("Acceptance Threshold n<sub>T</sub>:"))
        self.nT_slider = QSlider(Qt.Horizontal)
        self.nT_slider.setMinimum(0)
        self.nT_slider.setMaximum(100)
        self.nT_slider.setValue(70)  # initial value
        self.nT_slider.valueChanged.connect(self.update_nT_value)
        self.tab_p_layout.addWidget(self.nT_slider)
        self.nT_value_label = QLabel("0.7")
        self.tab_p_layout.addWidget(self.nT_value_label)

        self.nf_slider.valueChanged.connect(self.update_p_plot)
        self.nb_slider.valueChanged.connect(self.update_p_plot)
        self.nT_slider.valueChanged.connect(self.update_p_plot)
        self.nf_scale_input.textChanged.connect(self.update_p_plot)
        self.nf_shape_input.textChanged.connect(self.update_p_plot)
        self.sigmab_input.textChanged.connect(self.update_p_plot)

    def set_n_layout(self):
        """
        Initializes the layout for the n tab with the sliders and input fields.        
        """
        self.tab_n_layout.addWidget(QLabel("Contrast c"))
        self.c_slider = QSlider(Qt.Horizontal)
        self.c_slider.setMinimum(0)
        self.c_slider.setMaximum(100)
        self.c_slider.setValue(80) 
        self.c_slider.valueChanged.connect(self.update_c_value)
        self.tab_n_layout.addWidget(self.c_slider)
        self.c_value_label = QLabel("0.8")
        self.tab_n_layout.addWidget(self.c_value_label)

        self.tab_n_layout.addWidget(QLabel("Attacker Angles"))

        self.tab_n_layout.addWidget(QLabel("θ<sub>a</sub>"))
        self.thetaa_slider = QSlider(Qt.Horizontal)
        self.thetaa_slider.setMinimum(0)
        self.thetaa_slider.setMaximum(180)
        self.thetaa_slider.setValue(90) 
        self.thetaa_slider.valueChanged.connect(self.update_thetaa_value)
        self.tab_n_layout.addWidget(self.thetaa_slider)
        self.thetaa_value_label = QLabel("90°")
        self.tab_n_layout.addWidget(self.thetaa_value_label)

        self.tab_n_layout.addWidget(QLabel("φ<sub>a</sub>"))
        self.phia_slider = QSlider(Qt.Horizontal)
        self.phia_slider.setMinimum(0)
        self.phia_slider.setMaximum(360)
        self.phia_slider.setValue(180) 
        self.phia_slider.valueChanged.connect(self.update_phia_value)
        self.tab_n_layout.addWidget(self.phia_slider)
        self.phia_value_label = QLabel("180°")
        self.tab_n_layout.addWidget(self.phia_value_label)

        self.c_slider.valueChanged.connect(self.update_n_plot)
        self.thetaa_slider.valueChanged.connect(self.update_n_plot)
        self.phia_slider.valueChanged.connect(self.update_n_plot)

    def set_sigman_layout(self):
        """
        Initializes the layout for the sigman tab with the input fields.
        """
        self.tab_sigman_layout.addWidget(QLabel("|0⟩ Eigenvalue N<sub>0</sub>"))
        self.N0_input = QLineEdit()
        self.N0_input.setText('1')
        self.tab_sigman_layout.addWidget(self.N0_input)

        self.tab_sigman_layout.addWidget(QLabel("|1⟩ Eigenvalue N<sub>1</sub>"))
        self.N1_input = QLineEdit()
        self.N1_input.setText('10')
        self.tab_sigman_layout.addWidget(self.N1_input)

        self.tab_sigman_layout.addWidget(QLabel("Experimental Error σ<sub>exp</sub>"))
        self.sigmaexp_input = QLineEdit()
        self.sigmaexp_input.setText('1')
        self.tab_sigman_layout.addWidget(self.sigmaexp_input)

        self.N0_input.textChanged.connect(self.update_sigman_plot)
        self.N1_input.textChanged.connect(self.update_sigman_plot)
        self.sigmaexp_input.textChanged.connect(self.update_sigman_plot)

    def update_nf_value(self, value):
        """
        Updates the value of the nf slider label.
        """
        self.nf_value_label.setText(str(value/100))

    def update_nb_value(self, value):
        """
        Updates the value of the nb slider label.
        """
        self.nb_value_label.setText(str(value/100))

    def update_nT_value(self, value):
        """
        Updates the value of the nT slider label.
        """
        self.nT_value_label.setText(str(value/100))

    def update_c_value(self, value):
        """
        Updates the value of the c slider label.
        """
        self.c_value_label.setText(str(value/100))

    def update_thetaa_value(self, value):
        """
        Updates the value of the thetaa slider label.
        """
        self.thetaa_value_label.setText(str(value)+"°")

    def update_phia_value(self, value):
        """
        Updates the value of the phia slider label.
        """
        self.phia_value_label.setText(str(value)+"°")

    def set_right_layout(self):
        """
        Intializes the plot on the right side of the grid with an empty plot.
        """
        self.fig = Figure(dpi=400)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.fig)

        # Add a navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addWidget(self.canvas)
        self.update_p_plot()

    def calculate_p(self, nb_mean, sigmab, nf_shape, nf_loc, nf_scale, nT):
        """
        Calculates the acceptance probability of the bank and the forger by integrating the gaussian distributions.

        Parameters:
        nb_mean (float): The mean value of the bank distribution
        sigmab (float): The standard deviation of the bank distribution
        nf_shape (float): The shape of the forger distribution
        nf_loc (float): The location of the forger distribution
        nf_scale (float): The scale of the forger distribution
        """
        self.pb = quad(norm.pdf, nT, 1, args=(nb_mean, sigmab))[0] / quad(norm.pdf, 0, 1, args=(nb_mean, sigmab))[0]
        self.pf = quad(skewnorm.pdf, nT, 1, args=(nf_shape, nf_loc, nf_scale))[0] / quad(skewnorm.pdf, 0, 1, args=(nf_shape, nf_loc, nf_scale))[0]

    def update_n_plot(self):
        """
        Updates the plot for the n tab with the new values of the sliders.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        c = float(self.c_slider.value()/100)
        thetaa = float(self.thetaa_slider.value())
        phia = float(self.phia_slider.value())

        thetab = np.linspace(0, 180, 20)
        phib = np.linspace(0, 360, 20)

        cm = self.ax.contourf(thetab,phib, np.array(
            [[n_analytical(thetaa, tb, phia, pb, c) for tb in thetab] for pb in phib]
            ).reshape(20, 20), cmap='viridis')
        
        self.ax.set_xlabel(r"$\theta_b$")
        self.ax.set_ylabel(r"$\phi_b$")
        
        self.cbar = self.fig.colorbar(cm, ax=self.ax)
        self.cbar.set_label(r"n")

        self.fig.text(0.9, 0.9, 'arXiv:2412.08530[quant-ph]', ha='right', va='top', color='gray', alpha=0.5, fontsize=3, transform=self.fig.transFigure)

        self.canvas.draw()

    def update_p_plot(self):
        """
        Updates the plot for the p tab with the new values of the sliders.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        nf_loc = float(self.nf_slider.value() / 100.0)
        nb_mean = float(self.nb_slider.value() / 100.0)
        nf_shape = float(self.nf_shape_input.text())
        nf_scale = float(self.nf_scale_input.text())
        sigmab = float(self.sigmab_input.text())
        nT = float(self.nT_slider.value() / 100.0)

        x = np.linspace(0, 1, 1000)

        self.ax.plot(x, norm.pdf(x, nb_mean, sigmab), label="$n_b$")
        self.ax.plot(x, skewnorm.pdf(x, nf_shape, nf_loc, nf_scale), label="$n_f$")        
        self.ax.axvline(nT, color="black", linestyle="--", label="$n_T$", alpha=0.8)
        self.ax.fill_between(np.arange(nT, 1, 0.001), 0, norm.pdf(np.arange(nT, 1, 0.001), nb_mean, sigmab), alpha=0.5)
        self.ax.fill_between(np.arange(nT, 1, 0.001), 0, skewnorm.pdf(np.arange(nT, 1, 0.001), nf_shape, nf_loc, nf_scale), alpha=0.5)

        self.ax.legend()

        self.ax.set_xlabel(r"n")
        self.ax.set_ylabel(r"Distribution")
        self.ax.set_xlim(0, 1)

        self.calculate_p(nb_mean, sigmab, nf_shape, nf_loc, nf_scale, nT)
        self.ax.set_title(f"Bank Acceptance Probability: {self.pb:.5f} \n Forger Acceptance Probability: {self.pf:.5f}")

        self.fig.text(0.9, 0.9, 'arXiv:2412.08530[quant-ph]', ha='right', va='top', color='gray', alpha=0.5, fontsize=3, transform=self.fig.transFigure)

        self.canvas.draw()

    def update_sigman_plot(self):
        """
        Updates the plot for the sigman tab with the new values of the input fields.
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        N0 = float(self.N0_input.text())
        N1 = float(self.N1_input.text())
        sigma_exp = float(self.sigmaexp_input.text())

        theta = np.linspace(0, 180, 100)

        self.ax.plot(theta, [sigman(t, N0, N1, sigma_exp) for t in theta], lw=2)
        self.ax.set_xlabel(r"$\theta$")
        self.ax.set_ylabel(r"$\sigma_n$")
        self.ax.set_title(fr"Total Noise $\sigma_n(\theta)$, $c=${np.abs(N0-N1)/(N0+N1):.5f}")
        self.ax.set_xlim(0, 180)

        self.fig.text(0.9, 0.9, 'arXiv:2412.08530[quant-ph]', ha='right', va='top', color='gray', alpha=0.5, fontsize=3, transform=self.fig.transFigure)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication()
    window = InteractivePlotApp()
    window.show()
    sys.exit(app.exec())