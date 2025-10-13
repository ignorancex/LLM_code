# Copyright (c) 2024 Sebastian Sassi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
#
# This example showcases the performance of the Zernike-based Radon transform
# in an interactive way. For every change of the parameters, it calls the C++
# program `lab_radon_example` (found in `lab_radon_example.cpp`, which needs to
# be compiled first for this program to work), which computes a simplified toy
# version of the dark matter direct detection event rate.
import argparse
import subprocess
import io
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import scipy.integrate as integrate

amu_to_gev = 0.9315

class RadonPlotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_axes([0.05, 0.55, 0.9, 0.3])
        self.ax2 = self.fig.add_axes([0.05, 0.2, 0.9, 0.3])

        self.e_count = 50
        self.t_count = 24

        title_text = f"Mock event rates for {self.t_count} times and {self.e_count} energies; not meant to be realistic"
        self.ax1.set_title(title_text, pad=16, wrap=True, fontsize="xx-large", c="tab:red")

        self.ax1.callbacks.connect("xlim_changed", self.update_ax)
        self.ax2.callbacks.connect("xlim_changed", self.update_ax)

        self.ax_vmax = self.fig.add_axes([0.20, 0.12, 0.60, 0.03])
        self.ax_vdisp = self.fig.add_axes([0.20, 0.08, 0.60, 0.03])
        self.ax_dist_order = self.fig.add_axes([0.20, 0.04, 0.60, 0.03])
        self.ax_dm_mass = self.fig.add_axes([0.20, 0.0, 0.60, 0.03])

        self.ax_xe = self.fig.add_axes([0.95, 0.0, 0.05, 0.03])
        self.ax_ar = self.fig.add_axes([0.95, 0.05, 0.05, 0.03])
        self.ax_ge = self.fig.add_axes([0.95, 0.1, 0.05, 0.03])
        self.ax_si = self.fig.add_axes([0.95, 0.15, 0.05, 0.03])
        self.ax_c = self.fig.add_axes([0.95, 0.2, 0.05, 0.03])
        self.ax_f = self.fig.add_axes([0.95, 0.25, 0.05, 0.03])
        self.ax_na = self.fig.add_axes([0.95, 0.3, 0.05, 0.03])
        self.ax_i = self.fig.add_axes([0.95, 0.35, 0.05, 0.03])

        #self.fig.add_axes()

        self.nucleus_mass = {
            "Xe": amu_to_gev*131.293,
            "Ar": amu_to_gev*39.963,
            "Ge": amu_to_gev*73.630,
            "Si": amu_to_gev*28.085,
            "C": amu_to_gev*12.011,
            "F": amu_to_gev*18.998,
            "Na": amu_to_gev*22.990,
            "I": amu_to_gev*126.90
        }

        self.vmax = 537
        self.vdisp = 233
        self.dist_order = 30
        self.dm_mass = 1.0
        self.element = "Xe"
        self.tmin = 8200
        self.tmax = 8565
        self.emin = 0.0
        self.emax = 2000
        self.energies = np.linspace(0.0, self.emax, self.e_count)
        self.times = np.linspace(self.tmin, self.tmax, self.t_count)
        
        self.slider_vmax = widgets.Slider(
            ax=self.ax_vmax, label=r"$v_{esc}$", valmin=500, valmax=600, valinit=self.vmax)
        self.slider_vdisp = widgets.Slider(
            ax=self.ax_vdisp, label=r"$v_0$", valmin=200, valmax=300, valinit=self.vdisp)
        self.slider_dist_order = widgets.Slider(
            ax=self.ax_dist_order, label=r"$L$", valmin=2, valmax=50, valinit=self.vdisp, valstep=1)
        self.slider_dm_mass = widgets.Slider(
            ax=self.ax_dm_mass, label=r"$m_{DM}$", valmin=0.4, valmax=2.0, valinit=self.dm_mass)
        
        self.slider_vmax.on_changed(self.update_vmax)
        self.slider_vdisp.on_changed(self.update_vdisp)
        self.slider_dist_order.on_changed(self.update_dist_order)
        self.slider_dm_mass.on_changed(self.update_dm_mass)
        
        self.button_xe = widgets.Button(self.ax_xe, "Xe")
        self.button_ar = widgets.Button(self.ax_ar, "Ar")
        self.button_ge = widgets.Button(self.ax_ge, "Ge")
        self.button_si = widgets.Button(self.ax_si, "Si")
        self.button_c = widgets.Button(self.ax_c, "C")
        self.button_f = widgets.Button(self.ax_f, "F")
        self.button_na = widgets.Button(self.ax_na, "Na")
        self.button_i = widgets.Button(self.ax_i, "I")

        self.button_xe.on_clicked(functools.partial(self.update_element, "Xe"))
        self.button_ar.on_clicked(functools.partial(self.update_element, "Ar"))
        self.button_ge.on_clicked(functools.partial(self.update_element, "Ge"))
        self.button_si.on_clicked(functools.partial(self.update_element, "Si"))
        self.button_c.on_clicked(functools.partial(self.update_element, "C"))
        self.button_f.on_clicked(functools.partial(self.update_element, "F"))
        self.button_na.on_clicked(functools.partial(self.update_element, "Na"))
        self.button_i.on_clicked(functools.partial(self.update_element, "I"))

        self.rate_energy = self.compute_radon()
        self.rate = integrate.simpson(self.rate_energy, self.energies)

        self.pcmesh = self.ax1.pcolormesh(self.times, self.energies, self.rate_energy.T, vmin=0.0)
        self.line, = self.ax2.plot(self.times, self.rate)

        self.ax1.set_xlim(self.times[0], self.times[-1])
        self.ax1.set_ylim(0.0, self.emax)
        
        self.ax2.set_xlim(self.times[0], self.times[-1])
    
    def update_vmax(self, val):
        self.vmax = val
        self.update()

    def update_vdisp(self, val):
        self.vdisp = val
        self.update()

    def update_dist_order(self, val):
        self.dist_order = val
        self.update()

    def update_dm_mass(self, val):
        self.dm_mass = val
        self.update()

    def update_element(self, element, event):
        self.element = element
        self.update()
    
    def update_tmin(self, val):
        self.tmin = val
        self.times = np.arange(self.tmin, self.tmax, 15)
        self.update()
    
    def update_tmax(self, val):
        self.tmax = val
        self.times = np.arange(self.tmin, self.tmax, 15)
        self.update()

    def update_ax(self, ax):
        self.tmin, self.tmax = ax.get_xlim()
        if (ax == self.ax1):
            self.emin, self.emax = ax.get_ylim()
        self.emin = 0.0
        self.emax = max(0.0, self.emax)
        self.update()
    
    def update(self):
        if (self.emin < self.emax):
            self.rate_energy = self.compute_radon()
            self.rate = integrate.simpson(self.rate_energy, self.energies)

            self.pcmesh.set_array(self.rate_energy.T)
            self.line.set_data(self.times, self.rate)
            self.fig.canvas.draw()
    
    def compute_radon(self):
        print(f"vmax: {self.vmax}, vdisp: {self.vdisp}, order: {self.dist_order}, DM mass: {self.dm_mass}, Nucleus mass: {self.nucleus_mass[self.element]}, tmin: {self.tmin}, tmax: {self.tmax}, emax: {self.emax}")
        p = subprocess.run(f"./lab_radon_example {self.vmax} {self.vdisp} {int(self.dist_order)} {self.dm_mass} {self.nucleus_mass[self.element]} {self.tmin} {self.tmax} {self.emax}", capture_output=True, text=True, shell=True)
        data = np.loadtxt(io.StringIO(p.stdout))
        return data


if __name__ == "__main__":
    plotter = RadonPlotter()
    plt.show()
