import numpy as np
from psprop import run_psprop, wf_extract
import matplotlib.pyplot as plt

# A simple function for producing a single slit aperture
def rect(x, half_wid):
    return np.where(abs(x) <= half_wid, 1, 0)

# Simulation parameters
pixel_x = 1E-6 # meters 
# Half the width of the slit 
half_wid  = 100 * pixel_x # meters
wavelength = 500E-6 # meters
prop_dist  = 1E-6   # meters
points = 1000 

x1 = pixel_x * np.arange(-points/2, points/2, 1)
x2 = x1.copy()

X1, X2 = np.meshgrid(x1, x2)

moi = rect(X1, half_wid) * rect(X2, half_wid)

moi_wig_z = run_psprop(moi, x1, X1, X2, pixel_x, wavelength, prop_dist)

wf_z = wf_extract(moi_wig_z, x1)

inten = abs(wf_z)**2
phase = np.unwrap(np.angle(wf_z))

# Plot the intensity of the wave field
plt.subplots(1,2)
plt.subplot(1,2,1)
plt.plot(x1, inten)

# Plot the phase of the wave field
plt.subplot(1,2,2)
plt.plot(phase)

plt.show()

