import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import genlaguerre

# Function to calculate the radial part of the Laguerre-Gaussian mode
def laguerre_gauss_radial(r, p, l, w0):
    normalization = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + abs(l))))
    radial_term = (r * np.sqrt(2) / w0) ** abs(l)
    gaussian_envelope = np.exp(-r**2 / w0**2)
    laguerre_poly = genlaguerre(p, abs(l))(2 * r**2 / w0**2)
    return normalization * radial_term * gaussian_envelope * laguerre_poly / w0

def plot_axial():
  print(f"Plotting from results data[1, 2 ,3].csv")
  plt.figure(figsize=(3.8, 3.2), dpi=300)
  colors = ["r", "b", "g"]
  styles = ["-", "--", ":"]
  widths = [1.1, 1.1, 1.5]
  labels = [r'$p=0, m=1$', r'$p=1, m=1$']
  config_number = 2
  for i in range(config_number):
    data = pd.read_csv("results/axial"+str(i+1)+".csv")
    x = data["x"]
    y = data["y"]
    plt.plot(x, y, 
             linestyle=styles[i], 
             color=colors[i], 
             linewidth=widths[i], 
             label=labels[i])
  # 
  plt.xlabel(r'$t$')
  plt.ylabel(r'$|A|^2$')
  plt.grid(False)
  plt.tight_layout()
  plt.legend(loc="upper left", borderaxespad=0.5, handletextpad=0.3, labelspacing=0.3, borderpad=0.2)
  plt.savefig("media/axial.pdf", format="pdf", dpi=300)
  #
  plt.figure(figsize=(2, 1.6), dpi=300)
  for i in range(config_number):
    data = pd.read_csv("results/sigma"+str(i+1)+".csv")
    x = data["x"]
    y = data["y"]
    plt.plot(x, y, 
             linestyle=styles[i], 
             color=colors[i], 
             linewidth=widths[i])
    
  plt.xlabel(r'$t$')
  plt.ylabel(r'$\sigma$')
  plt.grid(False)
  plt.tight_layout()
  plt.savefig("media/sigma.pdf", format="pdf", dpi=300)
  
def plot_radial():
    l = 1        # Azimuthal index
    r = np.linspace(0, 3, 500)
    colors = ["r", "b", "g"]
    styles = ["-", "--", ":"]
    widths = [1.1, 1.1, 1.5]
    labels = [r'$p=0, m=1$', r'$p=1, m=1$']
    p_color = ["r", "b"]
    plt.figure(figsize=(3.8, 3.2), dpi=300)
    for p in [0, 1]:
      data = pd.read_csv("results/sigma"+str(p+1)+".csv")
      sigma = data['y']
      w0 = np.min(sigma)
      w0_styles = ["-", "--"]
      plt.plot(r, np.abs(laguerre_gauss_radial(r, p, l, w0))**2,
              ls="-", 
              color=colors[p], 
              label=labels[p], 
              lw=widths[p])
      plt.plot(r, np.abs(laguerre_gauss_radial(r, p, l, 1))**2,
              ls="--", 
              color=colors[p], 
              lw=widths[p])
    plt.xlabel(r'$r$')
    plt.ylabel(r'$|T|^2$')
    plt.grid(False)
    plt.legend(borderaxespad=0.5, handletextpad=0.3, labelspacing=0.3, borderpad=0.2)
    plt.tight_layout()
    plt.savefig("media/radial.pdf", format="pdf", dpi=300)