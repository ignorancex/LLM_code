import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import tempfile


# Magic function which converts currently plotted data into an actual image
# This is required for Spylon_kernel
def plotfig_magic():
    fo = tempfile.NamedTemporaryFile(suffix=".png", mode="w")
    fo.close()
    plt.savefig(fo.name)
    return Image(filename=fo.name)