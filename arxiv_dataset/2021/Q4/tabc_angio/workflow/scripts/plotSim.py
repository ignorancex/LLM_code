import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['w','r'])

im = np.load(snakemake.input[0],allow_pickle=True)

plt.matshow(np.transpose(im),cmap=cmap)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(snakemake.output[0],format="pdf",bbox_inches="tight")
plt.show()
plt.close()
