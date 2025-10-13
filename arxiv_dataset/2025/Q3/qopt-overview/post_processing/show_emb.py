#!/usr/bin/env ipython

import json
import matplotlib.pyplot as plt
from dwave_networkx import draw_pegasus_embedding, pegasus_graph

# Load the embedding
with open("./run_logs/dwave/embeddings/MWC3.emb.json", "r") as f:
    embjs = json.load(f)

embedding = {int(k): v for k, v in embjs.items()}

G = pegasus_graph(16)

# make an ad-hoc Viridis colormap
cmap = plt.cm.viridis
num_colors = len(embedding)
colors = {j: cmap(j / num_colors) for j in range(num_colors)}

plt.figure(figsize=(12, 12))
draw_pegasus_embedding(G, emb=embedding,
                       chain_color = colors,
                        show_labels=True, crosses = True,
                        font_color='white')
plt.show()
