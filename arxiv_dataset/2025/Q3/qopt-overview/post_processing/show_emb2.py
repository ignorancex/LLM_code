#!/usr/bin/env ipython

import json
import dwave_networkx as dnx
import networkx as nx
import matplotlib.pyplot as plt

# Load embedding from JSON
with open('./run_logs/dwave/embeddings/MWC84.emb.json', 'r') as f:
    embedding = json.load(f)

# Create Pegasus graph (adjust M based on your system)
M = 16
pegasus_graph = dnx.pegasus_graph(M, nice_coordinates=True)

# Extract used qubits
used_qubits = set()
for chain in embedding.values():
    used_qubits.update(chain)

# Create subgraph
G = pegasus_graph.subgraph(used_qubits).copy()

# Tag chain edges
for logical_var, chain in embedding.items():
    for i in range(len(chain) - 1):
        if G.has_edge(chain[i], chain[i + 1]):
            G[chain[i]][chain[i + 1]]['chain'] = logical_var

# Get positions
pos = nx.get_node_attributes(pegasus_graph, 'pos')
pos = {q: pos[q] for q in used_qubits}

# Plot
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', label='Qubits')
nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5)
colors = plt.cm.tab10.colors
for idx, (logical_var, chain) in enumerate(embedding.items()):
    chain_edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1) if G.has_edge(chain[i], chain[i + 1])]
    nx.draw_networkx_nodes(G, pos, nodelist=chain, node_size=100, node_color=colors[idx % len(colors)], label=f'Logical {logical_var}')
    nx.draw_networkx_edges(G, pos, edgelist=chain_edges, edge_color=colors[idx % len(colors)], width=2)

nx.draw_networkx_labels(G, pos, font_size=6)
plt.legend()
plt.title("Pegasus Embedding Visualization")
#plt.savefig('embedding.png')
plt.show()
