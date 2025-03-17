import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import networkx as nx

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples)
Y = 0.5 * X + np.random.randn(n_samples) * 0.1
Z = 0.3 * X + 0.4 * Y + np.random.randn(n_samples) * 0.1
data = np.column_stack((X, Y, Z))

# Define variable names
variable_names = ['X', 'Y', 'Z']

# Apply PC algorithm for causal discovery
causal_graph = pc(data, alpha=0.05)
# Print the learned graph
print(causal_graph.G)

# Build a NetworkX graph from the adjacency matrix
def build_nx_graph(causal_graph, labels):
    import networkx as nx
    G = nx.DiGraph()
    num_nodes = len(labels)
    G.add_nodes_from(range(num_nodes))
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_type = causal_graph.G.graph[i][j]
            if edge_type == 1:  # Directed edge from i to j
                G.add_edge(i, j)
            elif edge_type == -1:  # Directed edge from j to i
                G.add_edge(j, i)
            elif edge_type == 2:  # Undirected edge
                G.add_edge(i, j)
                G.add_edge(j, i)
    # Relabel nodes
    mapping = {i: label for i, label in enumerate(labels)}
    G = nx.relabel_nodes(G, mapping)
    return G

# Build the graph
G = build_nx_graph(causal_graph, variable_names)

# Plot the causal graph
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1500,
    node_color='lightblue',
    arrowsize=20,
    font_size=12,
    font_weight='bold'
)
plt.title('Causal Graph')
plt.show()
