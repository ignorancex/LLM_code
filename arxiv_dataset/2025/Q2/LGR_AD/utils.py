import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler

def read_data(path):
    # Load the dataset
    data = pd.read_csv(path)
    scaler = StandardScaler()
    # Drop any rows with missing values
    X = data.drop(data.columns[-1], axis=1)
    X=pd.DataFrame(scaler.fit_transform(X))
    y = data[data.columns[-1]]

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=round(time.time()))

    return X_train, X_test, y_train, y_test

def display_outliers(outliers):
    print("start")
    for element in outliers:
         print(element)
    print("finish")

def visualize_graph():
    import networkx as nx
    import matplotlib.pyplot as plt

    # Define the graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node(1, pos=(0, 0))
    G.add_node(2, pos=(1, 1))
    G.add_node(3, pos=(1, -1))
    G.add_node(4, pos=(2, 0))

    # Add edges with weights
    G.add_edge(1, 2, weight=2)
    G.add_edge(1, 3, weight=1)
    G.add_edge(1, 4, weight=3)
    G.add_edge(2, 3, weight=2)
    G.add_edge(2, 4, weight=4)

    # Get positions of nodes for plotting
    pos = nx.get_node_attributes(G, 'pos')

    # Get weights of edges for plotting
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # Set node colors and sizes
    node_colors = ['lightgreen', 'grey', 'red', 'lightblue']
    node_sizes = [800, 600, 600, 800]

    # Set edge colors and widths
    edge_colors = ['black', 'black', 'black', 'black', 'black']
    edge_widths = [2, 2, 2, 2, 2]

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_family='sans-serif')

    # Set plot limits and display the graph
    plt.xlim((-0.5, 2.5))
    plt.ylim((-1.5, 1.5))
    plt.axis('off')
    plt.show()


visualize_graph()
