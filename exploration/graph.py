import random
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(nodes):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add the root node
    G.add_node('OCMs')

    # Add children nodes
    # for i in range(1, num_nodes):
    for node in nodes:
        key, values = node.items()
        # parent = random.randint(0, i-1)
        G.add_node(key)
        for val in values:
            G.add_edge(key, ' '.join(val))

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

# Example usage: create a graph with 10 nodes
# create_graph(10)