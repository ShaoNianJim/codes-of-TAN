import pandas as pd
from Model.TAN import TAN
import networkx as nx
import matplotlib.pyplot as plt

"""
The root node of  TAN is  automatically selected from the data without the guidance of the expert
"""


# Read data for modeling
train_data = pd.read_excel("filepath/well_data.xlsx")
# Building TAN model
Tan = TAN()
Tan.fit(train_data, "label")
# visual TAN structure
graph = Tan
nodelist = graph.nodes
pos = nx.layout.shell_layout(graph)
pos = {'label': (-500, 1500), 'HG': (-100, 200), 'CRT': (400, 100),
       'ST': (1800, 250), 'FST': (-1400, 400),  'PS': (2500, 400),
       'RTH': (1000, 200), 'FP': (2000, 200)}
node_sizes = [1000 + 10 * i for i in range(len(graph))]
M = graph.number_of_edges()

edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
nodes = nx.draw_networkx_nodes(graph, pos, nodelist=nodelist, node_size=node_sizes, node_color="red", label='true')
edges = nx.draw_networkx_edges(graph, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=15, edge_color="blue",
                               edge_cmap=plt.cm.Blues, width=2)

labels = nx.draw_networkx_labels(graph, pos)
ax = plt.gca()
ax.set_axis_off()
plt.show()
