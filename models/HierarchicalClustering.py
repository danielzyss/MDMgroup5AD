import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

def correlationmatrix2graph(correlmatrixarray):
    grapharray = []

    for correlmatrix in correlmatrixarray:
        gr = nx.from_numpy_matrix(correlmatrix, parallel_edges=False)
        grapharray.append(gr)

    return grapharray


def graph2minimumspanningtree(grapharray):

    minimumspanningtreearray = []

    for graph in grapharray:
        mst = nx.minimum_spanning_tree(graph)
        minimumspanningtreearray.append(mst)

    return minimumspanningtreearray


def mst2dendrogram(mstarray, lab):

    clusteringarray = []

    for mst in mstarray:
        distance = nx.floyd_warshall_numpy(mst)
        linkagematrix = linkage(distance)
        dendro = dendrogram(linkagematrix, truncate_mode='mlab', labels=lab )
        clusteringarray.append(dendro)
        plt.show()

    return clusteringarray