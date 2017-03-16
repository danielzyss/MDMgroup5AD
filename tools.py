import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from preprocessing import *

def preprocess(dataframe):

    new_df = pd.DataFrame(columns=dataframe.columns)

    for i in range(0, dataframe.shape[0]):
        row =[]
        for j in dataframe.columns:

            EEG = dataframe[j][i]
            EEG = eliminateOutliers(EEG)
            #EEG = SGSmoothing(EEG)
            EEG = waveletShrinkageDenoising(EEG)

            row.append(EEG)

        new_df.loc[i] = row

    return new_df


def correlationDataFrame(dataframe):

    correlation = []

    for i in range(0, dataframe.shape[0]):

        correlMatrix = np.zeros(shape=(len(dataframe.columns), len(dataframe.columns)))

        for j, n in enumerate(dataframe.columns):

            for k, m in enumerate(dataframe.columns):

                c = abs(stats.pearsonr(dataframe[n][i], dataframe[m][i])[0])
                correlMatrix[j,k] = c

        correlation.append(correlMatrix)

    return np.array(correlation)


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


def mst2dendrogram(mstarray):

    clusteringarray = []

    for mst in mstarray:
        distance = nx.floyd_warshall_numpy(mst)
        linkagematrix = linkage(distance)
        dendro = dendrogram(linkagematrix, truncate_mode='mlab')
        clusteringarray.append(dendro)
        plt.show()

    return clusteringarray


def plotcorrelation(correlationarray):

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlationarray, vmin=0.0, vmax=1.0, square=True, cmap="YlGnBu")
    f.tight_layout()
    plt.show()
