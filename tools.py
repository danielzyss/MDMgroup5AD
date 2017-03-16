import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as signal
import scipy.stats as stats
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.spatial.distance import euclidean
import pywt
import scipy.signal as sgl
from wavelet_transform import *

def smoothDataFrame(dataframe):

    new_df = pd.DataFrame(columns=dataframe.columns)

    for i in range(0, dataframe.shape[0]):
        row =[]
        for j in dataframe.columns:

            EEG = dataframe[j][i]
            EEG = waveletShrinkageDenoising(EEG)
            plt.plot(EEG)
            EEG = eliminateOutliers(EEG)
            plt.plot(EEG)
            EEG = signal.savgol_filter(EEG, 101, 2, mode='nearest')
            plt.plot(EEG)
            plt.show()
            row.append(EEG)

        new_df.loc[i] = row

    return new_df

def eliminateOutliers(EEG):

    newEEG = np.array([])

    for section in np.split(EEG[:EEG.shape[0]-1], 2):
        mu = np.mean(section)
        sigma = np.std(section)
        for i, p in enumerate(section):
            if p>(mu + 2*sigma):
                section[i] = mu + 2*sigma
            if p<(mu - 2*sigma):
                section[i] = mu - 2*sigma
        newEEG = np.concatenate((newEEG, section))

    return newEEG

def localnormalise(array, a, b ):
    newarray = np.array([])

    Xmin = np.min(array)
    Xmax = np.max(array)

    for X in array:
        Xnew = a + ((X - Xmin)*(b-a))/(Xmax - Xmin)
        newarray = np.append(newarray, Xnew)

    return newarray

def thresholdEstimation(X):

    l = len(X)
    sx2 = [sx * sx for sx in np.absolute(X)]
    sx2.sort()

    cumsumsx2 = np.cumsum(sx2)

    risks = []
    for i in range(0, l):
        risks.append((l - 2 * (i + 1) + (cumsumsx2[i] + (l - 1 - i) * sx2[i])) / l)
    mini = np.argmin(risks)
    th = np.sqrt(sx2[mini])



def waveletShrinkageDenoising(EEG):

    plt.plot(EEG, color='red')

    W = cwt(EEG, 0.1,np.arange(1,40), wf='morlet', p=2)

    threshold = np.var(EEG) * np.sqrt(2* np.log(len(EEG)))
    print(threshold)
    newW = []
    for w in W:
        neww = pywt.threshold(w, threshold, 'hard')
        newW.append(neww)

    signal = icwt(newW, 0.1, np.arange(1,40), wf='morlet', p=2)
    signal = localnormalise(signal)
    plt.plot(signal, color='blue')
    plt.show()

    return EEG



def correlationDataFrame(dataframe):

    correlation = []

    for i in range(0, dataframe.shape[0]):

        correlMatrix = np.zeros(shape=(len(dataframe.columns), len(dataframe.columns)))

        for j, n in enumerate(dataframe.columns):

            for k, m in enumerate(dataframe.columns):

                c = abs(stats.pearsonr(dataframe[n][i], dataframe[m][i])[0])
                correlMatrix[j,k] = c

        # correlMatrix = normalize(correlMatrix)
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
    sns.heatmap(correlationarray, vmin=0.0, vmax=1.0, square=True)
    f.tight_layout()
    plt.show()
