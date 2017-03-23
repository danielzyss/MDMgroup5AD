import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns
import networkx as nx

from models.HierarchicalClustering import *
from models.AverageCorrelation import *
from models.PathAnalysis import *

def preprocess(dataframe, smoothing):

    new_df = pd.DataFrame(columns=dataframe.columns)

    print('Eliminating Outliers..')

    for i in range(0, dataframe.shape[0]):
        row =[]
        for j in dataframe.columns:

            EEG = dataframe[j][i]
            if smoothing == 'SG':
                EEG = SGSmoothing(EEG)
            elif smoothing == 'WSD':
                EEG = waveletShrinkageDenoising(EEG)

            row.append(EEG)

        new_df.loc[i] = row

    print('Denoising..')

    return new_df

def CorrelationAveragingModel(dataframe):

    print('Correlation Matrix..')
    correlMatArray = correlationDataFrame(dataframe)

    print('Averaging..')
    corelmat = ReinforcedAverageCorrelationMatrix(correlMatArray)

    return corelmat


def hierarchicalClusteringModel(dataframe, labels):

    print('Correlation Matrix..')
    correlMatArray = correlationDataFrame(dataframe)
    print('Graphing..')
    graphArray = correlationmatrix2graph(correlMatArray)
    print('Minimum Spanning Tree..')
    mstArray = graph2minimumspanningtree(graphArray)
    print('Dendrogram.. ')
    dendroarray = mst2dendrogram(mstArray, labels)

    return dendroarray

def BayesianPathModel(dataframe, slices, pathnb, positionDict, labels):

    print('Preprocessing...')
    PreprocessedSG = preprocess(dataframe, 'SG')
    PreprocessedWSD = preprocess(dataframe, 'WSD')

    print('General Correlation..')
    generalCorrelMatrix = correlationDataFrame(PreprocessedWSD)

    print('Sliced Correlation..')
    slicedCorrelArray = SlicedCorrelationDataframe(PreprocessedSG, slices)

    print('Bayesian Probability Matrix..')
    BayesianMatrix = HighCorrelationfrequencyMatrix(slicedCorrelArray, generalCorrelMatrix)

    print('Connected Weighted Graph..')
    G = createConnectedWeightedGraph(positionDict, BayesianMatrix, labels)

    print('Paths..')
    paths = findPotentialMostUsedPath(G, BayesianMatrix, labels, pathnb)

    return paths


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


def plotcorrelation(correlationarray, labels):

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlationarray, vmin=0.0, vmax=1.0, square=True, cmap="YlGnBu", yticklabels=False, xticklabels=False)

    plt.yticks(range(len(labels)), labels.tolist(), size='small')
    plt.xticks(range(len(labels)), labels.tolist(), size='small', rotation=90)
    f.tight_layout()

    plt.show()



