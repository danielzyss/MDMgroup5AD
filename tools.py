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

    print('Eliminating Outliers + Denoising..')


    for i in dataframe.index:
        row =[]
        for j in dataframe.columns:
            EEG = dataframe.get_value(i,j)
            EEG = eliminateOutliers(EEG)
            if smoothing == 'SG':
                EEG = SGSmoothing(EEG)
            elif smoothing == 'WSD':
                EEG = waveletShrinkageDenoising(EEG)

            row.append(EEG)

        new_df.loc[i] = row

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
    dendroarray, linkagematrix = mst2dendrogram(mstArray, labels)

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

    print('Plotting..')
    plotPath(paths, positionDict)

    return paths

def correlationDataFrame(dataframe):

    correlation = []

    for i in dataframe.index:

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



