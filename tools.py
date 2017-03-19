import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns
import networkx as nx

from models.HierarchicalClustering import *
from models.AverageCorrelation import *
from models.splittedTimeSeries import *


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

def hierarchicalClustering(dataframe, labels):

    correlMatArray = correlationDataFrame(dataframe)
    graphArray = correlationmatrix2graph(correlMatArray)
    mstArray = graph2minimumspanningtree(graphArray)
    dendroarray = mst2dendrogram(mstArray, labels)

    return dendroarray


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


def plotcorrelation(correlationarray):

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlationarray, vmin=0.0, vmax=1.0, square=True, cmap="YlGnBu")
    f.tight_layout()
    plt.show()
