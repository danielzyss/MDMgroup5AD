import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns
import networkx as nx
from scipy.spatial.distance import euclidean

from models.HierarchicalClustering import *
from models.AverageCorrelation import *


DictionnaryofPosition = {'AFz' : [55, 88], 'C1': [44, 55], 'C3': [33, 55],
                         'C4': [77, 55], 'C5': [22, 55], 'C6': (88, 55),
                         'F1': [47, 75], 'F3': [38, 76], 'F4': [72, 76],
                         'F5': [30, 80], 'F6': [80, 80], 'F7': [23, 86],
                         'F8': [87, 86], 'F9': [5, 75], 'P2': [63, 35],
                         'P4': [72, 34], 'P5': [30, 30], 'P6': [80, 30],
                         'AF2': [60, 88], 'AF4': [66, 88], 'AF6': [72, 90],
                         'Cz': [55, 55], 'F2': [63, 75], 'F10': [105, 75],
                         'AF1': [105, 75], 'AF3': [44,88], 'AF5': [38, 90],
                         'AF7': [33, 94], 'CP1': [44, 44], 'CP2': [66,44],
                         'CP3': [32, 41], 'CP4': [78, 41], 'CP5': [22, 36],
                         'CP6': [88, 36], 'CPz': [55,44], 'FC1': [44,66],
                         'FC3': [32, 69], 'FC5': [22, 74], 'FCz': [55,66],
                         'Fp2': [64, 102], 'L1': [45,2], 'L2': [65,2],
                         'O2': [64, 8], 'FT8': [95, 80], 'Fz': [55,77],
                         'Lz': [55,0], 'Nz': [55, 110], 'Oz': [55,11],
                         'PO1': [50, 22], 'PO10': [85, 8], 'PO2': [60, 22],
                         'PO3': [44,22], 'PO4': [66,22], 'PO5': [38, 20],
                         'PO7': [33,16], 'PO9': [25, 8], 'POz': [55,22],
                         'P1': [47, 35], 'P3': [38, 34], 'P7': [23,24],
                         'P9': [5,25], 'P10': [105, 25], 'T7': [11,55],
                         'T8': [99,55], 'T9': [0,55], 'T10': [110, 55],
                         'AF8': [77,94], 'FC2': [66,66], 'FC4': [78, 69],
                         'FC6': [88, 74], 'FT7': [15, 80], 'FT9': [2, 65],
                         'FT10': [108, 65], 'C2': [66,55], 'PO6': [72, 20],
                         'Pz': [55,33], 'TP7': [15, 30], 'TP8': [95, 30],
                         'TP9': [2, 35], 'TP10': [108, 35], 'Fp1' : [46,102],
                         'P8': [87, 24], 'O1':[46,8], 'PO8': [77,16]}


#1 create complete Graph
#2 find a way to put label in dataframe
#1.5 Create a geographic maps of the Electrode
#2 Obtain the correlation Matrix for the time series by probability
#4 weight graph with correlation
#5 find path from highest correlation to 5 nodes - 10 nodes - ...


def HighCorrelationfrequencyMatrix(slicedCorrelationArray):

    new_matrix = np.empty(shape=slicedCorrelationArray[0].shape)
    threshold = 0.2

    for i in range(new_matrix.shape[0]):

        for j in range(new_matrix.shape[1]):

            abovethreshold = 0
            for n in range(len(slicedCorrelationArray)):

                if slicedCorrelationArray[n][i,j] > threshold:
                    abovethreshold+=1
            frequency = abovethreshold/len(slicedCorrelationArray)
            new_matrix[i,j] = frequency

    return new_matrix

def createConnectedWeightedGraph(positiondict, probmat, label):

    G = nx.Graph()

    for n in positiondict.keys():
        G.add_node(n)

    for i in range(probmat.shape[0]):

        combination = []

        for j in range(probmat.shape[1]):

            if label[j] != label[i] and [label[i], label[j]] not in combination:

                combination.append([label[i], label[j]])
                combination.append([label[j], label[i]])
                w = euclidean(positiondict[label[i]], positiondict[label[j]])
                if w < 10:
                    w = w * (1- probmat[i,j])
                    G.add_edge(label[i],label[j], weight=w)
                else:
                    w = 999999 * (1- probmat[i,j])
                    G.add_edge(label[i], label[j], weight=w)

    return G

def findPotentialMostUsedPath(G, probmat, label):


    maxval = 0.0
    maxindex = []

    for i in range(0,probmat.shape[0]):
        for j in range(0,probmat.shape[1]):
            if i!=j:
                if probmat[i,j]>maxval:
                    maxindex = [i,j]
                    maxval = probmat[i,j]


    originA = label[maxindex[0]]
    originB = label[maxindex[1]]

    #path from A:

    path = nx.dijkstra_path(G, originA, originB)

    # node = originA
    # for i in range(0, pathlength):
    #     path.append(node)
    #     neighbors = G.neighbors(node)
    #     weights = np.array([G[i][x]['weight'] for x in neighbors])
    #     node = neighbors[np.argmin(weights)]

    return path



def SlicedCorrelationDataframe(dataframe, slices):
    correlation = []

    for i in range(0, dataframe.shape[0]):

        correlationslicearray = []

        splitedarray = [np.split(y, slices) for y in [x[:len(x) - 1] for x in dataframe.iloc[i, :].as_matrix()]]

        for sl in [np.array(splitedarray[:][n]) for n in range(0, slices)]:

            correlationslice = np.zeros(shape=(len(dataframe.columns), len(dataframe.columns)))

            for j in range(sl.shape[0]):

                for k in range(sl.shape[0]):
                    correlationslice[j, k] = abs(stats.pearsonr(sl[j], sl[k])[0])

            correlationslicearray.append(correlationslice)

        correlation.append(np.array(correlationslicearray))

    return np.array(correlation)