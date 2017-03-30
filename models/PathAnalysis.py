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
from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as patches
import tqdm


def plotElectrodes(PosDict):

    for key in PosDict.keys():
        plt.scatter(PosDict[key][0], PosDict[key][1])

    for key in PosDict.keys():
        legend = key + ': (' + str(PosDict[key][0]) + ',' + str(PosDict[key][1]) + ')'
        x = PosDict[key][0]
        y = PosDict[key][1]
        plt.annotate(legend, xy=(x,y),  textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'), fontsize=5)
    plt.xlim(-20, 170)
    plt.ylim(-20, 190)
    plt.xticks([])
    plt.yticks([])
    plt.title('Coordinates and physical disposition of the 64 electrodes')
    plt.savefig('ElectrodeCoordinates.png', dpi=300)

def HighCorrelationfrequencyMatrix(slicedCorrelationArray, priorMatArray):

    priorMat = np.empty(shape=priorMatArray[0].shape)
    for i in range(0, len(priorMatArray[0])):
        for j in range(0, len(priorMatArray[0])):
            aboveth = 0
            for p in range(len(priorMatArray)):
                if priorMatArray[p][i,j] > np.mean(priorMatArray[p]):
                    aboveth+=1
            priorMat[i,j] = aboveth/len(priorMatArray)


    bayesian_matrix = np.empty(shape=priorMat.shape)

    for i in range(0, bayesian_matrix.shape[0]):
        for j in range(0, bayesian_matrix.shape[1]):

            prior = priorMat[i,j]
            aboveth = 0
            for p in range(0,len(slicedCorrelationArray)):
                for s in range(0, len(slicedCorrelationArray[0])):
                    if slicedCorrelationArray[p][s][i,j] > np.mean(slicedCorrelationArray[p][s]):
                        aboveth +=1

            prob = aboveth/ (len(slicedCorrelationArray) * len(slicedCorrelationArray[0]))
            bayesian_matrix[i,j] = (prior * prob)/0.5

    bayesian_matrix = matrixnormalise(bayesian_matrix, 0, 1)
    return bayesian_matrix


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
                print(probmat[i,j])
                w = w*(1/probmat[i,j]) - 1
                G.add_edge(label[i], label[j], weight=w)

    return G

def findPotentialMostUsedPath(G, probmat, label, nb):

    paths = []
    combination = []

    for n in range(0, nb):

        maxval = 0.0
        maxindex = []

        for i in range(0,probmat.shape[0]):
            for j in range(0,probmat.shape[1]):
                if (i!=j) and ([i,j] not in combination) and (probmat[i,j]>maxval):
                    maxindex = [i,j]
                    maxval = probmat[i,j]
                    combination.append([i,j])
                    combination.append([j,i])

        A = label[maxindex[0]]
        B = label[maxindex[1]]

        paths.append(nx.dijkstra_path(G, A, B))

    return paths



def SlicedCorrelationDataframe(dataframe, slices):

    correlation = []

    for i in range(0, dataframe.shape[0]):

        correlationslicearray = []

        splitedarray = [np.array(np.split(y, slices)) for y in [x for x in dataframe.iloc[i, :].as_matrix()]]

        slicesarray = []
        for n in range(0, slices):
            slicesarray.append(np.array([x[n] for x in splitedarray]))

        for sl in slicesarray:
            correlationslice = np.zeros(shape=(len(sl), len(sl)))

            for j in range(len(sl)):
                for k in range(len(sl)):

                    correlationslice[j, k] = abs(stats.pearsonr(sl[j], sl[k])[0])

            correlationslicearray.append(correlationslice)

        correlation.append(np.array(correlationslicearray))

    return np.array(correlation)


def plotPath(paths, PosDict):

    fig = plt.figure()

    Z = [[0, 0], [0, 0]]
    levels = np.linspace(0,1,len(paths))
    CS3 = plt.contourf(Z, levels, cmap=cm.viridis)
    fig.clf()

    ax = fig.add_subplot('111')

    for e in PosDict.keys():
        ax.scatter(PosDict[e][0], PosDict[e][1], marker='x', s=30, color='k')

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(paths))]

    for coef, path in enumerate(paths):
        coef = len(colors) - 1 - coef
        for p in range(0, len(path) -1):
            x = [PosDict[path[p]][0], PosDict[path[p+1]][0]]
            y = [PosDict[path[p]][1], PosDict[path[p+1]][1]]
            ax.plot(x, y, color=colors[coef], alpha=0.8, linewidth = 5.0, marker='o')

    circle = patches.Circle(xy=[55,55], radius=60, edgecolor="k", facecolor="none")
    ax.add_patch(circle)

    fig.colorbar(CS3)
    plt.show()

