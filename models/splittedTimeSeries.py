import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns
import networkx as nx

from preprocessing import *


def SlicedCorrelationDataframe(dataframe, slices):

    correlation = []

    for i in range(0, dataframe.shape[0]):

        correlationslicearray = []

        splitedarray = [np.split(y, slices) for y in [x[:len(x)-1] for x in dataframe.iloc[i,:].as_matrix()]]

        for sl in [np.array(splitedarray[:][n]) for n in range(0, slices)]:

            correlationslice = np.zeros(shape=(len(dataframe.columns), len(dataframe.columns)))

            for j in range(sl.shape[0]):

                for k in range(sl.shape[0]):

                    correlationslice[j,k] = abs(stats.pearsonr(sl[j], sl[k])[0])

            correlationslicearray.append(correlationslice)

        correlation.append(np.array(correlationslicearray))

    return np.array(correlation)