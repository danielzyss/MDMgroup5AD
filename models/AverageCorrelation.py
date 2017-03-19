import pandas as pd
import numpy as np
import sys
import seaborn as sns

from preprocessing import *


def AverageCorrelationMatrix(correlMatrixArray):

    avgmat = np.empty(shape=correlMatrixArray[0].shape)

    for row in range(0,correlMatrixArray[0].shape[0]):
        for col in range(0,correlMatrixArray[0].shape[1]):
            avgmat[row, col] = np.mean(np.array([x[row, col] for x in correlMatrixArray]))

    return avgmat


def ReinforcedAverageCorrelationMatrix(correlMatrixArray):

    avgmat = np.empty(shape=correlMatrixArray[0].shape)

    for row in range(0, correlMatrixArray[0].shape[0]):
        for col in range(0, correlMatrixArray[0].shape[1]):
            vals = np.array([x[row, col] for x in correlMatrixArray])
            for i, v in enumerate(vals):
                if v >0.7:
                    vals[i]=1.0
                elif v>0.3 and v<0.7:
                    vals[i]=0.5
                else:
                    vals[i]=0.0
            avgmat[row, col] = np.mean(vals)

    return avgmat