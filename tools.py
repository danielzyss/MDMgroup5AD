import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as signal
import scipy.stats as stats
import seaborn as sns


def smoothDataFrame(dataframe):

    new_df = pd.DataFrame(columns=dataframe.columns)

    for i in range(0, dataframe.shape[0]):
        row =[]
        for j in dataframe.columns:

            EEG = dataframe[j][i]
            EEG = signal.savgol_filter(EEG, 101, 2, mode='nearest')
            row.append(EEG)

        new_df.loc[i] = row

    return new_df


def correlationDataFrame(dataframe):

    correlation = []

    for i in range(0, dataframe.shape[0]):
        correlMatrix = np.zeros(shape=(len(dataframe.columns), len(dataframe.columns)))

        for j, n in enumerate(dataframe.columns):

            for k, m in enumerate(dataframe.columns):

                c = stats.pearsonr(dataframe[n][i], dataframe[m][i])[0]
                correlMatrix[j,k] = c

        correlation.append(correlMatrix)

    return np.array(correlation)


def plotcorrelation(correlationarray):

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlationarray, vmax=1.0, square=True)
    f.tight_layout()
    plt.show()