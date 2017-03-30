import pickle
import numpy as np
import pandas as pd
from matlabTranslator import *
from tools import *
import sys
from models.AverageCorrelation import  *

DictionnaryofPosition = {'AFz' : [55, 88], 'C1': [44, 55], 'C3': [33, 55],
                         'C4': [77, 55], 'C5': [22, 55], 'C6': (88, 55),
                         'F1': [47, 75], 'F3': [38, 76], 'F4': [72, 76],
                         'F5': [30, 80], 'F6': [80, 80], 'F7': [23, 86],
                         'F8': [87, 86], 'F9': [5, 75], 'P2': [63, 35],
                         'P4': [72, 34], 'P5': [30, 30], 'P6': [80, 30],
                         'AF2': [60, 88], 'AF4': [66, 88], 'AF6': [72, 90],
                         'Cz': [55, 55], 'F2': [63, 75], 'F10': [105, 75],
                         'AF1': [50, 88], 'AF3': [44,88], 'AF5': [38, 90],
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

if __name__ == '__main__':

    print('Importing Data.. ')
    ADeyesclosed, ADeyesopened, CNTLeyesclosed, CNTLeyesopened = loadDataFrames()
    labels = ExtractLabels()

    #MODEL 1: Averaged Correlation

    ADeyesclosed_preprocessed = preprocess(ADeyesclosed, 'WSD')
    ADeyesclosed_Averaged_Correlation = CorrelationAveragingModel(ADeyesclosed_preprocessed)
    plotcorrelation(ADeyesclosed_Averaged_Correlation, labels)

    #MODEL 2: Hierarchical Clusteting

    ADeyesclosed_preprocessed = preprocess(ADeyesclosed, 'WSD')
    dendrogram_arrays = hierarchicalClusteringModel(ADeyesclosed_preprocessed, labels)

    #MODEL 3: Bayesian Path Analysis

    list_of_paths = BayesianPathModel(ADeyesclosed, 100, 30, DictionnaryofPosition, labels)
    print(list_of_paths)