import pickle
import numpy as np
import pandas as pd
from matlabTranslator import *
from tools import *
import sys

print('Importing Data.. ')
ADeyesclosed, ADeyesopened, CNTLeyesclosed, CNTLeyesopened = loadDataFrames()
labels = ExtractLabels()

if __name__ == '__main__':

    print('Preprocessing..')
    ADeyesclosedPreprocessed = preprocess(ADeyesclosed.head(1))

    # print('Model : Hierarchical Clustering')
    # dendrogramArray = hierarchicalClustering(ADeyesclosedPreprocessed, labels)

    path = PathAnalysisModel(ADeyesclosed.head(1), labels)
    print(path)


