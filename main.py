import pickle
import numpy as np
import pandas as pd
from matlabTranslator import *
from tools import *


print('Importing Data.. ')
ADeyesclosed, ADeyesopened, CNTLeyesclosed, CNTLeyesopened = loadDataFrames()
labels = ExtractLabels()

if __name__ == '__main__':

    print('Preprocessing..')
    ADeyesclosedPreprocessed = preprocess(ADeyesclosed.head(10))

    print('Dendrogram')
    dendrogramArray = hierarchicalClustering(ADeyesclosedPreprocessed, labels)












