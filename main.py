import pickle
import numpy as np
import pandas as pd
from matlabTranslator import *
from tools import *




print('Importing Data.. ')
ADeyesclosed, ADeyesopened, CNTLeyesclosed, CNTLeyesopened = loadDataFrames()

if __name__ == '__main__':

    print('Smoothing..')
    ADeyesclosedSmoothed = smoothDataFrame(ADeyesclosed)

    print('Creating Correlation Matrix..')
    correlMatArray = correlationDataFrame(ADeyesclosedSmoothed.head())

    # print('Plotting Correlation Matrix for Patient 0')
    # plotcorrelation(pd.DataFrame(correlMatArray[0]))

    print('Creating Graphs..')
    graphArray = correlationmatrix2graph(correlMatArray)

    print('Minimum Spanning Tree..')
    mstArray = graph2minimumspanningtree(graphArray)

    print('Dendrawing.. ')
    dendroarray = mst2dendrogram(graphArray)











