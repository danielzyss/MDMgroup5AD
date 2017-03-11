import pickle
import numpy as np
import pandas as pd
from matlabTranslator import *
from tools import *


#data import
print('Importing Data.. ')
ADeyesclosed, ADeyesopened, CNTLeyesclosed, CNTLeyesopened = loadDataFrames()

if __name__ == '__main__':

    print('Smoothing..')
    ADeyesclosedSmoothed = smoothDataFrame(ADeyesclosed)
    print('Creating Correlation Matrix..')
    correlMat = correlationDataFrame(ADeyesclosedSmoothed)
    print('Plotting Correlation Matrix for Patient 0')
    plotcorrelation(pd.DataFrame(correlMat[0]))







