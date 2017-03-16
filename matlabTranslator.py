import pickle
import numpy as np
import pandas as pd
import scipy.io as io


def MatlabStructure2PandasDataframe(fileaddress, dictionnaryname, outputname):

    ADdata = io.loadmat(fileaddress)

    ADdict = ADdata[dictionnaryname]
    newDFeyesclosed = pd.DataFrame(columns=['electrode ' + str(i) for i in range(0, 64)])
    newDFeyesopened = pd.DataFrame(columns=['electrode ' + str(i) for i in range(0, 64)])

    for i in range(0, ADdict.shape[1]):
        patient = ADdict[0, i]

        # EYES CLOSED :
        newrow = []
        for j in range(0, 64):
            newrow.append(np.array(patient[0][j, :]))
        newDFeyesclosed.loc[i] = newrow

        # EYES OPENED :
        newrow = []
        for j in range(0, 64):
            newrow.append(np.array(patient[1][j, :]))
        newDFeyesopened.loc[i] = newrow

    newDFeyesclosed.to_pickle('data/' + outputname + 'eyesclosed.pk')
    newDFeyesopened.to_pickle('data/' + outputname + 'eyesopened.pk')

def loadDataFrames():

    ADeyesclosed = pd.read_pickle('data/ADeyesclosed.pk')
    ADeyesopened = pd.read_pickle('data/ADeyesopened.pk')
    CONTROLeyesclosed = pd.read_pickle('data/CONTROLeyesclosed.pk')
    CONTROLeyesopened = pd.read_pickle('data/CONTROLeyesopened.pk')

    return ADeyesclosed, ADeyesopened , CONTROLeyesclosed, CONTROLeyesopened


def ExtractLabels():

    control = io.loadmat('data/Data_from_Cntl.mat')
    labels = [x for x in control['Data_from_Cntl'][0][3][3]]
    labels = [l[0] for l in labels[0]]

    return(np.array(labels))
