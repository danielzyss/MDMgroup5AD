import pandas as pd
import numpy as np
import scipy.signal as sgl
import pywt
import matplotlib.pyplot as plt
import sys

def eliminateOutliers(EEG):

    newEEG = np.array([])

    for section in np.split(EEG[:EEG.shape[0]], 10):
        mu = np.mean(section)
        sigma = np.std(section)
        for i, p in enumerate(section):
            if p>(mu + 2*sigma):
                section[i] = mu + 2*sigma
            if p<(mu - 2*sigma):
                section[i] = mu - 2*sigma
        newEEG = np.concatenate((newEEG, section))

    return newEEG

def localnormalise(array, a, b ):
    newarray = np.array([])

    Xmin = np.min(array)
    Xmax = np.max(array)

    for X in array:
        Xnew = a + ((X - Xmin)*(b-a))/(Xmax - Xmin)
        newarray = np.append(newarray, Xnew)

    return newarray

def matrixnormalise(matrix, a, b):
    newmatrix = np.empty(shape=matrix.shape)
    Xmin = np.min(matrix)
    Xmax = np.max(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            newmatrix[i,j] = a + ((matrix[i,j]-Xmin)*(b-a))/(Xmax-Xmin)

    return newmatrix

def WaveletthresholdEstimation(X):

    l = len(X)
    sx2 = [sx * sx for sx in np.absolute(X)]
    sx2.sort()

    cumsumsx2 = np.cumsum(sx2)

    risks = []
    for i in range(0, l):
        risks.append((l - 2 * (i + 1) + (cumsumsx2[i] + (l - 1 - i) * sx2[i])) / l)
    mini = np.argmin(risks)
    th = np.sqrt(sx2[mini])

    return th

def waveletShrinkageDenoising(EEG):

    W = cwt(EEG, 0.1,np.arange(1,31), wf='morlet', p=1)

    threshold = WaveletthresholdEstimation(EEG)

    newW = []
    for w in W:
        neww = pywt.threshold(w, threshold, 'hard')
        newW.append(neww)

    signal = icwt(newW, 0.1, np.arange(1,31), wf='morlet', p=1)

    return signal

def cwt(x, dt, scales, wf='dog', p=2):
    """Continuous Wavelet Tranform.

    :Parameters:
       x : 1d array_like object
          data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter ('omega0' for morlet, 'm' for paul
          and dog)

    :Returns:
       X : 2d numpy array
          transformed data
    """

    x_arr = np.asarray(x) - np.mean(x)
    scales_arr = np.asarray(scales)

    if x_arr.ndim is not 1:
        raise ValueError('x must be an 1d numpy array of list')

    if scales_arr.ndim is not 1:
        raise ValueError('scales must be an 1d numpy array of list')

    w = angularfreq(N=x_arr.shape[0], dt=dt)

    if wf == 'morlet':
        wft = morletft(s=scales_arr, w=w, w0=p, dt=dt)
    else:
        raise ValueError('wavelet function is not available')

    X_ARR = np.empty((wft.shape[0], wft.shape[1]), dtype=np.complex128)

    x_arr_ft = np.fft.fft(x_arr)

    for i in range(X_ARR.shape[0]):
        X_ARR[i] = np.fft.ifft(x_arr_ft * wft[i])

    return X_ARR

def icwt(X, dt, scales, wf='dog', p=2):
    """Inverse Continuous Wavelet Tranform.
    The reconstruction factor is not applied.

    :Parameters:
       X : 2d array_like object
          transformed data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter

    :Returns:
       x : 1d numpy array
          data
    """

    X_arr = np.asarray(X)
    scales_arr = np.asarray(scales)

    if X_arr.shape[0] != scales_arr.shape[0]:
        raise ValueError('X, scales: shape mismatch')

    # See (11), (13) at page 68
    X_ARR = np.empty_like(X_arr)
    for i in range(scales_arr.shape[0]):
        X_ARR[i] = X_arr[i] / np.sqrt(scales_arr[i])

    x = np.sum(np.real(X_ARR), axis=0)

    return x

def angularfreq(N, dt):
    """Compute angular frequencies.

    :Parameters:   
       N : integer
          number of data samples
       dt : float
          time step

    :Returns:
        angular frequencies : 1d numpy array
    """

    # See (5) at page 64.

    N2 = N / 2.0
    w = np.empty(N)

    for i in range(w.shape[0]):
        if i <= N2:
            w[i] = (2 * np.pi * i) / (N * dt)
        else:
            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w

def morletft(s, w, w0, dt):
    """Fourier tranformed morlet function.

    Input
      * *s*    - scales
      * *w*    - angular frequencies
      * *w0*   - omega0 (frequency)
      * *dt*   - time step
    Output
      * (normalized) fourier transformed morlet function
    """

    p = 0.75112554446494251  # pi**(-1.0/4.0)
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = w > 0

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0) ** 2 / 2.0)

    return wavelet

def normalization(s, dt):
    PI2 = 2 * np.pi
    return np.sqrt((PI2 * s) / dt)

def SGSmoothing(EEG):
    return sgl.savgol_filter(EEG,101,2)