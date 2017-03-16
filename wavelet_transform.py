import numpy as np

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
