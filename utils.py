"""
Code copyright Christopher J. Tralie, 2024
Attribution-NonCommercial-ShareAlike 4.0 International


Share — copy and redistribute the material in any medium or format
The licensor cannot revoke these freedoms as long as you follow the license terms.

 Under the following terms:
    Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
    NonCommercial — You may not use the material for commercial purposes .
    NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.
    No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

import numpy as np

DB_MIN = -1000

hann_window = lambda N: (0.5*(1 - np.cos(2*np.pi*np.arange(N)/N))).astype(np.float32)

def load_corpus(path, sr, dc_normalize=True, amp_normalize=True):
    """
    Load a corpus of audio

    Parameters
    ----------
    path: string
        Path to folder or file
    sr: int
        Sample rate to use
    dc_normalize: bool
        If True, do a DC-offset normalization on each clip
    amp_normalize: bool
        If True (default), normalize the audio sample range to be in [-1, 1]
    
    Returns
    -------
    ndarray(n_samples)
        The audio samples
    """
    import glob
    import os
    import librosa
    from warnings import filterwarnings
    filterwarnings("ignore", message="librosa.core.audio.__audioread_load*")
    filterwarnings("ignore", message="PySoundFile failed.*")
    samples = []
    files = [path]
    if os.path.isdir(path):
        files = glob.glob(path + os.path.sep + "**", recursive=True)
        files = [f for f in files if os.path.isfile(f)]
    N = 0
    for f in sorted(files):
        try:
            try:
                x, sr = librosa.load(f, sr=sr, mono=True)
            except:
                print("Error loading", f)
                continue
            N += x.size
            if dc_normalize:
                x -= np.mean(x)
            if amp_normalize:
                norm = np.max(np.abs(x))
                if norm > 0:
                    x = x/norm
            samples.append(x)
        except:
            pass
    if len(samples) == 0:
        print("Error: No usable files found at ", path)
    assert(len(samples) > 0)
    return np.concatenate(samples)

def get_mel_filterbank(sr, win, mel_bands=40, min_freq=0.0, max_freq=8000):
    """
    Return a mel-spaced triangular filterbank

    Parameters
    ----------
    sr: int
        Audio sample rate
    win: int
        Window length
    mel_bands: int
        Number of bands to use
    min_freq: float 
        Minimum frequency for Mel filterbank
    max_freq: float
        Maximum frequency for Mel filterbank

    Returns
    -------
    ndarray(mel_bands, win//2+1)
        Mel filterbank, with each filter per row
    """
    melbounds = np.array([min_freq, max_freq])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], mel_bands+2)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((win-1)/float(sr))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)
    # Create mel triangular filterbank
    melfbank = np.zeros((mel_bands, win//2+1), dtype=np.float32)
    for i in range(1, mel_bands+1):
        thisbin = binbins[i]
        lbin = binbins[i-1]
        rbin = thisbin + (thisbin - lbin)
        rbin = binbins[i+1]
        melfbank[i-1, lbin:thisbin+1] = np.linspace(0, 1, 1 + (thisbin - lbin))
        melfbank[i-1, thisbin:rbin+1] = np.linspace(1, 0, 1 + (rbin - thisbin))
    melfbank = melfbank/np.sum(melfbank, 1)[:, None]
    return melfbank

def stochastic_universal_sample(ws):
    """
    Resample indices according to universal stochastic sampling.
    Also known as "systematic resampling"
    [1] G. Kitagawa, "Monte Carlo filter and smoother and non-Gaussian nonlinear
    state space models," J. Comput. Graph. Stat., vol. 5, no. 1, pp. 1-25, 1996.
    [2] J. Carpenter, P. Clifford, and P. Fearnhead, "An improved particle filter [...]"

    Parameters
    ----------
    ndarray(P)
        The normalized weights of the particles
    
    Returns
    -------
    ndarray(P, dtype=int)
        Indices of sampled particles, with replacement
    """
    counts = np.zeros(ws.size)
    w = np.zeros(ws.size+1)
    order = np.random.permutation(ws.size)
    w[1::] = ws.flatten()[order]
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(len(ws)):
        while not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = min(idx, ws.size-1) # Safeguard
        counts[order[idx]] += 1
        p += 1/len(ws)
        if p >= 1:
            p -= 1
            idx = 0
    choices = np.zeros(ws.size, dtype=int)
    idx = 0
    for i in range(len(counts)):
        for w in range(int(counts[i])):
            choices[idx] = i
            idx += 1
    return choices

def do_nmf_kl(V, W, n_iters):
    """
    Perform Kullback-Liebler nonnegative matrix factorization
    
    Parameters
    ----------
    V: ndarray(win, n_frames), nonnegative
        Spectrogram amplitudes to factorize
    n_iters: int
        Number of iterations to perform
    
    Returns
    -------
    H: ndarray(K, n_frames), nonnegative
        The activations over time
    """
    p = W.shape[1]
    H = np.random.rand(p, V.shape[1])
    ## Apply multiplicative update rules n_iters times in a loop
    for i in range(n_iters):
        VL = V/(W.dot(H))
        H *= (W.T).dot(VL)/np.sum(W, 0)[:, None]
    return H

def get_kl_fit(V, W, H):
    """
    Compute the KL distance for a NMF

    Parameters
    ----------
    V: ndarray(win, n_frames), nonnegative
        Target spectrogram that's being factorized
    W: ndarray(win, K), nonnegative
        The spectral template columns
    H: ndarray(K, n_frames), nonnegative
        The activations over time
    
    Returns
    -------
    float: KL fit
    """
    Vi = W.dot(H)
    Vi[Vi == 0] = 1
    logarg = V/Vi
    logarg[logarg == 0] = 1
    return np.mean(V*np.log(logarg) - V + Vi)
