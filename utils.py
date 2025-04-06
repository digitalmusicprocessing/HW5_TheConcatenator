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

def stochastic_universal_sample(ws, target_points):
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
    target_points: int
        The number of desired samples
    
    Returns
    -------
    ndarray(P, dtype=int)
        Indices of sampled particles, with replacement
    ndarray(P)
        Weights of the new samples
    """
    counts = np.zeros(ws.size)
    w = np.zeros(ws.size+1)
    order = np.random.permutation(ws.size)
    w[1::] = ws.flatten()[order]
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = min(idx, ws.size-1) # Safeguard
        counts[order[idx]] += 1
        p += 1/target_points
        if p >= 1:
            p -= 1
            idx = 0
    ws_new = np.zeros(ws.size)
    choices = np.zeros(ws.size)
    idx = 0
    for i in range(len(counts)):
        for w in range(int(counts[i])):
            choices[idx] = i
            ws_new[idx] = ws[i]/counts[i]
            idx += 1
    ws_new /= np.sum(ws_new)
    return choices, ws_new