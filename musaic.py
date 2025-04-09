import numpy as np
import matplotlib.pyplot as plt
from utils import get_mel_filterbank, do_nmf_kl, hann_window, stochastic_universal_sample


def musaic(corpus, ytarget, sr, win, p, pd=0.95, temperature=50, n_particles=100, mel_bands=100, fmax=4000):
    """
    Implement a vanilla version of "The Concatenator"[1]

    [1] Christopher J. Tralie and Ben Cantil. The concatenator: A bayesian approach to real time
    concatenative musaicing. In Proceedings of the 25th Conference of the International Society for
    Music Information Retrieval (ISMIR 2024) (To Appear). International Society for Music Information
    Retrieval (ISMIR), 2024.

    Parameters
    ----------
    corpus: ndarray(n_corpus_samples)
        Raw audio in the corpus
    ytarget: ndarray(n_target_samples)
        Raw audio samples in the target
    sr: int
        Audio sample rate
    win: int
        Window length to use in processing (hop length is exactly half of this)
    p: int
        Number of activations in the corpus to use for each window
    pd: float
        Probability of jumping to the next window in the corpus
    temperature: float
        Temperature to use in the observations
    n_particles: int
        Number of particles to use
    mel_bands: int
        Number of triangle filters to use in the mel spectrogram
    fmax: int
        Maximum frequency to use in the mel spectrogram (minimum frequency is 0)
    
    Returns
    -------
    ndarray(n_target_samples)
        The result of audio musaicing
    """
    hop = win//2

    ## TODO: Fill this in