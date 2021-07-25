import numpy as np
import scipy as sp
from scipy.io.wavfile import read  
import matplotlib.pyplot as plt
from scipy import signal
import sklearn.metrics.pairwise as metrics

import os.path as ops

def val_paths(fpath, spath):
    
    assert ops.exists(fpath), '{} file does not exist.'.format(fpath)
    assert ops.exists(spath), '{} file does not exist.'.format(spath)

def rm_bg_noise(array, freq):
    b, a = signal.butter(5, 1000/(freq/2), btype = 'highpass')

    filtered_signal = signal.lfilter(b, a, array)
    plt.plot(filtered_signal)  
    plt.title('Highpass Filter')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')
    plt.show()

    c, d = signal.butter(5, 380/(freq/2), btype = 'lowpass')
    filtered_signal = signal.lfilter(c, d, filtered_signal)
    plt.plot(filtered_signal)
    plt.title('Lowpass Filter')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')
    plt.show()

    return filtered_signal

def load_audio(fpath, spath):
    val_paths(fpath, spath)

    (Frequency1, array1) = read(fpath)
    (Frequency2, array2) = read(spath)

    filtered_array2 = rm_bg_noise(array2, Frequency2)

    return array1, filtered_array2

def plot_waveform(array1, array2):

    plt.subplot(1, 2, 1)
    plt.plot(array1)
    plt.title('Audio Clip #1')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(array2)
    plt.title('Audio Clip #2')  
    plt.xlabel('Frequency(Hz)')  
    plt.ylabel('Amplitude')

    plt.show()

def gen_spectrogram(arr):
    spec, _, _, im = plt.specgram(arr, Fs = 1, noverlap=200)

    return spec, im

def plot_spectagram(array1, array2):
    ax1 = plt.subplot(221)
    plt.plot(array1)
    ax2 = plt.subplot(222)
    plt.plot(array2)
    plt.subplot(223, sharex = ax1)
    _, im1 = gen_spectrogram(array1)
    plt.subplot(224, sharex = ax2)
    _, im2 = gen_spectrogram(array2)
    
    plt.show()

def diff(fpath, spath, plot_wave = False, plot_spec = False):
    val_paths(fpath, spath)

    arr1, arr2 = load_audio(fpath, spath)

    spec1, _ = gen_spectrogram(arr1)
    spec2, _ = gen_spectrogram(arr2)

    if len(arr1) > len(arr2):
        arr1 = arr1[:len(arr2)]
    elif len(arr1) < len(arr2):
        arr2 = arr2[:len(arr1)]

    if spec1.shape[1] > spec2.shape[1]:
        spec1 = spec1[:,:spec2.shape[1]]
    elif spec1.shape[1] < spec2.shape[1]:
        spec2 = spec2[:,:spec1.shape[1]] 

    # Similarity should be higher to match
    cos_simi = sp.linalg.norm(metrics.cosine_similarity(spec1, spec2))

    # Difference should be lower to match
    wave_diff = sp.linalg.norm(arr1 - arr2)

    if plot_wave:
        plot_waveform(arr1, arr2)

    if plot_spec:
        plot_spectagram(arr1, arr2)

    return cos_simi, wave_diff


