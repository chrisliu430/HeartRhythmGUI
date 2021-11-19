# -*- coding: utf-8 -*-

import os
import librosa
import glob
import pywt
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import soundfile as sf
#paper thersold
def Eatimation(_med75, _mean, _v):
    if _med75 < _v:
        _thersold = _med75*(1-(_v-_med75))
    elif _med75 > _v and _med75 < _mean :
        _thersold = _med75
    else:
        _thersold = _med75 + (_med75-_mean)
    return _thersold
def thersold(_D):
    __D = copy.deepcopy(_D)
    _D_sort = np.sort(np.abs(__D))
    _D_med75 = np.percentile(_D_sort, 75)
    _D_mean = np.mean(_D_sort)
    _D_var = np.var(_D_sort)
    _D_thersold = Eatimation(_D_med75, _D_mean, _D_var)
    return _D_thersold
def beta_value(_D):
    _D_sort = np.sort(np.abs(_D))
    _D_med75 = np.percentile(_D_sort, 75)
    _D_var = np.var(_D_sort)
    if _D_med75 <=_D_var:
        beta = 1.3
    else:
        beta = 1.4
    return beta
def mid_function(_D, _thersold, _beta):
    new_D = []
    alpha = 1
    T1 = alpha * _thersold
    T2 = _beta * _thersold
    for i in _D:
        _i = np.abs(i)
        if _i>T2:
            new_D.append(i)
        elif T1 <= _i <= T2:
            new_D.append((i**3)/T2**2)
        else:
            new_D.append(0)
    return new_D
def paper_wavelet(data):
    coeffs = pywt.wavedec(data, 'coif5', level=5)
    D5 = copy.deepcopy(coeffs[1])
    D4 = copy.deepcopy(coeffs[2])
    D5_thersold = thersold(D5)
    D4_thersold = thersold(D4)
    # rescaling
    D5_rescaling = np.percentile(np.abs(D5), 50)/0.6745
    D4_rescaling = np.percentile(np.abs(D4), 50)/0.6745
    D5_thersold_rescaling = D5_thersold * D5_rescaling
    D4_thersold_rescaling = D4_thersold * D4_rescaling

    new_coeffs = []
    D5_beta = beta_value(D5)
    D4_beta = beta_value(D4)
    new_D5 = mid_function(D5, D5_thersold_rescaling, D5_beta)
    new_D4 = mid_function(D4, D4_thersold_rescaling, D4_beta)

    new_A5 = np.zeros_like(coeffs[0])
    new_D3 = np.zeros_like(coeffs[3])
    new_D2 = np.zeros_like(coeffs[4])
    new_D1 = np.zeros_like(coeffs[5])

    new_D5 = np.array(new_D5)
    new_D4 = np.array(new_D4)

    new_coeffs.append(new_A5)
    new_coeffs.append(new_D5)
    new_coeffs.append(new_D4)
    new_coeffs.append(new_D3)
    new_coeffs.append(new_D2)
    new_coeffs.append(new_D1)

    datarec = pywt.waverec(new_coeffs, 'coif5')
    return datarec

if __name__ == '__main__':
    data, sr = librosa.load('C:/Users/siou/Desktop/cunyisoundtest/ISU_data/電子聽診個案資料/murmur/murmur/001-5-017H-202008241600.WAV', sr=2000)
    denoise_data = paper_wavelet(data)
    down_sample_audio_data = librosa.resample(denoise_data.T, orig_sr=2000, target_sr=1000).T
    plt.subplot(311)
    plt.title('original signal')
    plt.plot(data)
    plt.subplot(313)
    plt.plot(down_sample_audio_data)
    plt.title('denoise signal')
    plt.show()

