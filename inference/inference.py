#
from __future__ import division


import sys
import wave
import librosa


import librosa.display
import numpy as np


import pandas as pd
import math
import os

import gc


import numpy as np
from numpy.random import choice
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import os
from collections import Counter
from keras.regularizers import l1_l2
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD

# PCM TO WAV
# we are going to use 40 files for training
dialectArray = ["hebei", "kejia", "minnan", "shanghai", "nanchang", "changsha"]



def pcmToWav(fileName, destination):
    files = os.listdir(fileName) # ../dataset
    for file in files:
        if file == ".DS_Store":
            continue
        print(file)
        pcmfile = open(fileName+"/"+file, 'rb')
        pcmdata = pcmfile.read()
        pcmfile.close()

        # print(destination+".wav")
        wavfile = wave.open(destination+"/"+file, 'wb')
        wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wavfile.writeframes(pcmdata)
        wavfile.close()
    del files


# Trim
def trimmer(filename, destination):
    files = os.listdir(filename)
    for file in files:

        if file == ".DS_Store":
            continue
        y, sr = librosa.load(filename+"/"+file, sr=16000)
        y_trimmed, index = librosa.effects.trim(y, top_db=12, frame_length=2)
        # print(librosa.get_duration(y), librosa.get_duration(y_trimmed))

        # destination = trimmed_destination + sample_file[:-4] + '.wav'
        librosa.output.write_wav(destination+"/"+file, y_trimmed, sr)
    del y, files


def createdata(filename, destination, **kwargs):
    duration = kwargs['dur']
    fft_win = kwargs['fft_win']
    fft_hop = kwargs['fft_hop']
    n_mfcc = kwargs['n_mfcc']

    files = os.listdir(filename)
    for file in files:
        y, sr = librosa.load(filename+"/"+file, 16000)
        yn = y/(abs(y).max())
        # k = 0
        n_fft = int(round(fft_win/1000*sr))
        hop_length = int(round(fft_hop/1000*sr))

        # print(y.shape[0])
        i = 0
        j = y.shape[0]
        if j-i <= round(duration*sr):
            yseg = windowcut(yn, i, j, sr=sr, dur=duration,
                                discard_short=kwargs['discard_short'])
            savemfcc(savetarget=destination+"/"+file+'_0', y=yseg, sr=sr,
                     n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        else:
            m = 0

            def cut(x): return (2*x+1)*round(duration*sr/2)
            slices = filter(None, map(lambda n: (int(i+cut(n)-cut(0)), int(i+cut(n)+cut(0)))
                                        if i+cut(n) < j else None, range(int(math.ceil(duration*sr)))))
            for tu in slices:
                ii = int(tu[0])
                jj = int(tu[1])
                yseg = windowcut(yn, ii, jj, sr=sr, dur=duration,
                                    discard_short=kwargs['discard_short'])
                savemfcc(savetarget=destination+"/"+file+'_'+str(m), y=yseg, sr=sr,
                         n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                m += 1
                break

    #     savemfcc(savetarget = "dsadjbajksd", y=yn, sr=sr, n_mfcc=n_mfcc, n_fft = n_fft)
        print('Done')
    del files,y,yn,yseg


def savemfcc(savetarget, y, sr, n_mfcc,  **kwargs):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, **kwargs)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    # ALthough a test showed not much difference, by eye, it seems rescaling each is better.
    # rescale each matrix

    # print(mfcc[0])
    # print(mfcc[1])
#     print(mfcc[2])

#     print(mfcc[1:].min())

#     duration = 1
#     wincut(duration ,mfcc)
    # print(mfcc.shape)
    # print(mfcc_delta2.shape)

    # rescale from 0 to 1
    res = np.array([rescale(mfcc[1:]), rescale(
        mfcc_delta[1:]), rescale(mfcc_delta2[1:])])
    # print(res.shape)
    # rescale all at once (deltas will be squeezed since mfcc has larger scales)
    # res = rescale(np.array([mfcc[1:],mfcc_delta[1:],mfcc_delta2[1:]]))

    # Found out that the window cut function cannot be removed.
    # brining back the window cut function

#    plt.figure(figsize=(12, 10))
#    librosa.display.specshow(mfcc, x_axis='time')
##     librosa.display.specshow(res[0],x_axis='time')
#    plt.colorbar()
#    plt.title('MFCC')
#    plt.tight_layout()
    np.save(savetarget, res)


def windowcut(y, i, j, dur=1, sr=16000, discard_short=True):
    """ Returns a slice of y with a specified width
    dur: width of window in second
    """
    if dur < len(y)/sr:
        left = round((i+j)/2)-round(dur*sr/2)
        right = left+round(dur*sr)
        if left < 0:
            left = 0
            right = round(dur*sr)
        elif right > len(y):
            right = len(y)
            left = right - round(dur*sr)
        return y[int(left):int(right)]
    else:  # discard data if total length is smaller than duration we want
        if discard_short:
            return None
        else:  # padd with zeros at the end
			return np.array(np.append(y, np.zeros(dur*sr-len(y))))


def rescale(m):
    # rescale by global max of absolute values
    offset = m.min()
    scale = m.max()-m.min()
    return (m-offset)/scale


def load_data():

    files = os.listdir("dataset/numpified")
    model = load_model("model.model")
    result = open("../result/result.txt","w")

    for fn in files:
        x = np.load("dataset/numpified/"+fn)
        d = x.shape[2]-3*x.shape[1]
        d1 = int(round(d/2))
        d2 = d-d1
        xs = x.reshape((3*x.shape[1], x.shape[2]))[:, d1:x.shape[2]-d2]
        xs = np.array([[xs]])
        # print(xs.shape)
        xs = shaping(xs)
        res = model.predict(xs)
        tempRes = 0;
        for i in range(0,len(res[0])):
            if res[0][i] > res[0][tempRes]:
                tempRes = i
        result.write("%-40s %s\n\n"%(fn[:-6],dialectArray[int(tempRes)]))

    result.close()

    


def shaping(X):
    X_new = X.reshape((X.shape[0], 1, X[0].shape[1], X.shape[2]))
    return X_new


pcmToWav("../dataset","dataset/wav")
trimmer("dataset/wav","dataset/trimmed")
createdata("dataset/trimmed","dataset/numpified",dur=1, discard_short=False, fft_win=30, fft_hop=25, n_mfcc=13)
load_data()

