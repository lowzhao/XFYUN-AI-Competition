from __future__ import division


import sys
import wave
import librosa


import librosa.display
import numpy as np
import matplotlib.pyplot as plt


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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D
from keras.utils import np_utils
from keras.optimizers import SGD

# PCM TO WAV
# we are going to use 40 files for training
dialectArray = ["hebei", "kejia", "minnan", "shanghai", "nanchang", "changsha"]

def pcmToWav(fileName, destination):
    files = os.listdir(fileName)
    for file in files:
        if file == ".DS_Store":
            continue
        print(file)
        pcmfile = open(fileName+"/"+file, 'rb')
        pcmdata = pcmfile.read()
        pcmfile.close()

        # print(destination+".wav")
        wavfile = wave.open(destination+"/"+file+'.wav', 'wb')
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
        if file == ".DS_Store":
            continue
        y, sr = librosa.load(filename+"/"+file, 16000)
        yn = y/(abs(y).max())
        k = 0
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
    # res = np.array([rescale(mfcc[1:]), rescale(mfcc_delta[1:]), rescale(mfcc_delta2[1:])])
    res = np.array([rescale(mfcc[1:])])
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


def load_data(testsplit=0.2, balance=True, size=20):
    X = []
    y = []
    dialectIndex = -1
    for dialect in dialectArray:
        dialectIndex += 1
        files = os.listdir(dialect+"/train/numpified/")
        Xi = []
        
        for fn in files:
            #         directory = DATADIR+FOLDER[key]
            #         files = os.listdir(directory)

            if fn == ".DS_Store":
                continue
            #            if balance:
            #                 files = choice(files, size)

            #         for fn in files:
            x = np.load(dialect+"/train/numpified/"+fn)
            # wantsed to reshape into a square
            # first trying to put importance in the center data
            # calculate the number of data to be deleted at the left right
            # so for (3,41,50) its impossible because the data is rectangle and longer at its verticle side to make square and not losing verticle data we will need to do padding which is quite meaning less,
            # the resulting ndarray should be (150,41)
            # what we want is that
            # the (3,41,13) so that it can become (3*13,41) then become (39,39) after removing the two
#            d = x.shape[2]-3*x.shape[1]
#            d1 = int(round(d/2))
#            d2 = d-d1
#            xs = x.reshape((3*x.shape[1], x.shape[2]))[:, d1:x.shape[2]-d2]
            xs = x.reshape((x.shape[1], x.shape[2]))[:, :]
            Xi.append(xs)
            yi = list(np.ones(len(Xi))*dialectIndex)

        X += Xi
        # the difference between  += and .append() is that += is like extend() which add two array like [a] += [b] // [a,b] but [a].append([b,c]) will give you [a,[b,c]]
        y += yi
        print(dialect+" done!" +"total data points: "+str(len(Xi)))
        
    print("all dialect done!")
    X = np.array(X)
    y = np.array(y).T

    X = np.array(X)
    y = np.array(y).T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsplit)

    X_train, y_train = shaping(X_train, y_train)
    X_test, y_test = shaping(X_test, y_test)
#    print (X_train)
#    print (y_train)
    print (X_test)
    print (y_test)
    return X_train, X_test, y_train, y_test


def shaping(X, y):
    X_new = X.reshape((X.shape[0], 1, X[0].shape[1], X.shape[2]))
    y_new = np_utils.to_categorical(y, 6)
    return X_new, y_new


def single_layer(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Convolution2D(64, (6,6),
                            border_mode='valid', input_shape=(1, 36, 36)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.007, decay=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=1, nb_epoch=15,
              verbose=1, validation_split=0.2)
    y_pred = model.predict_classes(X_test)
    temp = []
    model.save("model.model"  )
    for item in y_test:
        for item2 in range(len(item)):
            if item[item2] == 1:
                temp.append(item2)
    print ("Single 2d-conv net Result")
    print (classification_report(item2, y_pred))
    return model


def double_layer(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 41,41)))
    model.add(Convolution2D(128, (3,3),
                            border_mode='valid', activation='relu'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(6))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.0001, decay=0.00001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=2, nb_epoch=10,
              verbose=1, validation_split=0.2)
    y_pred = model.predict_classes(X_test)
    temp = []
    for item in y_test:
        for item2 in range(len(item)):
            if item[item2] == 1:
                temp.append(item2)

    model.save("model10.model"  )
    print ("Multipayer 2d-conv net Result")
    for item in range(len(y_pred)):
        print(y_pred[item])
        print(y_test[item])
    print (classification_report(temp, y_pred))
    return model


def ConvJS(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 41, 41)))
    model.add(Conv2D(64,(3,3),padding="valid", kernel_initializer="glorot_normal", activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3),kernel_initializer="glorot_normal", activation="relu"))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(1, 1)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3),border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), dim_ordering="th", strides=(1, 1)))
    model.add(Dropout(0.5))
#
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3),border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(6, kernel_initializer='glorot_normal'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=2, epochs=2,
              verbose=1, validation_split=0.2)
    model.save("model.model"  )
    print(X_test,y_test)
    y_pred = model.predict_classes(X_test)
    temp = []
    for item in y_test:
        for item2 in range(len(item)):
            if item[item2] == 1:
                temp.append(item2)
    print("Multipayer 2d-conv net Result")
    for item in y_pred:
        print(item)
    print(y_pred)
    print (classification_report(temp, y_pred))
    return model


filename = ""
destination = ""

#for dialect in dialectArray:
#   for index in xrange(23,26):
#       if(index < 10):
#           destination = dialect+"/train/wave/speaker0"+str(index)
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#
#           pcmToWav(dialect+"/train/speaker0"+str(index), destination)
#
#           destination = dialect+"/train/trimmed/speaker0"+str(index)
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#           trimmer(dialect+"/train/wave/speaker0"+str(index), destination)
#
#           destination = dialect+"/train/numpified/"
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#
#           createdata(dialect+"/train/trimmed/speaker0"+str(index), destination,
#                       dur=1, discard_short=False, fft_win=30, fft_hop=25, n_mfcc=42)
#       else:
#           destination = dialect+"/train/wave/speaker"+str(index)
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#
#           pcmToWav(dialect+"/train/speaker"+str(index),destination)
#
#           destination = dialect+"/train/trimmed/speaker"+str(index)
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#           trimmer(dialect+"/train/wave/speaker"+str(index), destination)
#
#           destination = dialect+"/train/numpified/"
#           if not os.path.exists(destination):
#               os.makedirs(destination)
#
#           createdata(dialect+"/train/trimmed/speaker"+str(index), destination,
#                       dur=1, discard_short=False, fft_win=30, fft_hop=25, n_mfcc=42)


a,b,c,d = load_data()
model = ConvJS(a,b,c,d)
