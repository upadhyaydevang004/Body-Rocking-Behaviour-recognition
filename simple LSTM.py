# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 01:14:23 2019

@author: Shalin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:30:20 2019

@author: Shalin
"""
import os
import pandas as pd
import numpy as np
from itertools import chain
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D#, TimeDistributed
#from keras.utils import np_utils
#import keras
#import tensorflow as tf
#import keras.backend as K
#import argparse
#import matplotlib.pyplot as plt


def load_det(y):
    return np.loadtxt(y)

if __name__ =='__main__':
    
    label_path = 'Training Data B/'
    predict_file_name = 'prediction.txt'
    
    X_train = pd.read_csv('Train_Features/train_features.csv', header=None).values.tolist()
    y_train = pd.read_csv('Train_Features/train_labels.csv', header=None).values.tolist()
    X_test = pd.read_csv('Test_Features/Session02/features.csv', header=None).values.tolist()
    y_test = pd.read_csv('Training Data B/Session02/Detection.txt',delim_whitespace= True, skipinitialspace= True, header=None).values.tolist()
    
# =============================================================================
#     #feature scaling    - normalize columnwise 
#     from sklearn import preprocessing
#     x = df.values #returns a numpy array
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     df = pd.DataFrame(x_scaled)
# =============================================================================
 
    model = Sequential()
    # define LSTM model
    model.add(LSTM(512,input_shape=(84,1)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='softmax'))
    model.add(Dense(64,activation='softmax'))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print(model.summary())
    history = model.fit(X_train,
                        y_train, 
                        batch_size=512, 
                        epochs=10,
                        validation_data=(X_test,y_test),
                        verbose=1)