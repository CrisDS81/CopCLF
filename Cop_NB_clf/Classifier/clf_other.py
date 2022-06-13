# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:45:22 2022

@author: Cristiano
"""



import matplotlib.pyplot as plt

import numpy as np
#import earthpy.plot as ep
import seaborn as sns
#import earthpy.spatial as es

#import plotly.graph_objects as go
#import plotly.express as px

from scipy.io import loadmat

import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from tqdm import tqdm
from numpy.random import seed
from time import time

seed(11)


def CNN(X_train,y_train,X_test,y_test):
    

    ip_shape = X_train[1].shape

    n_outputs =17

    X_train[1].ravel().shape


    model = Sequential(name = 'Salinas_CNN')

    model.add(Conv1D(filters = 64, kernel_size = 3, activation ='relu', input_shape =(ip_shape[0],1), name = 'Layer1'))
    model.add(Conv1D(filters = 64, kernel_size = 3, activation ='relu' , name = 'Layer2'))
    model.add(Conv1D(filters = 64, kernel_size = 3, activation ='relu' , name = 'Layer3'))

    model.add(MaxPooling1D(pool_size = 2, name = 'MaxPooling_Layer1'))
    model.add(Dropout(0.4, name = 'Dropout1'))

    model.add(Conv1D(filters = 32, kernel_size = 3, activation ='relu', name = 'Layer4'))
    model.add(Conv1D(filters = 32, kernel_size = 3, activation ='relu', name = 'Layer5'))
    model.add(Conv1D(filters = 32, kernel_size = 3, activation ='relu', name = 'Layer6'))

    model.add(MaxPooling1D(pool_size = 2, name = 'MaxPooling_Layer2'))
    model.add(Dropout(0.4, name = 'Dropout2'))
    
    model.add(Flatten(name = 'Flatten'))

    model.add(Dense(25, activation='relu', name = 'DenseLayer'))

    model.add(Dense(n_outputs, activation='softmax', name = 'OutputLayer'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor = 'val_loss',
                            mode = 'min',
                            min_delta = 0,
                            patience = 10,
                            restore_best_weights = True)

    checkpoint = ModelCheckpoint(filepath = 'Salinas_Model.h5', 
                             monitor = 'val_loss', 
                             mode ='min', 
                             save_best_only = True)

    tensorboard = TensorBoard(log_dir='SA_logs/{}'.format(time()))

    hist = model.fit(X_train, 
                       y_train, 
                       epochs = 100, 
                       batch_size = 256 , 
                       validation_data = (X_test, y_test), 
                       callbacks=[early_stop,
                                  checkpoint,
                                  tensorboard])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    pred = np.argmax(model.predict(X_test), axis=1)
    return pred
