# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:31:39 2022

@author: Cristiano
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from osgeo import gdal
from pathlib import Path
import pandas as pd
from utils.other_utils import select_n_components, dim_reduction
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
#from fast_ml.model_development import train_valid_test_split
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def timeseries_clf_data():
    filepath_ts_train_sin = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/sin1_data/'
    filepath_ts_test_sin = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/sin1_data/'
    filepath_ts_train_dyadic = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/dyadic_data/'
    filepath_ts_test_dyadic = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/dyadic_data/'
    filepath_ts_train_arnold= r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/arnoldtongue_data/'
    filepath_ts_test_arnold = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/arnoldtongue_data/'
    filepath_ts_train_log = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/logmap_data/'
    filepath_ts_test_log = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/logmap_data/'
    filepath_ts_train_gaunoise = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/gwn_data/'
    filepath_ts_test_gaunoise = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/gwn_data/'
    filepath_ts_train_ross = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/rossler_data/'
    filepath_ts_test_ross = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/rossler_data/'



    onlyfiles = [f for f in listdir(filepath_ts_train_sin) if isfile(join(filepath_ts_train_sin, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_sin) if isfile(join(filepath_ts_test_sin, f))]
    sin_train=np.zeros((2000,2501))
    sin_test=np.zeros((2000,2501))
    label_sin=np.ones(2000)
    label_dyadic=np.ones(2000)
    label_arnold=np.ones(2000)
    label_log=np.ones(2000)
    label_gaunoise=np.ones(2000)
    label_ross=np.ones(2000)
    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_sin+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        sin_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_sin+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        sin_test[j,:2500]=ts_test
    
    onlyfiles = [f for f in listdir(filepath_ts_train_dyadic) if isfile(join(filepath_ts_train_dyadic, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_dyadic) if isfile(join(filepath_ts_test_dyadic, f))]
    dyadic_train=np.zeros((2000,2501))
    dyadic_test=np.zeros((2000,2501))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_dyadic+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        dyadic_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_dyadic+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        dyadic_test[j,:2500]=ts_test
        
    onlyfiles = [f for f in listdir(filepath_ts_train_arnold) if isfile(join(filepath_ts_train_arnold, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_arnold) if isfile(join(filepath_ts_test_arnold, f))]
    arnold_train=np.zeros((2000,2501))
    arnold_test=np.zeros((2000,2501))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_arnold+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        arnold_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_arnold+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        arnold_test[j,:2500]=ts_test

    
    
    onlyfiles = [f for f in listdir(filepath_ts_train_log) if isfile(join(filepath_ts_train_log, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_log) if isfile(join(filepath_ts_test_log, f))]
    log_train=np.zeros((2000,2501))
    log_test=np.zeros((2000,2501))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_log+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        log_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_log+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        log_test[j,:2500]=ts_test
    
    onlyfiles = [f for f in listdir(filepath_ts_train_gaunoise) if isfile(join(filepath_ts_train_gaunoise, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_gaunoise) if isfile(join(filepath_ts_test_gaunoise, f))]
    gaunoise_train=np.zeros((2000,2501))
    gaunoise_test=np.zeros((2000,2501))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_gaunoise+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        gaunoise_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_gaunoise+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        gaunoise_test[j,:2500]=ts_test
        
    onlyfiles = [f for f in listdir(filepath_ts_train_ross) if isfile(join(filepath_ts_train_ross, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_ross) if isfile(join(filepath_ts_test_ross, f))]
    ross_train=np.zeros((2000,2501))
    ross_test=np.zeros((2000,2501))
    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_ross+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_train=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_train=ts_train.reshape(n*m)
        ross_train[i,:2500]=ts_train
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_ross+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        #plt.imshow(fullarray, cmap='gray')
        ts_test=np.copy(fullarray)
        n=raster.RasterXSize
        m=raster.RasterYSize
        ts_test=ts_test.reshape(n*m)
        ross_test[j,:2500]=ts_test

    sin_train[:,2500]=label_sin*0
    sin_test[:,2500]=label_sin*0
    dyadic_train[:,2500]=label_dyadic*1
    dyadic_test[:,2500]=label_dyadic*1
    arnold_train[:,2500]=label_arnold*2
    arnold_test[:,2500]=label_arnold*2
    log_train[:,2500]=label_log*3
    log_test[:,2500]=label_log*3
    gaunoise_train[:,2500]=label_gaunoise*4
    gaunoise_test[:,2500]=label_gaunoise*4
    ross_train[:,2500]=label_ross*5
    ross_test[:,2500]=label_ross*5
    
    X=np.concatenate((sin_train, dyadic_train, arnold_train, log_train, gaunoise_train,ross_train), axis=0)
    y=np.concatenate((sin_test, dyadic_test, arnold_test, log_test, gaunoise_test,ross_test), axis=0)
    #X_train=X[:,:2500]
    #X_test=y[:,:2500]
    #y_train=X[:,2500]
    #y_test=y[:,2500]
    np.savetxt('C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/X_train.csv',X, delimiter=",")
    np.savetxt('C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/X_test.csv',y, delimiter=",")

def load_ts_orlando():
    X = np.loadtxt('C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/X_train.csv', delimiter=",")
    y = np.loadtxt('C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/X_test.csv', delimiter=",")
    
    X_train=X[:,:2500]
    X_test=y[:,:2500]
    y_train=X[:,2500]
    y_test=y[:,2500]
    return [X_train, X_test, y_train, y_test, X, y]

def load_ts_orlando_NN():
    filepath_ts_train_sin = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/sin1_data/'
    filepath_ts_test_sin = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/sin1_data/'
    filepath_ts_train_dyadic = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/dyadic_data/'
    filepath_ts_test_dyadic = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/dyadic_data/'
    filepath_ts_train_arnold= r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/arnoldtongue_data/'
    filepath_ts_test_arnold = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/arnoldtongue_data/'
    filepath_ts_train_log = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/logmap_data/'
    filepath_ts_test_log = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/logmap_data/'
    filepath_ts_train_gaunoise = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/gwn_data/'
    filepath_ts_test_gaunoise = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/gwn_data/'
    filepath_ts_train_ross = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/train_data/rossler_data/'
    filepath_ts_test_ross = r'C:/Users/Cristiano/Desktop/Orlando_paper_tseries/Transfer/test_data/rossler_data/'



    onlyfiles = [f for f in listdir(filepath_ts_train_sin) if isfile(join(filepath_ts_train_sin, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_sin) if isfile(join(filepath_ts_test_sin, f))]
    sin_train=np.zeros((len(onlyfiles),50,50))
    sin_test=np.zeros((len(onlyfiles),50,50))
    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_sin+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())

        sin_train[i]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_sin+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        sin_test[j]=fullarray
    
    onlyfiles = [f for f in listdir(filepath_ts_train_dyadic) if isfile(join(filepath_ts_train_dyadic, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_dyadic) if isfile(join(filepath_ts_test_dyadic, f))]
    dyadic_train=np.zeros((len(onlyfiles),50,50))
    dyadic_test=np.zeros((len(onlyfiles),50,50))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_dyadic+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        dyadic_train[i]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_dyadic+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        dyadic_test[j]=fullarray
        
    onlyfiles = [f for f in listdir(filepath_ts_train_arnold) if isfile(join(filepath_ts_train_arnold, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_arnold) if isfile(join(filepath_ts_test_arnold, f))]
    arnold_train=np.zeros((len(onlyfiles),50,50))
    arnold_test=np.zeros((len(onlyfiles),50,50))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_arnold+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        arnold_train[i]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_arnold+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        arnold_test[j]=fullarray

    
    
    onlyfiles = [f for f in listdir(filepath_ts_train_log) if isfile(join(filepath_ts_train_log, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_log) if isfile(join(filepath_ts_test_log, f))]
    log_train=np.zeros((len(onlyfiles),50,50))
    log_test=np.zeros((len(onlyfiles),50,50))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_log+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        log_train[i]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_log+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())

        log_test[j]=fullarray
    
    onlyfiles = [f for f in listdir(filepath_ts_train_gaunoise) if isfile(join(filepath_ts_train_gaunoise, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_gaunoise) if isfile(join(filepath_ts_test_gaunoise, f))]
    gaunoise_train=np.zeros((len(onlyfiles),50,50))
    gaunoise_test=np.zeros((len(onlyfiles),50,50))

    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_gaunoise+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())

        gaunoise_train[i]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_gaunoise+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())

        gaunoise_test[j]=fullarray
        
    onlyfiles = [f for f in listdir(filepath_ts_train_ross) if isfile(join(filepath_ts_train_ross, f))]
    onlyfiles_test = [f for f in listdir(filepath_ts_test_ross) if isfile(join(filepath_ts_test_ross, f))]
    ross_train=np.zeros((len(onlyfiles),50,50))
    ross_test=np.zeros((len(onlyfiles),50,50))
    
    for i in np.arange(len(onlyfiles)):
        filepath=filepath_ts_train_ross+onlyfiles[i]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())

        ross_train[i,:2500]=fullarray
        #----------------------------------------
    for j in np.arange(len(onlyfiles_test)):
        filepath=filepath_ts_test_ross+onlyfiles_test[j]
        raster = gdal.Open(filepath)
        raster.GetProjection()
        fullarray = np.array(raster.ReadAsArray())
        ross_test[j,:2500]=fullarray

    sin_train_lab=np.ones((1999,1))*0
    sin_test_lab=np.ones((1999,1))*0
    dyadic_train_lab=np.ones((1999,1))*1
    dyadic_test_lab=np.ones((1999,1))*1
    arnold_train_lab=np.ones((1999,1))*2
    arnold_test_lab=np.ones((1999,1))*2
    log_train_lab=np.ones((1999,1))*3
    log_test_lab=np.ones((1999,1))*3
    gaunoise_train_lab=np.ones((1999,1))*4
    gaunoise_test_lab=np.ones((1999,1))*4
    ross_train_lab=np.ones((1999,1))*5
    ross_test_lab=np.ones((1999,1))*5
    
    X_train=np.concatenate((sin_train, dyadic_train, arnold_train, log_train, gaunoise_train,ross_train), axis=0)
    X_test=np.concatenate((sin_test, dyadic_test, arnold_test, log_test, gaunoise_test,ross_test), axis=0)
    X_train_lab=np.concatenate((sin_train_lab,dyadic_train_lab,arnold_train_lab,log_train_lab,gaunoise_train_lab,ross_train_lab),axis=0)
    X_test_lab=np.concatenate((sin_test_lab,dyadic_test_lab,arnold_test_lab,log_test_lab,gaunoise_test_lab,ross_test_lab),axis=0)

    return [X_train, X_test, X_train_lab, X_test_lab]



    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_label_red_reunion_island.csv",X_label_red, delimiter=",")
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_label_red_reunion_island.csv",X_no_label_red, delimiter=",")
    
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_label_reunion_island.csv",X_label, delimiter=",")
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_label_reunion_island.csv",X_no_label, delimiter=",")
    
    #np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_ndvi_label_reunion_island.csv",X_no_ndvi_label, delimiter=",")
