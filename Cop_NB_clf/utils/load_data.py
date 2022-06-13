# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:50:01 2021

@author: Cristiano
"""

from sklearn.model_selection import train_test_split
import numpy as np
import csv
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from utils.other_utils import nor255, nor01
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA

def load_data():
    #X_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/paviaU.csv',delimiter=',')
    #y_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/paviaU_gt.csv',delimiter=',')
       
    
    
    X_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/salinas_corrected.csv',delimiter=',')
    y_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/salinas_gt.csv',delimiter=',')
   
    
    #X_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/Indian_pines_corrected.csv',delimiter=',')
    #y_all = np.loadtxt('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/Indian_pines_gt.csv',delimiter=',')
    #merged_ts is dataset without labels
    
    rows=np.where(y_all!=0)[0]
    y=y_all[rows]
    X=X_all[rows,:]
    #X=nor01(X)
    #X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, test_size=0.7)
    #X_val, X_test, y_val, y_test  = train_test_split(X_test, y_test,stratify=y_test, test_size=0.5)
    
    #scaler = StandardScaler()# Fit on training set only.
    
    #X=scaler.fit_transform(X)# Apply transform to both the training set and the test set.
    #-----------------------------------------------
    # Make an instance of the Model
    # define transform
    #svd = TruncatedSVD(n_components=18)#7 is the best for now
    # prepare transform on dataset
    # apply transform to dataset
    #transformed = svd.fit_transform(X)
    #X=transformed.copy()
    #X_test = svd.transform(X_test)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.50)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4)
    
    '''check if labels start from 0, otherwise change
    if np.min(y_train)==1:
        y_train=y_train-1
    if np.min(y_test)==1:
        y_test=y_test-1'''
    
    
    return [X_train, X_test, y_train, y_test, X_all, y_all, X, y, rows]

 
    

def iris_data():
    iris = load_iris()

    X = iris['data']
    y = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test