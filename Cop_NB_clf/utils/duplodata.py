# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:15:21 2021

@author: Cristiano
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def duplodata():
    ts_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/train_0.npy')
    ts_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/val_0.npy')
    ts_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/test_0.npy')
    
    labels_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/train_y0_30.npy')
    labels_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/validation_y0_20.npy')
    labels_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/test_y0_30.npy')
    
    #merged_ts is dataset without labels
    
    merged_ts = np.concatenate((ts_train,ts_val),axis=0)
    X_train=merged_ts.copy()
    merged_labels = np.concatenate((labels_train,labels_val),axis=0)
    
    '''check if labels start from 0, otherwise change'''
    #if np.min(merged_labels)==1:
    #    merged_labels=merged_labels-1
    #if np.min(labels_test)==1:
    #    labels_test=labels_test-1
    
    y_train=merged_labels.copy()
    y_test=labels_test.copy()
    X_test=ts_test.copy()
    
    return X_train, X_test, y_train, y_test


def duplodata_reduction():
    ts_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/train_0.npy')
    ts_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/val_0.npy')
    ts_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/test_0.npy')
    
    labels_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/train_y0_30.npy')
    labels_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/validation_y0_20.npy')
    labels_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/test_y0_30.npy')
    
    #merged_ts is dataset without labels
    
    X_train = np.concatenate((ts_train,ts_val),axis=0)
    merged_labels = np.concatenate((labels_train,labels_val),axis=0)
    
    '''check if labels start from 0, otherwise change'''
    if np.min(merged_labels)==1:
        merged_labels=merged_labels-1
    if np.min(labels_test)==1:
        labels_test=labels_test-1
    
    y_train=merged_labels.copy()
    y_test=labels_test.copy()
    X_test=ts_test.copy()
        
    X=X_train.copy()
    y=y_train.copy()
    X_train, X_te, y_train, y_te = train_test_split(X, y, stratify=y, test_size=0.8)
    
    
    return X_train, X_test, y_train, y_test

def duplodata_reduction_1():#with stratified shuffle split (to complete!!)
    ts_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/train_0.npy')
    ts_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/val_0.npy')
    ts_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/test_0.npy')
    
    labels_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/train_y0_30.npy')
    labels_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/validation_y0_20.npy')
    labels_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/test_y0_30.npy')
    
    #merged_ts is dataset without labels
    
    X_train = np.concatenate((ts_train,ts_val),axis=0)
    merged_labels = np.concatenate((labels_train,labels_val),axis=0)
    
    '''check if labels start from 0, otherwise change'''
    if np.min(merged_labels)==1:
        merged_labels=merged_labels-1
    if np.min(labels_test)==1:
        labels_test=labels_test-1
    
    y_train=merged_labels.copy()
    y_test=labels_test.copy()
    X_test=ts_test.copy()
        
    X=X_train.copy()
    y=y_train.copy()
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state=42)
    Train, Test=sss.get_n_splits(X, y)
    
    return X_train, X_test, y_train, y_test