# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:19:43 2020

@author: Cristiano
"""
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from Cop_NB_byrow import Cop_NB

from sklearn.model_selection import PredefinedSplit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB

ts_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/train_0.npy')
ts_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/val_0.npy')
ts_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/test_0.npy')

labels_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/train_y0_30.npy')
labels_val = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/validation_y0_20.npy')
labels_test = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/ground_truth/test_y0_30.npy')

#merged_ts is dataset without labels

merged_ts = np.concatenate((ts_train,ts_val),axis=0)
merged_labels = np.concatenate((labels_train,labels_val),axis=0)
'''check if labels start from 0, otherwise change'''
if np.min(merged_labels)==1:
    merged_labels=merged_labels-1
if np.min(labels_test)==1:
    labels_test=labels_test-1



'''
Strat_Split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state=42)
for train_index, test_index in Strat_Split.split(transformed, merged_labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_mio = transformed[test_index]
    y_mio = merged_labels[test_index]  '''
#scaler = StandardScaler()# Fit on training set only.
#scaler.fit(merged_ts)# Apply transform to both the training set and the test set.
# Apply transform to both the training set and the test set.
#merged_ts = scaler.transform(merged_ts)
#ts_test = scaler.transform(ts_test)
# Make an instance of the Model
# define transform
svd = TruncatedSVD(n_components=5)#7 is the best for now
# prepare transform on dataset
svd.fit(merged_ts)
# apply transform to dataset
transformed = svd.transform(merged_ts)
ts_test = svd.transform(ts_test)
X_train=np.concatenate((transformed,merged_labels.reshape(-1,1)),axis=1)
X_test=ts_test


model=GaussianNB()
model.fit(transformed,merged_labels)
predicted = model.predict(ts_test)
expected = labels_test
accuracy=metrics.accuracy_score(expected, predicted)
accuracy_report=metrics.classification_report(expected, predicted)


print("Accuracy:",metrics.accuracy_score(expected, predicted))
print("Accuracy_report: \n",metrics.classification_report(expected, predicted))




