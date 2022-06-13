# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:29:04 2021

@author: Cristiano
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
data = loadmat('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/paviaU.mat')['paviaU']

gt = loadmat('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Hype_img/PaviaU_gt.mat')['paviaU_gt']

print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands: {data.shape[-1]}')
df = pd.DataFrame(data.reshape(data.shape[0]*data.shape[1], -1))

df.columns = [f'band{i}' for i in range(1, df.shape[-1]+1)]

df['class'] = gt.ravel()


def plot_data(data):
  fig = plt.figure(figsize=(12, 10))
  plt.imshow(data, cmap='nipy_spectral')
  plt.colorbar()
  plt.axis('off')
  plt.show()

plot_data(gt)

dff=df.values[:,:-1]
scaler = StandardScaler()# Fit on training set only.
scaler.fit(dff)# Apply transform to both the training set and the test set.
# Apply transform to both the training set and the test set. 
dff = scaler.transform(dff)
svd = TruncatedSVD(n_components=20)#7 is the best for now

svd.fit(dff)
# apply transform to dataset
dff = svd.transform(dff)
dff=np.concatenate((dff,gt.reshape(-1,1)),axis=1)
ae_df = pd.DataFrame(dff)
ae_df.columns = [f'band{i}' for i in range(1, ae_df.shape[-1]+1)]
ae_df=ae_df.rename(columns={'band21': 'class'})
ae_df.head()

res_df = ae_df[ae_df['class'] != 0]
res_df.shape


X = res_df.iloc[:, :-1].values
y = res_df.iloc[:, -1].values

X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.70, stratify = y.ravel())
print(f"X_train Shape: {X_train.shape}\nX_test Shape: {X_test.shape}\ny_train Shape: {y_train.shape}\ny_test Shape:{y_test.shape}")



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, knn_pred)*100}\n")
print(classification_report(y_test, knn_pred))


def predict_class(df, cls):
    pred=[]
    for i in range(df.shape[0]):
        if df.iloc[i, :][-1] == 0:
            pred.append(0)
        else: 
            pred.append(cls.predict(df.iloc[i, :][:-1].values.reshape(1,-1))[0])
    return np.array(pred)

def class_map(X_all,y_all, cls, name=None):
    pred=[]
    for i in range(X_all.shape[0]):
        if y_all[i]== 0:
            pred.append(0)
        else: 
            pred.append(cls.predict(X_all[i,:].reshape(1,-1))[0])
    return np.array(pred)


def class_map_cop(X,y, cls, name=None):
    pred=[]
    for i in range(df.shape[0]):
        if y[i] == 0:
            pred.append(0)
        else: 
            classes, predicted, predicted_prob= cls.predict(X[i,:].reshape(1,-1))
            predicted=predicted[0]
            pred.append(predicted)
            
    return np.array(pred)


pred = predict_class(ae_df, knn)



print(accuracy_score(ae_df['class'].values, pred)*100)
plt.imshow(pred.reshape((512, 217)), cmap='nipy_spectral')
plt.show()

