# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:04:00 2020

@author: Cristiano
"""


from csv import reader
from random import seed
from random import randrange
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from copulae import GaussianCopula 
from copulae.core import pseudo_obs
from copulas.multivariate import GaussianMultivariate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



#dataset=input("Select path of dataset:")
#filename=dataset
#dataset = pd.read_csv(filename,header=0)
ts_train = np.load('C:/Users/Cristiano/Desktop/Python_Code/CoDuPlo/Dataset/Duplo_Data/reunion_learned_features/train_0.npy')
dataset=np.array(ts_train)
#dataset=dataset[:,:3]
def fit_copula(dataset):
    dataset=np.array(dataset)
    #dataset=dataset[:,:-1]
    gc_copulae=GaussianCopula(dim=dataset.shape[1])
    gc_copulas=GaussianMultivariate()
    fit_copulae=gc_copulae.fit(dataset,est_var=True)
    fit_copulas=gc_copulas.fit(dataset)
    gc_copulae_pdf=gc_copulae.pdf(pseudo_obs(dataset))
    gc_copulae_cdf=gc_copulae.cdf(pseudo_obs(dataset))
    gc_copulas_pdf=gc_copulas.pdf(pseudo_obs(dataset))
    return gc_copulae_pdf, gc_copulas_pdf

gc_copulae_pdf, gc_copulas_pdf=fit_copula(dataset)


# d = 3 # Number of dimensions
# mean = np.array([-2., 1.,1.8])
# cov = np.matrix([
#     [1, 0.8, 0.6], 
#     [0.8, 1, 0.3],
#     [0.6, 0.3,1]
# ])
# X = np.random.multivariate_normal(mean,cov, (100))
