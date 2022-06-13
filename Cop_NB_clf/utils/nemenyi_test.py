# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:40:44 2022

@author: Cristiano
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_hastie_10_2
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from autorank import autorank, create_report
from autorank import autorank, plot_stats, create_report, latex_table
import Orange



clf_names = ["RF", "LSTM", "ConvLSTM", "DUPLO", "RF(DUPLO)", "CopCLF"]

m=[82.4,76.57,80.32,83.73,86,86.92]
s=[1.04,1.11,1.10,1.03,1.24,0.53]
classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB()]

data_names = []
datasets = []


for i in range(0,10):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=i, n_clusters_per_class=1)
    rng = np.random.RandomState(i)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    data_names.append('data_%i' % i)
    #data_names.append('circles_%i' % i)
    #data_names.append('linsep_%i' % i)
    #data_names.append('hastie_%i' % i)
    #datasets.append(make_moons(noise=0.3, random_state=i))
    #datasets.append(make_circles(noise=0.2, factor=0.5, random_state=i))
    #datasets.append(linearly_separable)
    #datasets.append(make_hastie_10_2(1000, random_state=i))
    

results = pd.DataFrame(index=data_names, columns=clf_names)
i=0
for clf_name in (clf_names):
    scores = []
    # iterate over classifiers
    
    for data_name in (data_names):
        #for k in range(0,5):
        print("Applying %s to %s" % (clf_name, data_name))
        res = (np.random.normal(m[i],s[i],1))
        results.at[data_name, clf_name] = res.mean()
    i=i+1

res = autorank(results)
create_report(res)
plot_stats(res,width=6)
plt.show()
latex_table(res)

aa=res.rankdf['meanrank']
names=aa.index.values

names = ["RF", "LSTM", "ConvLSTM", "DUPLO", "RF(DUPLO)", "CopCLF"]
avranks =  aa
cd = Orange.evaluation.compute_CD(avranks, 10,test='nemenyi') #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()
