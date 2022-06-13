# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:51:06 2021

@author: Cristiano
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import PredefinedSplit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from ristretto.svd import compute_rsvd
from sklearn import manifold
import pandas as pd 
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils.duplodata import duplodata
from sklearn.datasets import load_iris
from utils.load_data import load_data, iris_data
from utils.duplodata import duplodata
from Cop_NB_new import Cop_NB_new


X_train, X_test, y_train, y_test=iris_data()
#X_train, X_test, y_train, y_test=duplodata()



# clf=Cop_NB_new(cop_type_opt='Bernstein', estimation_margins='emp_qs',  options_cop={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}})
# clf.fit(clf.fit(X_train, y_train))
# predicted, predicted_prob=clf.predict(X_test)


clf = GaussianProcessClassifier()
clf.fit(X_train, y_train)
predicted=clf.predict(X_test)
predicted_prob=clf.predict_proba(X_test)

expected = y_test.copy()
accuracy=metrics.accuracy_score(expected, predicted)
accuracy_report=metrics.classification_report(expected, predicted,output_dict=True)
C=confusion_matrix(np.array(expected),np.array(predicted))
accuracy_report_df = pd.DataFrame(accuracy_report).transpose()
print("Accuracy:",metrics.accuracy_score(expected, predicted))
print("Accuracy_report: \n",metrics.classification_report(expected, predicted))
