# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:16:12 2021

@author: Cristiano
"""

# -*- coding: utf-8 -*-
'''
Created on Sat Nov  7 10:01:05 2020

@author: Cristiano

Cop_NB for CoDuPlo dataset
'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
#from Cop_NB import Cop_NB
#from Cop_NB_byrow import Cop_NB
from sklearn.model_selection import PredefinedSplit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
from sklearn import metrics
import pandas as pd
from sklearn.naive_bayes import GaussianNB
#from ristretto.svd import compute_rsvd
from sklearn import manifold
from utils.duplodata import duplodata, duplodata_reduction
from utils.duplodata_all_tools import duplodata_all, duplodata_exp
from pathlib import Path
from Cop_NB_old import *
from Cop_NB_new import *
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from utils.load_data import iris_data, load_data
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
#-------duplo data----------------------------
#X_train, X_test, y_train, y_test=duplodata()
#X_train, X_test, y_train, y_test=duplodata_reduction()
#n=2

#----------for iris dataset uncomment row below----------
#X_train, X_test, y_train, y_test=iris_data()
#n=0

path = Path('E:/Reunion_Island_INRAE/Reunion_island_all')
X_label=pd.read_csv(path/'X_label_reunion_island.csv')
X_no_label=pd.read_csv(path/'X_no_label_reunion_island.csv')
X_label=X_label.values
gt=X_label[:,-1]
X_no_label=X_no_label.values
X=X_no_label.copy()
y=gt.copy()


#X_train, X_test, y_train, y_test, X_all, y_all, X,y, rows=load_data()
#X=X_all
#y=y_all

#X_train, X_test, y_train, y_test, X_all, y_all, X,y, rows=load_data()
#X_train, X_test, y_train, y_test=duplodata()
#X_train, X_test, y_train, y_test=duplodata_exp()
n=2


#==============================================================================
#--- other idea is to create a bootsrap and repeat the classification----------TO DO
course_of_dimensionality=n

if course_of_dimensionality==1:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X_train)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Make an instance of the Model
    pca = PCA(n_components=8, withen=True)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    #X_train_label=np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    #X_test=ts_test
#=============================================================================
elif course_of_dimensionality==2:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X_train = scaler.transform(X)
    #X_test = scaler.transform(X_test)
    # Make an instance of the Model
    # define transform
    svd = TruncatedSVD(n_components=22)#7 is the best for now
    # prepare transform on dataset
    svd.fit(X)
    # apply transform to dataset
    transformed = svd.transform(X)
    X=transformed.copy()
    #X_test = svd.transform(X_test)
    
    #X_train_label=np.concatenate((transformed,y_train.reshape(-1,1)),axis=1)
    
elif course_of_dimensionality==3:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X)
    X_test = scaler.transform(X)
    # Make an instance of the Model
    rsvd = PCA(n_components=19, whiten=True, svd_solver="randomized", random_state=41).fit(X)
    X = rsvd.transform(X)
    #X_test = rsvd.transform(X_test)
    #X_train_label=np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    #X_test=ts_test
    
    
elif course_of_dimensionality==4:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X_train)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Make an instance of the Model
    ipca = IncrementalPCA(n_components=8,batch_size=20).fit(X_train)
    X_train = ipca.transform(X_train)
    X_test = ipca.transform(X_test)
    #X_train_label=np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    #X_test=ts_test
    

    
print("Start classification with Cop_NB")

#---oversampling try if you want to balance the dataset---

#print('Original dataset shape %s' % Counter(y_train))
#aaa={0:10000, 7:5000, 10:5000, 12: 5000}
#aaa={9:100,15: 1000}
#sm = SMOTE(sampling_strategy=aaa,random_state=42)
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X_train, y_train)
#X_test, y_test=sm.fit_resample(X_test, y_test)
#ros = RandomOverSampler(sampling_strategy=aaa,random_state=42)
#X_train, y_train = ros.fit_resample(X_train, y_train)
#X_test, y_test = ros.fit_resample(X_train, y_train)

#predicted, xclass_prob = Cop_NB(X_train, X_test)

'''
input: cop_type_opt have to be list like if use packagecopulae i.e. ['all'] 
or ['Gaussian', 'Student'], 
have to be string like 'Vine' for Vine and 'Bernstein' for Bernstein
the option estimation_margins can be, emp_be, or KDEpy, or pobs, if pobs
this work with pobs for eval ecdf and KDEpy for eval epdf(!!!)
you can change the option options_kde if you want a different
kernel or different bandwidth
you can change the option options_cop if you want a different
method for fitting copula
'''

#-------------try with COP classifier------------------------

def Kfold_(X, y):
    lst_accu_stratified = []
    lst_accu_stratified_f1 = []
    lst_accu_stratified_K = []
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    #skf = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state=42)
    for train_index, test_index in skf.split(X, y):
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        clf=Cop_NB_new(cop_type_opt='Bernstein', estimation_margins='KDEpy',  options_cop={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':100, 'maxiter': 100,'gtol': 1.0e-3}})
        clf.fit(x_train_fold, y_train_fold)
        classes, predicted, predicted_prob=clf.predict(x_test_fold)
        #clf1=KNeighborsClassifier(n_neighbors = 30)
        #clf1.fit(x_train_fold, y_train_fold)
        #predicted= clf1.predict(x_test_fold)
        
        expected = y_test_fold.copy()
        score_f1=f1_score(expected, predicted, average='weighted')
        score_K=cohen_kappa_score(expected, predicted)
        accuracy=metrics.accuracy_score(expected, predicted)
        
        lst_accu_stratified.append(accuracy)
        lst_accu_stratified_f1.append(score_f1)
        lst_accu_stratified_K.append(score_K)
    print('List of possible accuracy:', lst_accu_stratified)
    print('List of possible f1 accuracy:', lst_accu_stratified_f1)
    print('List of possible Kappa accuracy:', lst_accu_stratified_K)

    return lst_accu_stratified, lst_accu_stratified_f1, lst_accu_stratified_K

#   Provare con STRATIFIEDSHUFFLESPLIT
    sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state=42)
    Train=sss.split(X, y)
#-------------try with other classifier------------------------

lst_accu_stratified, lst_accu_stratified_f1, lst_accu_stratified_K=Kfold_(X, y)

mu=np.mean(lst_accu_stratified)*100
sigma=np.std(lst_accu_stratified)*100

mu_f1=np.mean(lst_accu_stratified_f1)*100
sigma_f1=np.std(lst_accu_stratified_f1)*100

mu_K=np.mean(lst_accu_stratified_K)
sigma_K=np.std(lst_accu_stratified_K)

print(r'Accuracy:  %.2f' %mu + '\u00B1'+'%.2f ' %sigma)
print(r'Accuracy f1:  %.2f' %mu_f1 + '\u00B1'+'%.2f ' %sigma_f1)
print(r'Accuracy K:  %.4f' %mu_K + '\u00B1'+'%.4f ' %sigma_K)


expected = y_test.copy()
accuracy=metrics.accuracy_score(expected, predicted)
accuracy_report=metrics.classification_report(expected, predicted,output_dict=True)
C=confusion_matrix(np.array(expected),np.array(predicted))
accuracy_report_df = pd.DataFrame(accuracy_report).transpose()

'''
accuracy_report_df['Copula']=''
accuracy_report_df['log_like']=''
accuracy_report_df['aic']=''
for k, class_ in enumerate(classes):
    accuracy_report_df['Copula'][k]=class_['elements']['cop_class'].name
    accuracy_report_df['log_like'][k]=np.sum(class_['elements']['loglik'])
    accuracy_report_df['aic'][k]=class_['elements']['aic']
    
'''
MCC = matthews_corrcoef( expected, predicted) 
MCC = (MCC+1.0)/2.0  # così abbiamo i valori tra 0 e 1
accuracy_report_df.to_csv('C:/Users/Cristiano/Desktop/acc_KNN_rep.csv', index= True, float_format='%.3f')
accuracy_report_df.to_csv('./result/accuracy_rep_CoDuplo.csv', index= True, float_format='%.3f')
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % ((sum(scores)*100)/float(len(scores))))
print("Accuracy:",metrics.accuracy_score(expected, predicted))
print("Accuracy_report: \n" ,metrics.classification_report(expected, predicted))
print("Accuracy_MCC: \n %.3f"% (MCC))
# MCC1 = matthews_corrcoef( expected, predicted1)
# MCC1 = (MCC+1.0)/2.0
#print("Accuracy_MCC for other classifier: \n %.3f"%( MCC1) )
#print("Accuracy_report for other classifier: \n", metrics.classification_report(expected, predicted1))
# plt.imshow(y.reshape(145,145), cmap=plt.cm.gray)
#numpy.savetxt("foo.csv", a, delimiter=",")Save matrix