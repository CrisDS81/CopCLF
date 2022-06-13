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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
from sklearn import metrics
import pandas as pd
from sklearn.naive_bayes import GaussianNB
#from ristretto.svd import compute_rsvd
from sklearn import manifold
from utils.duplodata import duplodata, duplodata_reduction
from utils.duplodata_all_tools import duplodata_all, duplodata_obj, duplodata_all_val, duplodata_exp, duplodata_all_red
from pathlib import Path
#from Cop_NB_old import *
from Cop_NB_new import *
from collections import Counter
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.over_sampling import SMOTE
from utils.load_data import iris_data, load_data
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils.other_utils import plot_confusion_matrix, class_map, nor01, nor255, nor1
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from Classifier.clf_other import CNN
import time

#-------duplo data----------------------------
#X_train, X_test, y_train, y_test=duplodata()
#X_train, X_test, y_train, y_test=duplodata_reduction()
#n=2

#----------for iris dataset uncomment row below----------
#X_train, X_test, y_train, y_test=iris_data()
#-----------Reunion Island------------------------------
start = time.time()


path = Path('E:/Reunion_Island_INRAE/Reunion_island_all')

#X_train, X_test, y_train, y_test, X, y=duplodata_all(path)
X_train, X_test, y_train, y_test, X, y=duplodata_obj(path)

#-------------------------------------------------------
#X_train, X_test, y_train, y_test, X, y=duplodata_all_val(path)

#X_train, X_test, y_train, y_test, X, y=duplodata_all(path)
#X_train, X_test, y_train, y_test=duplodata_exp()
#X_train, X_test, y_train, y_test, X_all, y_all, X,y, rows=load_data()
#X_train, X_test, y_train, y_test=iris_data()
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
    startsvd = time.time()
    #--------------------------
    #---------------------------------------
    scaler = StandardScaler()# Fit on training set only.
    #scaler=MinMaxScaler((-1,1))
    scaler.fit(X_train)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #-----------------------------------------------
    # Make an instance of the Model
    # define transform
    n_components=15
    svd = TruncatedSVD(n_components)#7 is the best for now
    # prepare transform on dataset
    svd.fit(X_train)
    # apply transform to dataset
    transformed = svd.transform(X_train)
    
    X_train=transformed.copy()
    X_test = svd.transform(X_test)
    #-------------------------
    Vt=svd.components_
    #Vt=Vt[:,:n_components]
    #X_test=X_test.dot(Vt.T)
    #X=scaler.transform(X)
    #X=svd.transform(X)
    # X_all=scaler.transform(X_all)
    # X_all=svd.transform(X_all)
    
    #X_train_label=np.concatenate((transformed,y_train.reshape(-1,1)),axis=1)
    endsvd = time.time()
    print('time svd:', (endsvd - startsvd))
    
elif course_of_dimensionality==3:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X_train)# Apply transform to both the training set and the test set.
    #----------------------------------------------------------
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #-----------------------------------------------------
    # Make an instance of the Model
    rsvd = PCA(n_components=22, whiten=True, svd_solver="randomized", random_state=41).fit(X_train)
    X_train = rsvd.transform(X_train)
    X_test = rsvd.transform(X_test)
    #X_train_label=np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
    #X_test=ts_test
    
elif course_of_dimensionality==4:
    scaler = StandardScaler()# Fit on training set only.
    scaler.fit(X_train)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Make an instance of the Model
    ipca = IncrementalPCA(n_components=12,batch_size=20).fit(X_train)
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
input: cop_type_opt have to be list like if use package copulae i.e. ['all'] 
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
startlearning = time.time()
cop_type=['all']
#cop_type='Vine'
clf=Cop_NB_new(cop_type_opt=cop_type, estimation_margins='KDEpy',  
               options_cop={'method':'L-BFGS-B','options':{'ftol':1e-04, 
            'eps':1e-05,'maxfun':100, 'maxiter': 100,'gtol': 1.0e-3}})

#clf=Cop_NB(cop_type='Bernstein', method_kde='KDEpy', options_cop=64)

clf.fit(X_train, y_train)
endlearning = time.time()
print('time learning:', (endlearning - startlearning))

startpred = time.time()
classes, predicted, predicted_prob= clf.predict(X_test)
endpred=time.time()
print('timepred:', (endpred - startpred))


end = time.time()
print('total time:', (end - start)/60)
sumtime=(endsvd-startsvd)+(endlearning-startlearning)+(endpred-startpred)
print('total sum time min', sumtime/60 )
print('total sum time sec', sumtime)

#produce classification map of all dataset or for patch of dataset decomment line below
#----------------------------------------------------------

class_map=np.zeros(y_all.shape)
Cmap=scaler.transform(X)
Cmap=svd.transform(Cmap)
_, predict_all, _= clf.predict(Cmap)
predict_all= clf4.predict(Cmap)
class_map[rows]=predict_all
#plt.figure(figsize = (20,10))
plt.imshow(class_map.reshape(512,217), cmap='nipy_spectral')
plt.imshow(class_map.reshape(610,340), cmap='nipy_spectral')
plt.imshow(y_all.reshape(610,340), cmap='nipy_spectral')
plt.imshow(y_all.reshape(512,217), cmap='nipy_spectral')
#----------------------------------------------------------

#-------------try with other classifier------------------------

clf1=KNeighborsClassifier(n_neighbors = 10)
clf1.fit(X_train, y_train.ravel())
predicted1= clf1.predict(X_test)


clf2=GaussianNB()
predicted2=clf2.fit(X_train, y_train).predict(X_test)

clf3= SVC(C = 1.0, kernel = 'rbf', cache_size = 4*1024)
clf3.fit(X_train, y_train)
predicted3=clf3.predict(X_test)

clf4= RandomForestClassifier(max_depth=19, random_state=0)
clf4.fit(X_train, y_train)
predicted4=clf4.predict(X_test)


predict5=CNN(X_train,y_train,X_test,y_test)


expected = y_test.copy()
accuracy=metrics.accuracy_score(expected, predicted)
f1=f1_score(expected,predicted,average='weighted')
accuracy_report=metrics.classification_report(expected, predicted,output_dict=True,digits=4)
conf_mat=confusion_matrix(np.array(expected),np.array(predicted), normalize='true')
conf_mat=np.around(conf_mat,4)
conf_mat=conf_mat*100



'''Normalized confusion Matrix decomment below'''

class_reunion=['Crop cultivations','Sugar cane','Orchards','Forest plantations','Meadow','Forest','Shrubby savannah','Herbaceous savannah','Bare rocks', 'Urbanized areas','Greenhouse crop','Water surfaces','Shadows']
class_pavia_uni=['1. Asphalt', '2. Meadows', '3. Gravel', '4. Trees', 
                 '5. Painted metal sheets','6. Bare Soil', '7. Bitumen', 
                 '8. Self-Blocking Bricks', '9. Shadows']
class_salinas=classes = ['1.Brocoli_green_weeds_1', '2.Brocoli_green_weeds_2',
                      '3.Fallow',
                      '4.Fallow_rough_plow',
                      '5.Fallow_smooth',
                      '6.Stubble',
                      '7.Celery',
                      '8.Grapes_untrained',
                      '9.Soil_vinyard_develop',
                      '10.Corn_senesced_green_weeds',
                      '11.Lettuce_romaine_4wk',
                      '12.Lettuce_romaine_5wk',
                      '13.Lettuce_romaine_6wk',
                      '14.Lettuce_romaine_7wk',
                      '15.Vinyard_untrained',
                      '16.Vinyard_vertical_trellis']


class_label=class_pavia_uni
class_label=class_reunion
accuracies = conf_mat.copy()#normalized confusion matrix
fig, ax = plt.subplots(figsize=(10,8))
cb = ax.imshow(accuracies, cmap='Blues')
plt.xticks(range(len(class_label)), class_label,rotation=90)
plt.yticks(range(len(class_label)), class_label)

for i in range(len(class_label)):
    for j in range(len(class_label)):
        color='blue' if accuracies[i,j] < 50 else 'white'
        ax.text(i , j, '%.2f' % accuracies[i, j], horizontalalignment='center', verticalalignment='center', color=color)

plt.colorbar(cb, ax=ax)
plt.show()

plt.figure(figsize=(10,5))
sns.set(font_scale=0.8)
ax=sns.heatmap(conf_mat, square=True,cmap='viridis', vmin=0, vmax=1, xticklabels=2, yticklabels=2)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

plt.matshow(conf_mat,cmap='rainbow')
# plt.title('Confusion matrix of the classifier')
# plt.colorbar()
# plt.show()


ax = sns.heatmap(conf_mat)

accuracy_report_df = pd.DataFrame(accuracy_report).transpose()

accuracy_report_df.values[:,:3]=accuracy_report_df.values[:,:3]*100
accuracy_report_df.values[:,:3]=accuracy_report_df.values[:,:3].round(2)
if type(cop_type)==list:
    accuracy_report_df['Copula']=''
    accuracy_report_df['log_like']=''
    accuracy_report_df['aic']=''
    for k, class_ in enumerate(classes):
        accuracy_report_df['Copula'][k]=class_['elements']['cop_class'].name
        accuracy_report_df['log_like'][k]=(class_['elements']['log_like']).copy()
        accuracy_report_df['aic'][k]=class_['elements']['aic'].round(1)

MCC = matthews_corrcoef( expected, predicted) 
MCC = (MCC+1.0)/2.0  # così abbiamo i valori tra 0 e 1

K=cohen_kappa_score(expected, predicted)
print(K)
accuracy_report_df.to_csv('./results_table/acc_rep_cop.csv', index= True, float_format='%.2f')
accuracy_report_df.to_csv('./result_reunion_island/acc_rep_cop_pavia.csv', index= True, float_format='%.2f')
#accuracy_report_df.to_csv('./result/accuracy_rep_CoDuplo.csv', index= True, float_format='%.3f')
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % ((sum(scores)*100)/float(len(scores))))
print("Accuracy:",metrics.accuracy_score(expected, predicted))
print("Accuracy_report: \n" ,metrics.classification_report(expected, predicted, digits=4))
print(accuracy_report_df)
print("Accuracy_MCC: \n %.3f"% (MCC))
# MCC1 = matthews_corrcoef( expected, predicted1)
# MCC1 = (MCC+1.0)/2.0
#print("Accuracy_MCC for other classifier: \n %.3f"%( MCC1) )
#print("Accuracy_report for other classifier: \n", metrics.classification_report(expected, predicted1))
# plt.imshow(y.reshape(145,145), cmap=plt.cm.gray)
#numpy.savetxt("foo.csv", a, delimiter=",")Save matrix