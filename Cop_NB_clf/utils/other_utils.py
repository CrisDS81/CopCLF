# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:08:33 2021

@author: Cristiano
"""
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import colors
import matplotlib.ticker as mtick

def _best_split(self, X, y):
    '''
    Helper function, calculates the best split for given features and target
    
    :param X: np.array, features
    :param y: np.array or list, target
    :return: dict
    '''
    best_split = {}
    best_info_gain = -1
    n_rows, n_cols = X.shape
    
    # For every dataset feature
    for f_idx in range(n_cols):
        X_curr = X[:, f_idx]
        # For every unique value of that feature
        for threshold in np.unique(X_curr):
            # Construct a dataset and split it to the left and right parts
            # Left part includes records lower or equal to the threshold
            # Right part includes records higher than the threshold
            df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
            df_left = np.array([row for row in df if row[f_idx] <= threshold])
            df_right = np.array([row for row in df if row[f_idx] > threshold])

            # Do the calculation only if there's data in both subsets
            if len(df_left) > 0 and len(df_right) > 0:
                # Obtain the value of the target variable for subsets
                y = df[:, -1]
                y_left = df_left[:, -1]
                y_right = df_right[:, -1]

                # Caclulate the information gain and save the split parameters
                # if the current split if better then the previous best
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_info_gain:
                    best_split = {
                        'feature_index': f_idx,
                        'threshold': threshold,
                        'df_left': df_left,
                        'df_right': df_right,
                        'gain': gain
                    }
                    best_info_gain = gain
    return best_split


def _sample(X, y):
    '''
    Helper function used for boostrap sampling.
    
    :param X: np.array, features
    :param y: np.array, target
    :return: tuple (sample of features, sample of target)
    '''
    n_rows, n_cols = X.shape
    # Sample with replacement
    samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
    return X[samples], y[samples]

def select_n_components(X, goal_var: float) -> int:
    scaler = StandardScaler()# Fit on training set only.
    X=scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=X.shape[1]-1)
    svd.fit(X)
    var_ratio = svd.explained_variance_ratio_
    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    plt.plot(np.cumsum(var_ratio))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    # Return the number of components
    return n_components

def reconstructed_svd(X, n_comp):
    scaler = StandardScaler()# Fit on training set only.
    scaler=MinMaxScaler((-1,1))
    #scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X = scaler.fit_transform(X)
    # define transform
    svd = TruncatedSVD(n_components=n_comp)#7 is the best for now
    # prepare transform on dataset
    X_svd=svd.fit_transform(X)
    Vt=svd.components_
    matrix_svd=X_svd.dot(Vt)
    return X_svd, matrix_svd

def dim_reduction(X, n_components):
    #scaler = StandardScaler()# Fit on training set only.
    scaler=MinMaxScaler((-1,1))
    scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X = scaler.transform(X)
    svd = TruncatedSVD(n_components)#7 is the best for now
    # prepare transform on dataset
    svd.fit(X)
    # apply transform to dataset
    transformed = svd.transform(X)
    X=transformed.copy()
    return X



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true))-0.5)
    plt.ylim(len(np.unique(y_pred))-0.5, -0.5)
    return ax

def class_map(scaler, reduction, X, y, x_all, y_all, rows, clf):
    class_map=np.zeros(y_all.shape)
    X=scaler.transform(X)
    X=reduction.transform(X)
    _, predicted,_=clf.predict(X)
    class_map[rows]=predicted
    plt.imshow(class_map.reshape(610,340),cmap='nipy_spectral')
    return class_map

def  plot_barplot_compare():
    plt.rcParams["figure.figsize"] = (25,11)
    plotdata = pd.DataFrame({
        "RF":[61.67,91.94,70.12,65.63,83.10,85.91,73.23,67.47,73.96,82.98,10.87,92.53,88.40],
        "LSTM":[42.68,88.20,64.20,53.56,76.51,79.51,59.01,60.53,70.86,81.61,18.23,92.16,86.55],
        "ConvLSTM":[49.07,89.86,66.78,67.07,79.37,84.18,64.55,65.05,74.99,86.73,37.74,91.71,89.61],
        "DuPLO":[62.36,92.09,73.24,70.40,82.88,84.59,70.29,63.40,82.02,90.47,40.31,93.26,90.76],
        "RF(DuPLO)":[65.72,92.98,75.39,73.22,85.40,87.30,75.76,67.97,86.32,92.05,43.88,93.87,90.29],
        "CopCLF":[73.99, 93.21, 66.99, 78.93, 90.24, 88.18, 81.63, 80.95,       89.99, 91.57, 56.38, 95.57, 94.87]
        }, 
        index=['Crop cultivations','Sugar cane','Orchards','Forest plantations','Meadow','Forest','Shrubby savannah','Herbaceous savannah','Bare rocks', 'Urbanized areas','Greenhouse crop','Water surfaces','Shadows']
    )
    #plotdata.plot(kind='bar')
    w = 0.8
    plotdata.plot.bar(rot=0, width=w)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=6, fontsize = 'x-large')
    #plt.title("Mince Pie Consumption Study")
    #plt.xlabel("Family Member")
    #plt.ylabel("Pies Consumed")
    
def  plot_barplot_compare_pavia():
    plt.rcParams["figure.figsize"] = (20,11)
    plotdata = pd.DataFrame({
        "Accuracy":[92.49, 92.73, 92.81, 92.83, 93.22],
        "F-Measure":[92.48, 92.74, 92.80, 93.00, 93.20],
        "K":[90.00, 90.38, 90.52, 90.80, 91.00],
           }, 
        index=['20%','30%','40%','50%','60%']
        )
    #plotdata.plot(kind='bar')
    w = 0.7
    plotdata.plot.bar(rot=0, width=w)
    plt.grid(axis='y')
    plt.legend()#loc='upper center', bbox_to_anchor=(0.5, -0.05),
              #fancybox=True, shadow=True, ncol=6, fontsize = 'x-large')
    #plt.title("Mince Pie Consumption Study")
    plt.xlabel("Training set percentage")
    plt.ylabel("Results")
    
def  plot_barplot_compare_pavia_2():
    Accuracy=[92.49, 92.73, 92.81, 92.83, 93.22]
    FMeasure=[92.48, 92.74, 92.80, 93.00, 93.20]
    K=[90.00, 90.38, 90.52, 90.80, 91.00]
    labels=['20%','30%','40%','50%','60%']

    #plotdata.plot(kind='bar')
    width = 0.20
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(20, 5))
    rects1 = ax.bar(x +0, Accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + 0.25, FMeasure, width, label='F-Measure')
    rects3 = ax.bar(x +0.5, K, width, label='K')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores',size = 15)
    ax.set_xlabel('Training set percentage', size=15)
    #ax.set_title('Scores by Accuracy and Percentage of Training-set Pavia', size=20)
    ymin=0
    ymax=100
    ax.set_ylim([ymin, ymax])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_ticks(np.arange(ymin, ymax, 10))
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(labels)
    #ax = plt.gca()
    #plt.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,fontsize = 'x-large')
    #ax.legend(loc=1, fontsize = 'x-large')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize = 'x-large')
    #ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.grid(color='gray', linestyle='--', linewidth=1, axis='y', alpha=0.5)
    #fig.tight_layout()
    plt.show()
    
def  plot_barplot_compare_Salinas_2():
    Accuracy=[91.59, 92.02, 92.41, 92.48, 92.67]
    FMeasure=[92.63, 92.05, 92.43, 92.50, 92.70]
    K=[90.64, 91.12, 91.56, 91.64, 91.85]
    labels=['20%','30%','40%','50%','60%']

    #plotdata.plot(kind='bar')
    width = 0.20
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(20, 5))
    rects1 = ax.bar(x +0, Accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + 0.25, FMeasure, width, label='F-Measure')
    rects3 = ax.bar(x +0.5, K, width, label='K')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores',size = 15)
    ax.set_xlabel('Training set percentage', size=15)
    #ax.set_title('Scores by Accuracy and Percentage of Training-set Pavia', size=20)
    ymin=0
    ymax=100
    ax.set_ylim([ymin, ymax])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_ticks(np.arange(ymin, ymax, 10))
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(labels)
    #ax = plt.gca()
    #plt.legend(bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,fontsize = 'x-large')
    #ax.legend(loc=1, fontsize = 'x-large')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3, fontsize = 'x-large')
    #ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 3, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.grid(color='gray', linestyle='--', linewidth=1, axis='y', alpha=0.5)
    #fig.tight_layout()
    plt.show()


#--------------------------------------------------
def spatialCorrection(A, windowSize=1):
        print("------Spatial Correction new")
        row, col = A.shape
        C= np.copy(A)
        Acount1 =np.zeros(A.shape)
        Acount0 =np.zeros(A.shape) 
        windArea=np.zeros(A.shape) 
        Acount1o =np.zeros(A.shape)
        Acount0o =np.zeros(A.shape) 
        windowArea=np.zeros(A.shape) 
        for i in range(-windowSize, +windowSize+1):
            for j in range(-windowSize, +windowSize+1):
                rowS=max(0,i)
                rowE=min(row,row+i)
                colS=max(0,j)
                colE=min(col,col+j)
                Acount1[row-rowE:row-rowS, col-colE:col-colS] += A[rowS:rowE, colS:colE]                
                windArea[row-rowE:row-rowS, col-colE:col-colS]+=1
                
        Acount0 = windArea-Acount1
        C[ np.logical_or(Acount1 == windArea-1,Acount1 > windArea*0.65)  ] = 1   
        #C[Acount1 > windArea*0.65 ] = 1  
        C[np.logical_or(Acount0 == windArea-1, Acount0 > windArea*0.65) ] = 0            
        #C[Acount0 > windArea*0.65 ] = 0    
        
        return C
    
def nor255(matrix):
    mi=np.abs(matrix.min())
    ma=matrix.max()
    normalized=(matrix+mi)/(ma+mi)
    return (normalized*255).astype(int)

def nor1(matrix):
    ma=abs(matrix).max()
    return matrix/ma

def nor01(matrix):
    mi=(matrix.min())
    ma=(matrix).max()
    if abs(ma-mi) > 0:
      return (matrix-mi)/(ma-mi)
    else:
      if abs(ma) > 0:
         return (matrix-mi)/ma
      else:
         return matrix


label_COLOR_MAP = {1: "#a6e6cc",
      2: "#fbfe2a" ,
       3: "#e68000" ,
        4: "#f2cca6",
     5: "#0fe64d" ,
        6: "#007e00" ,
        7:"#a6ff80" ,
        8: "#1c544d" ,
        9: "#967d5b" ,
       10: "#866672" ,
        11: "#cc0000" ,
       12: "#00ccf2" ,
        13: "#344f21"}
aa=list(label_COLOR_MAP.values())
cmap_my = colors.ListedColormap(aa)


