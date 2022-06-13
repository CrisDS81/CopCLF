# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 07:29:34 2020

@author: Cristiano

TO DO: finire questo codice mettendo il classificatore come classe poi mettere metodo
fit e poi mettere metodo predict
unire tutto mettere scelt del training e del test set
mettere scelta copula gaussiana o bernstein
se si sceglie bernstei scegliere i bin
mettere scelta per empirical marginal con kernel fft o con kernel skt learn
"""
from csv import reader
from random import seed
from random import randrange
import pandas as pd
import numpy as np
#from sklearn import metrics
from sklearn.metrics import f1_score
from copulae import GaussianCopula, StudentCopula, ClaytonCopula, GumbelCopula, FrankCopula
from copulae.core import pseudo_obs
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from statsmodels.nonparametric.kde import KDEUnivariate
import openturns as ot
from openturns import BernsteinCopulaFactory_ComputeLogLikelihoodBinNumber
from statsmodels.distributions.empirical_distribution import ECDF
from utils.KDE_estimation import KDE_estimation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pyvinecopulib as pv
from scipy.interpolate import *
import pyvinecopulib as pvine
from copulae.copula.estimator import fit_copula
from utils.cop_family import cop_family



class Cop_NB_old():
    """
    The NAIVE BAYES COPULA classifier
    """

    def __init__(self, cop_type, estimation_margins, options_cop: dict = None, options_kde: dict = None, option_spline: dict=None):
        """
        Choose type of copula to use in Naive Bayes

        Parameters
        ----------
        Type: str, optional
            type of the copula, 'Cop_Gaussian', other choose 'Bernstein'
            if 'Cop_Gaussian' put inside the option to pass to optimize.minimize (default SLSQP)
            i.e. optim_options={'method':'BFGS', 'options':{'maxiter': 50,'gtol': 1e-2}}
            if ' Bernstein' make option about 'bin'
        """
        self.cop_type=cop_type
        self.estimation_margins=estimation_margins #KDEpy or spline
        self.options_cop=options_cop  #dictionary
        self.options_kde=options_kde
        self.option_spline=option_spline
        
        #optim_options={'method':'L-BFGS-B', 'options':{'maxiter': 50,'gtol': 1.0e-2}}
        
    def fit(self, X, class_weight=None):
        X=np.array(X)
        classes = separate_by_class(X)
        for class_ in classes:
            elements={}
            if self.cop_type=='Gaussian':
                cop_class=GaussianCopula(class_['class_data'].shape[1])
                print('start fit copula')
                fit_copula(cop_class,class_['class_data'], x0=None, method='ml', optim_options= self.options_cop, verbose=1, scale=0.2)
                #cop_class.fit(class_['class_data'], optim_options= self.options_cop)
            elif self.cop_type=='Bernstein':
                print('start fit brn copula')
                binNum=ot.BernsteinCopulaFactory_ComputeAMISEBinNumber(pseudo_obs(class_['class_data']))
                cop_class=ot.EmpiricalBernsteinCopula(pseudo_obs(class_['class_data']), binNum, False)
            elif self.cop_type=='Student':
                cop_class=StudentCopula(class_['class_data'].shape[1],df=15)
                print('start fit tcopula')
                cop_class.fit(class_['class_data'], optim_options= self.options_cop)
            elif self.cop_type=='Vine':
                cop_class=pvine.Vinecop(class_['class_data'].shape[1])
                print('start fit vine copula')
                cop_class.select(pseudo_obs(class_['class_data']),controls = pv.FitControlsVinecop(trunc_lvl=0))
                
            print('start fit empirical marginal')
            '''TO DO
            ricordati di controllare bene bene la parte per l'ecdf!!!!!!!!!!
            sia qui sia nel predict
            '''
            m=np.min(class_['class_data'], axis=0)
            M=np.max(class_['class_data'], axis=0)
            l=np.linspace(m-100, m,100, axis=0)
            L=np.linspace(M, M+100,100, axis=0)
            x=np.concatenate((l,np.sort(class_['class_data'],axis=0),L), axis=0)
            #x=np.sort(class_['class_data'], axis=0)
            #y=pseudo_obs(x)
        
            #f = [interp1d(x[:,i],y[:,i], fill_value=(1.0e-6,1-1.0e-7), bounds_error=False) for i in range(class_['class_data'].shape[1])]
            #ecdf = [ECDF(class_n['class_'][:,i]) for i in range(class_n['class_'].shape[1])]
            #ecdf = [ECDF(x[:,i]) for i in range(class_['class_data'].shape[1])]
            #-------------------------------------------------------------------------
            ecdf = [ECDF(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
            #y=ecdf(x, axis=0)
            y=[ecdf[i](x[:,i]) for i in range(class_['class_data'].shape[1])]
            y=np.array(y).T
            #y=np.round(y,8)
            # y=np.where(y<=0, 1.0e-5, y)
            # y=np.where(y>=1, 1-1.0e-5, y)
            
            
            ecdf1=[interp1d(x[:,i], y[:,i], fill_value=(1.0e-1,1-1.0e-1), bounds_error=False) for i in range(class_['class_data'].shape[1])]
            #-------------------------------------------------------------------------------
            #tck=[splrep(x[:,i], y[:,i], s=0.01) for i in range(class_['class_data'].shape[1])]
            #ecdf1=[LSQUnivariateSpline(x[:,i], y[:,i], tck[i][0]) for i in range(class_['class_data'].shape[1])]
            
            #ecdf1=[splrep(x[:,i], y[:,i]) for i in range(class_['class_data'].shape[1])]#tuple object
            #s=[UnivariateSpline(x[:,i], x[:,i]) for i in range(x.shape[1])]
            #knt=[s[i].get_knots() for i in range (x.shape[1])]
            #ecdf1=[LSQUnivariateSpline(x[:,i], y[:,i],knt[i]) for i in range(class_['class_data'].shape[1])]
            
            
            
            kde_obj=[KDE_estimation(self.estimation_margins, self.options_kde)for i in range(class_['class_data'].shape[1])]
            [kde_obj[i].fit(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
            elements=({'mean':np.mean(class_['class_data'],axis=0),
                             'stdev':np.std(class_['class_data'],axis=0),
                             #'kde_obj':kde_obj,
                             'kde_epdf_marginal':kde_obj, #or kde if is used instead kde_fft
                             'cop_class':cop_class,
                             'ecdf':ecdf1,
                             #'ecdf':f,
                             
                             'len_obs_class':class_['class_data'].shape[0]
                             })
            print('elements done for class')
            class_['elements']=elements
        self.classes=classes
        return self
    
    # Predict the class for a given row or matrix
    def predict(self, inputData):#log=False ToDo):     
        print('Start Prediction')
        posterior_prob = posterior_class_probabilities(self, self.classes, inputData)
        xclass_prob=list(posterior_prob.values())
        xclass_prob=np.asarray(xclass_prob)
        xclass_prob=xclass_prob.reshape(xclass_prob.shape[0],inputData.shape[0]).T
        best_label=np.argmax(xclass_prob, axis=1)
        return best_label, xclass_prob
    
    def weighted_predict(self, inputData):#log=ToDo):
        
        '''
        ---- write the code belove for find class weight and see nitti last lesson to passive remote sensing----
        from sklearn.utils.class_weight import compute_class_weight
        class_weight_list = compute_class_weight('balanced', np.unique(y_train_labels), y_train_labels)
        class_weight = dict(zip(np.unique(y_train_labels), class_weight_list))
        '''
        print('Start Prediction')
        posterior_prob = posterior_class_probabilities(self, self.classes, inputData)
        xclass_prob=list(posterior_prob.values())
        xclass_prob=np.asarray(xclass_prob)
        xclass_prob=xclass_prob.reshape(xclass_prob.shape[0],inputData.shape[0]).T
        
        best_label=np.argmax(xclass_prob, axis=1)
        return best_label, xclass_prob
    
    def predict_log(self, inputData):     
        print('Start Prediction')
        posterior_prob, posterior_logprob = posterior_class_probabilities(self, self.classes, inputData)
        xclass_logprob=list(posterior_logprob.values())
        xclass_logprob=np.asarray(xclass_logprob)
        xclass_logprob=xclass_logprob.reshape(xclass_logprob.shape[0],inputData.shape[0]).T
        best_label=np.argmax(xclass_logprob, axis=1)
        return best_label, xclass_logprob

                   
def separate_by_class(X):
    classes = []
    X_new=X[:,:-1]
    for k in np.unique(X[:,-1].astype(int)):
        classes.append({
            'class_data': X_new[np.where(X[:,-1]==k)],
            })
    return classes

def posterior_class_probabilities(self, classes, inputData):
    posterior_prob=dict()
    #posterior_logprob=dict()
    k=0   
    total_obs = sum([class_['elements']['len_obs_class'] for class_ in classes])
    for class_ in classes:
        epdf=np.zeros(inputData.shape)
        pseudo_ecdf=np.zeros(inputData.shape)
        for i in range(inputData.shape[1]):
            epdf[:,i]=class_['elements']['kde_epdf_marginal'][i].eval_pdf(inputData[:,i])
            pseudo_ecdf[:,i]=class_['elements']['ecdf'][i](inputData[:,i])
        #===================================================================
        #==========aggiustamento dei parametri per le marginali e per l'ecdf====
        #=====nota bene il problema è qui!!!=================================
        #pseudo_ecdf=np.where(pseudo_ecdf==0, 1.0e-10, pseudo_ecdf)
        #pseudo_ecdf=np.where(pseudo_ecdf==1, 1-1.0e-10, pseudo_ecdf)
        #epdf=np.where(epdf==0, 1.0e-10, epdf)
            
        epdf_mult=np.prod(epdf,axis=1)
        #try with log sum!!!!!!!!!!!!!!!!!!!!!!#################TO DO!
        s=np.log(epdf)
        s1=np.sum(s,axis=1)
        #prior probability
        class_['elements']['prior_class_prob']= class_['elements']['len_obs_class']/(total_obs)
        #loglike probability
        #------------------------------USE WIYH PSEUDO EPDF!!---------------------------------
        if self.cop_type=='Bernstein':
            class_['elements']['loglik']= np.asanyarray(class_['elements']['cop_class'].computePDF(pseudo_ecdf))*epdf_mult.reshape(-1,1)
        else:
            class_['elements']['loglik']= class_['elements']['cop_class'].pdf(pseudo_ecdf)*epdf_mult.reshape(-1,1)

        #-------------------------------USE WITHOUT PSEUDO EPDF---------------------------------
        #Aclass_['elements']['loglik']= class_['elements']['cop_class'].pdf(pseudo_obs(inputData))*epdf_mult
        #posterior probability
        class_['elements']['loglik'].reshape(-1,1)
        class_['elements']['posterior']=class_['elements']['prior_class_prob']*(class_['elements']['loglik'])
        class_['elements']['posterior'].reshape(-1,1)
        '''log prob
        if self.cop_type=='Bernstein':
            class_['elements']['logposterior']=np.asanyarray(class_['elements']['cop_class'].computePDF(pseudo_ecdf))+s1.reshape(-1,1)+np.log(class_['elements']['prior_class_prob'])
        else:
            class_['elements']['logposterior']= np.log(class_['elements']['cop_class'].pdf(pseudo_ecdf))+s1+np.log(class_['elements']['prior_class_prob'])
        posterior_logprob[k]=(class_['elements']['logposterior']).reshape(-1,1)
        '''
        posterior_prob[k]=(class_['elements']['posterior']).reshape(-1,1)
    
        k=k+1
    return posterior_prob #posterior_logprob