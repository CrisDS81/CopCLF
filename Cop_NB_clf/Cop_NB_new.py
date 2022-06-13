# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:37:02 2021

@author: Cristiano
"""

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
from copulae import NormalCopula, StudentCopula, GumbelCopula, FrankCopula, ClaytonCopula
from sklearn.metrics import f1_score
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
from utils.cop_family import cop_family_new
from copulae.core import pseudo_obs as pobs
from copulae.copula.estimator.misc import is_archimedean, is_elliptical
from utils.distr_emp_bs import distempcont_bs2
from scipy.interpolate import InterpolatedUnivariateSpline,splprep
#from distfit import distfit



class Cop_NB_new():
    """
    The NAIVE BAYES COPULA classifier
    """

    def __init__(self, cop_type_opt, estimation_margins, options_cop: dict = None, options_kde: dict = None, option_spline: dict=None):
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
        self.cop_type_opt=cop_type_opt
        self.estimation_margins=estimation_margins #KDEpy or spline
        self.options_cop=options_cop #dictionary
        self.options_kde=options_kde
        self.option_spline=option_spline
        
        #optim_options={'method':'L-BFGS-B', 'options':{'maxiter': 50,'gtol': 1.0e-2}}
        
    def fit(self, X, y, class_weight=None):
        
        X=np.concatenate((X,y.reshape(-1,1)), axis=1)
        classes = separate_by_class(self,X)
        elements={}
        
        for class_ in classes:
            
            epdf=np.zeros(class_['class_data'].shape)
            print('start fit empirical marginal')
            '''TO DO
            ricordati di controllare bene bene la parte per l'ecdf!!!!!!!!!!
            sia qui sia nel predict
            '''
            if self.estimation_margins=='emp_qs':
                kde_qs=[]
                ecdf_qs=[]
                kde_obj=[distempcont_bs2(class_['class_data'][:,i],wd=np.ones(len(class_['class_data'])),opt_max_EMP={'emp_distr_linear':False,'emp_distr_allbs':True, 'distr_emp_bs2_bw':np.nan,'distemp_bs2_linear':False})[0] for i in range(class_['class_data'].shape[1])]
                ecdf=[distempcont_bs2(class_['class_data'][:,i],wd=np.ones(len(class_['class_data'])),opt_max_EMP={'emp_distr_linear':True,'emp_distr_allbs':False, 'distr_emp_bs2_bw':np.nan,'distemp_bs2_linear':False})[1] for i in range(class_['class_data'].shape[1])]
            #calcolo la densità di probabilità con la tecnica fftkde
            elif self.estimation_margins=='KDEpy':
                m=np.min(class_['class_data'], axis=0)
                M=np.max(class_['class_data'], axis=0)
                l=np.linspace(m-100, m,100, axis=0)
                L=np.linspace(M, M+100,100, axis=0)
                x=np.concatenate((l,np.sort(class_['class_data'], axis=0),L), axis=0)
                #x=np.linspace(m, M,1000)
              
                #	x=np.sort(class_['class_data'], axis=0)
            
                #f = [interp1d(x[:,i],y[:,i], fill_value=(1.0e-6,1-1.0e-7), bounds_error=False) for i in range(class_['class_data'].shape[1])]
                #ecdf = [ECDF(class_n['class_'][:,i]) for i in range(class_n['class_'].shape[1])]
                #ecdf = [ECDF(x[:,i]) for i in range(class_['class_data'].shape[1])]
                #-------------------------------------------------------------------------
                ecdf_fit = [ECDF(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
                #y=ecdf(x, axis=0)
                y=[ecdf_fit[i](x[:,i]) for i in range(class_['class_data'].shape[1])]
                y=np.array(y).T
                #y=np.round(y,8)
                y=np.where(y<=1e-6, 1e-6, y)#or use 1e-3
                y=np.where(y>=1-1e-6, 1-1e-6, y)#or use 1-1e-3
                #try with pobs of copulae
                #y=pobs(class_['class_data'])
                
                #calcolo l'interpolazione per fittare i dati di test su questa ecdf
                ecdf=[interp1d(x[:,i], y[:,i], fill_value=(1.0e-4,1-1.0e-6), bounds_error=False) for i in range(class_['class_data'].shape[1])]
                #ecdf=[InterpolatedUnivariateSpline(x[:,i], y[:,i], w=None, bbox=[None, None], k=3, ext=0, check_finite=False) for i in range(class_['class_data'].shape[1])]
                #-------------------------------------------------------------------------------
                #tck=[splrep(x[:,i], y[:,i], s=0.01) for i in range(class_['class_data'].shape[1])]
                #ecdf1=[LSQUnivariateSpline(x[:,i], y[:,i], tck[i][0]) for i in range(class_['class_data'].shape[1])]
                
                #ecdf1=[splrep(x[:,i], y[:,i]) for i in range(class_['class_data'].shape[1])]#tuple object
                #s=[UnivariateSpline(x[:,i], x[:,i]) for i in range(x.shape[1])]
                #knt=[s[i].get_knots() for i in range (x.shape[1])]
                #ecdf1=[LSQUnivariateSpline(x[:,i], y[:,i],knt[i]) for i in range(class_['class_data'].shape[1])]
                kde_obj=[KDE_estimation(self.estimation_margins, self.options_kde)for i in range(class_['class_data'].shape[1])]
                [kde_obj[i].fit(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
            elif self.estimation_margins=='pobs': 
                # m=np.min(class_['class_data'], axis=0)
                # M=np.max(class_['class_data'], axis=0)
                # l=np.linspace(m-100, m,100, axis=0)
                # L=np.linspace(M, M+100,100, axis=0)
                # x=np.concatenate((l,np.sort(class_['class_data'], axis=0),L), axis=0)
                x=np.sort(class_['class_data'], axis=0)
                y=pseudo_obs(x)
                ecdf=[interp1d(x[:,i], y[:,i], fill_value=(1.0e-6,1-1.0e-6), bounds_error=False) for i in range(class_['class_data'].shape[1])]
                kde_obj=[KDE_estimation(self.estimation_margins, self.options_kde)for i in range(class_['class_data'].shape[1])]
                [kde_obj[i].fit(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
            elif self.estimation_margins=='distfit': 
                pdf_fit=[]
                dist = distfit()
                ecdf_fit=[dist.fit_transform(class_['class_data'][:,i])for i in range(class_['class_data'].shape[1])]
                ecdf=[ecdf_fit[i]['model'].cdf(class_['class_data'][:,i])for i in range(class_['class_data'].shape[1])]
              
                kde_obj=[KDE_estimation(self.estimation_margins, self.options_kde) for i in range(class_['class_data'].shape[1])]
                # Determine best-fitting probability distribution for data
                [kde_obj[i].fit(class_['class_data'][:,i]) for i in range(class_['class_data'].shape[1])]
                
            
            #for i in range(class_['class_data'].shape[1]):
            #    epdf[:,i]=kde_obj[i].eval_pdf(class_['class_data'][:,i])
            #epdf_mult=np.prod(epdf,axis=1)
            if type(self.cop_type_opt)==list:            
                aic=[]
                cop_type=cop_family_new(self.cop_type_opt)#, dim=class_['class_data'].shape[1])
                for name, cop_class in cop_type:
                    
                    cop_class=cop_class(dim=class_['class_data'].shape[1])
                    print('start fit copula:' , name)
                    if is_elliptical(cop_class):                        
                        fit_copula(cop_class,data=pobs(class_['class_data']), x0=None, method='ml', optim_options= self.options_cop, verbose=1, scale=0.02)
 
                        log_like=(cop_class.log_lik(class_['class_data']))
                        #log_like=np.sum(np.log(cop_class.pdf(pobs(class_['class_data']))*epdf_mult))
                        _aic=2*len(cop_class.params)-2*(log_like)
                    elif is_archimedean(cop_class): #cop_class.fit(class_['class_data'], optim_options= self.options_cop)
                        fit_copula(cop_class,data=pobs(class_['class_data']), x0=1+1e-3, method='ml', optim_options= self.options_cop, verbose=1, scale=0.02)
                        #log_like=np.sum(np.log(cop_class.pdf(pobs(class_['class_data']))*epdf_mult))
                        log_like=(cop_class.log_lik(class_['class_data']))
                        _aic=2-2*(log_like)
                    #log_like=(cop_class.log_lik(class_['class_data']))
                    aic.append({'model':name,
                                'cop_class':cop_class,
                                'aic':_aic,
                                'log_like':log_like,
                                })
                best_aic = min(aic, key=lambda x: x['aic'])
                cop_class=best_aic['cop_class']
                aic=best_aic['aic']
                llike=best_aic['log_like']
            elif type(self.cop_type_opt)!=list and self.cop_type_opt=='Bernstein':
                print('start fit brn copula')
                binNum=ot.BernsteinCopulaFactory_ComputeAMISEBinNumber(pseudo_obs(class_['class_data']))
                #binNum=ot.BernsteinCopulaFactory_ComputeLogLikelihoodBinNumber(pseudo_obs(class_['class_data']))
                #shape=class_['class_data'].shape
                #print(shape)
                #print(binNum)
                cop_class=ot.EmpiricalBernsteinCopula(pseudo_obs(class_['class_data']), binNum, False)
           
            elif type(self.cop_type_opt)!=list and self.cop_type_opt=='Vine':
                #cop_class=pvine.Bicop()
                #cop_class.select(pobs(class_['class_data']))
                #controls = pv.FitControlsVinecop(family_set=[ pv.BicopFamily.gumbel])
                controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gumbel, pv.BicopFamily.gaussian, pv.BicopFamily.frank,pv.BicopFamily.clayton,pv.BicopFamily.student,pv.BicopFamily.joe,pv.BicopFamily.indep])
                cop_class = pv.Vinecop(pseudo_obs(class_['class_data']), controls=controls)          #cop_class=pvine.Vinecop(class_['class_data'].shape[1])
                print('start fit vine copula')
                #family=pvine.all
                #family=family[1:4]
                #cop_class.select(pseudo_obs(class_['class_data']),controls = pv.FitControlsVinecop(family_set=family))# ,select_trunc_lvl=True))
            
            if type(self.cop_type_opt)==list:                           
                elements=({'mean':np.mean(class_['class_data'],axis=0),
                                 'stdev':np.std(class_['class_data'],axis=0),
                                 #'kde_obj':kde_obj,
                                 'kde_epdf_marginal':kde_obj,
                                 'cop_class':cop_class,
                                 'log_like':llike,
                                 'aic':aic,
                                 'ecdf':ecdf,
                                 'len_obs_class':class_['class_data'].shape[0]
                                 })
            else:
                elements=({'mean':np.mean(class_['class_data'],axis=0),
                           'stdev':np.std(class_['class_data'],axis=0),
                           #'kde_obj':kde_obj,
                           'kde_epdf_marginal':kde_obj,
                           'cop_class':cop_class,
                           'ecdf':ecdf,
                           'len_obs_class':class_['class_data'].shape[0]
                           })
            print('elements done for class')
            class_['elements']=elements
            
        self.classes=classes
        return self
    
    # Predict the class for a given row or matrix
    def predict(self, inputData):#log=False ToDo):     
        print('Start Prediction')
        posterior_prob = posterior_class_probabilities(self, inputData)
        xclass_prob=list(posterior_prob.values())
        
        xclass_prob=np.asarray(xclass_prob)
        xclass_prob=xclass_prob.reshape(xclass_prob.shape[0],inputData.shape[0]).T
        normalization=np.sum(xclass_prob, axis=1).reshape(-1,1)
        normalization=np.where(normalization<=1.0e-2, 1.0e-2, normalization)
        #normalization=np.where(normalization<=1.0e-8, 1.0e-8, normalization)
        xclass_prob=xclass_prob/normalization
        #per avere la predizioni sul numero di classi giusto
        #devo utilizzare il dataframe
        aa=pd.DataFrame(xclass_prob)
        aa.columns=self.num_class
        best_label=np.array(aa.idxmax(axis="columns"))
        print('--------------Prediction done-----------------')
        #best_label=np.argmax(xclass_prob, axis=1)     
        
        return [self.classes, best_label, xclass_prob]
    
    def predict_log(self, inputData):     
        print('Start Prediction')
        posterior_prob, posterior_logprob = posterior_class_probabilities(self, self.classes, inputData)
        xclass_logprob=list(posterior_logprob.values())
        xclass_logprob=np.asarray(xclass_logprob)
        xclass_logprob=xclass_logprob.reshape(xclass_logprob.shape[0],inputData.shape[0]).T
        best_label=np.argmax(xclass_logprob, axis=1)
        return best_label, xclass_logprob

                   
def separate_by_class(self,X):
    classes = []
    X_new=X[:,:-1]
    for k in np.unique(X[:,-1].astype(int)):
        classes.append({
            'class_data': X_new[np.where(X[:,-1]==k)],
            #'class_number':k,
            })
    self.num_class=np.unique(X[:,-1].astype(int))
    return classes

def posterior_class_probabilities(self, inputData):
    classes=self.classes
    posterior_prob=dict()
    #posterior_logprob=dict()
    k=0   
    total_obs = sum([class_['elements']['len_obs_class'] for class_ in classes])
    for class_ in classes:
        epdf=np.zeros(inputData.shape)
        pseudo_ecdf=np.zeros(inputData.shape)
        if self.estimation_margins=='KDEpy':
            for i in range(inputData.shape[1]):
                epdf[:,i]=class_['elements']['kde_epdf_marginal'][i].eval_pdf(inputData[:,i])
                pseudo_ecdf[:,i]=class_['elements']['ecdf'][i](inputData[:,i])
                #pseudo_ecdf=np.where(pseudo_ecdf>=1-1.0e-2,1-1.0e-2, pseudo_ecdf)
                #pseudo_ecdf=np.where(pseudo_ecdf<=1.0e-2, 1.0e-2, pseudo_ecdf)
        elif self.estimation_margins=='emp_qs':
            for i in range(inputData.shape[1]):
                epdf[:,i]=class_['elements']['kde_epdf_marginal'][i](inputData[:,i])
                pseudo_ecdf[:,i]=class_['elements']['ecdf'][i](inputData[:,i])
        elif self.estimation_margins=='pobs':
            for i in range(inputData.shape[1]):
                epdf[:,i]=class_['elements']['kde_epdf_marginal'][i].eval_pdf(inputData[:,i])
                pseudo_ecdf[:,i]=class_['elements']['ecdf'][i](inputData[:,i])
                #---- try with sample evaluation of pobs of input Data-----
                #pseudo_ecdf=pobs(inputData)
        elif self.estimation_margins=='distfit':
            for i in range(inputData.shape[1]):
                epdf[:,i]=class_['elements']['kde_epdf_marginal'][i].pdf(inputData[:,i])
                pseudo_ecdf[:,i]=class_['elements']['ecdf'][i](inputData[:,i])
                        #---- try with sample evaluation of pobs of input Data-----
                        #pseudo_ecdf=pobs(inputData)
                
        #pseudo_ecdf[:,i]=pobs((inputData[:,i]))
        #===================================================================
        #==========aggiustamento dei parametri per le marginali e per l'ecdf====
        #=====nota bene il problema è qui!!!=================================
        #pseudo_ecdf=np.where(pseudo_ecdf<=1e-3, 1.0e-7, pseudo_ecdf)
        #pseudo_ecdf=np.where(pseudo_ecdf>=1-1e-3, 1-1.0e-7, pseudo_ecdf)
        #epdf=np.where(epdf<=1.0e-6, 1.0e-6, epdf)
        
        
        epdf_mult=np.prod(epdf,axis=1)
        #epdf_mult=np.where(epdf_mult<=1.0e-4, 1.0e-8, epdf_mult)

        #prior probability
        class_['elements']['prior_class_prob']= class_['elements']['len_obs_class']/(total_obs)
        #loglike probability
        #------------------------------USE WIYH PSEUDO EPDF!!---------------------------------
        if self.cop_type_opt=='Bernstein':
            class_['elements']['loglik']= np.asanyarray(class_['elements']['cop_class'].computePDF(pseudo_ecdf))*epdf_mult.reshape(-1,1)
        else:
            class_['elements']['loglik']= class_['elements']['cop_class'].pdf(pseudo_ecdf)*epdf_mult
        #-------------------------------USE WITHOUT PSEUDO EPDF and with pobs---------------------------------
        #class_['elements']['loglik']= class_['elements']['cop_class'].pdf(pseudo_obs(inputData))*epdf_mult
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