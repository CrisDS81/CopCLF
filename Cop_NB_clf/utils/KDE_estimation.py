# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:57:17 2020

@author: Cristiano
"""
import numpy as np
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from copulae.utility.dict import merge_dict
from scipy.interpolate import *
from copulae.core import pseudo_obs as pobs
from scipy.optimize import OptimizeResult, minimize
#from distfit import distfit



class KDE_estimation():
    """
    The class for marginal empirical density estimation using different tecniques
    """
    def __init__(self, method_kde, option_kde):
            

        self.method_kde=method_kde
        self.options_kde = option_kde
        
    def fit(self, marginal):
        options_kde=form_options(marginal, self.method_kde, self.options_kde or {})
        if self.method_kde=='sklearn_kde_GS':
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': 10**np.linspace(-1, 1, 100)}, cv=20)# 20-fold cross-validation
            grid.fit(marginal[:,None])
            kde = grid.best_estimator_
        elif self.method_kde=='sklearn_kde':
            kde = KernelDensity(**options_kde)
            kde.fit(marginal[:, None])
        elif self.method_kde=='scipy_kde':
            kde = gaussian_kde(marginal, **options_kde)
        elif self.method_kde=='statsmodel_kde':
            kde= KDEUnivariate(marginal)
            kde.fit(**options_kde)
        elif self.method_kde=='KDEpy':
            
            #weights = marginal **2
            #x_w=res_ECM1 = minimize(FFTKDE(**options_kde).fit(marginal,weights=x_w),x0=np.ones(len(marginal)) )
            
            kde_=FFTKDE(**options_kde).fit(marginal,weights=None)
            x,y=kde_.evaluate()#with bw 0.2 fixed seams to work better
            kde=interp1d(x, y, fill_value=1.0e-8,bounds_error=False)
            #kde=interp1d(x, y,bounds_error=False)
            #---------------------------------------------------------------
            #kde=UnivariateSpline(x,y, ext=3)
        #-----------option pobs with fftkde-------------------
        elif self.method_kde=='pobs':
            
            #weights = marginal **2
            #x_w=res_ECM1 = minimize(FFTKDE(**options_kde).fit(marginal,weights=x_w),x0=np.ones(len(marginal)) )
            
            kde_=FFTKDE(**options_kde).fit(marginal,weights=None)
            x,y=kde_.evaluate()#with bw 0.2 fixed seams to work better
            kde=interp1d(x, y, fill_value=1.0e-8,bounds_error=False)
            #kde=interp1d(x, y,bounds_error=False)
            #---------------------------------------------------------------
            #kde=UnivariateSpline(x,y, ext=3)
        elif self.method_kde=='distfit':
            
            pdf_fit=[]
            dist = distfit()
            dist.fit(marginal)
            kde=dist.model['model'].pdf(marginal)
            #kde=interp1d(x, y,bounds_error=False)
            #---------------------------------------------------------------
            #kde=UnivariateSpline(x,y, ext=3)
        self.kde=kde
        return kde
    
    def eval_pdf(self, inputData):  #to complete!
        if self.method_kde=='sklearn_kde_GS':
            kde_pdf=self.kde.score_samples(inputData.reshape(-1,1))
            dens=np.exp(kde_pdf)
        elif self.method_kde=='sklearn_kde':
            kde_pdf=self.kde.score_samples(inputData.reshape(-1,1))
            dens=np.exp(kde_pdf)
        elif self.method_kde=='scipy_kde':
            dens=self.kde.evaluate(inputData)
        elif self.method_kde=='statsmodel_kde':
            dens=self.kde.evaluate(inputData)
        elif self.method_kde=='KDEpy':
            # y= kde.evaluate(x_grid)
            # l=min(inputData) ; u=max(inputData)
            # d=10
            # x_grid = np.linspace(-d+l, d+u, num=2**10)
            # f = interp1d(x, y, kind="spline", assume_sorted=True)
            dens=self.kde(inputData)
        #--------- option pobs with fftkde-------------------------------
        elif self.method_kde=='pobs':
            # y= kde.evaluate(x_grid)
            # l=min(inputData) ; u=max(inputData)
            # d=10
            # x_grid = np.linspace(-d+l, d+u, num=2**10)
            # f = interp1d(x, y, kind="spline", assume_sorted=True)
            dens=self.kde(inputData)
        elif self.method_kde=='distfit':
            # y= kde.evaluate(x_grid)
            # l=min(inputData) ; u=max(inputData)
            # d=10
            # x_grid = np.linspace(-d+l, d+u, num=2**10)
            # f = interp1d(x, y, kind="spline", assume_sorted=True)
            dens=self.kde(inputData)
    
        self.dens=dens
        return dens


def form_options(marginal, method_kde, options_kde: dict):

    if method_kde=='scipy_kde':
        bwidth=0.2
        return merge_dict({
                'bw_method':bwidth / marginal.std(ddof=1),
        }, options_kde)
    elif method_kde=='sklearn_kde':
        return merge_dict({
                'kernel': 'gaussian',
                'bandwidth':0.2,
        }, options_kde)
    elif method_kde==('statsmodel_kde'):
        return merge_dict({
                'kernel': 'gau',
                'bw':'silverman',
                'fft':True
                }, options_kde)
    elif method_kde==('KDEpy'):
        return merge_dict({
                #'kernel': 'cosine',
                #'bw':'ISJ',
                'bw':'silverman',
                #'bw':0.1,
                #'bw':1 + ((len(marginal)**(1/3))),
            
        }, options_kde)
    #-----------option pobs with fftkde-----------------------------
    elif method_kde==('pobs'):
        return merge_dict({
                #'kernel': 'cosine',
                'bw':'ISJ',
                #'bw':'silverman',
                #'bw':0.2,
                #'bw':1 + ((len(marginal)**(1/3))),
            
        }, options_kde)
