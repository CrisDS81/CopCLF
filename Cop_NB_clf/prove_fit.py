
"""
Created on Thu Feb 10 11:30:23 2022

@author: Cristiano
"""
import numpy as np
from distfit import distfit

# Generate 10000 normal distribution samples with mean 0, std dev of 3 
X = np.random.normal(0, 3, 100000)

# Initialize distfit
dist = distfit(bins=150)

# Determine best-fitting probability distribution for data
dist.fit_transform(X)# -*- coding: utf-8 -*-


