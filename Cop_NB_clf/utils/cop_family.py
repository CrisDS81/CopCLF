# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:36:59 2021

@author: Cristiano
"""
from copulae.elliptical import GaussianCopula, StudentCopula
from copulae.archimedean import FrankCopula, ClaytonCopula, GumbelCopula
from copulae import IndepCopula

def cop_family_new(cop_fam_opt: list):
    cop_fam = [
              ('Gaussian', GaussianCopula, 
              #('Student', StudentCopula),
              ('Frank', FrankCopula,
              ('Gumbel', GumbelCopula, 
              ('Clayton', ClaytonCopula,
              #('Indep', IndepCopula),
              ]
    if cop_fam_opt[0]=='all':
        cop_fam = cop_fam
    else:
        for cop in cop_fam_opt:
            a=cop_fam_opt
            cop_fam_new=[]
            for k in range(len(a)):
                for name, cop in cop_fam:
                    if name==a[k]:
                        cop_fam_new.append((name,cop))
            cop_fam=cop_fam_new      
        
    return cop_fam