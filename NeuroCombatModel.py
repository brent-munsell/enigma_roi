#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from enigma_data import enigma_data
from neurocombat_sklearn import CombatModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

class NeuroCombatModel:
    def __init__(self, data, sites, discrete_covariates=None, continuous_covariates=None ):
                    self.data = data.getFrame()
                    self.sites = sites
                    self.discrete_covariates = discrete_covariates
                    self.continuous_covariates = continuous_covariates
                    self.sites_new = None
                    self.discrete_covariates_new = None
                    self.continuous_covariates_new = None
                    self.dropped_index_X = None
                    
                    # Creating model
                    self.model = CombatModel()
                    
    def fit(self,X,y):
        i = X[:,-1].astype(int)
        self.sites_new = self.sites.iloc[i]
        self.discrete_covariates_new = self.discrete_covariates.iloc[i]
        self.continuous_covariates_new = self.continuous_covariates.iloc[i]
        self.dropped_index_X = np.delete(X, -1, 1)
        
        return self.model.fit(self.dropped_index_X,self.sites_new,self.discrete_covariates_new,self.continuous_covariates_new)
    
    def transform(self,X):
            i = X[:,-1].astype(int)
            self.sites_new = self.sites.iloc[i] 
            self.discrete_covariates_new = self.discrete_covariates.iloc[i]
            self.continuous_covariates_new = self.continuous_covariates.iloc[i]
            self.dropped_index_X = np.delete(X, -1, 1)
            
            return self.model.transform(self.dropped_index_X,self.sites_new,self.discrete_covariates_new,self.continuous_covariates_new)
        
    def fit_transform(self,X,y): 
            i = X[:,-1].astype(int)
            self.sites_new = self.sites.iloc[i]
            self.discrete_covariates_new = self.discrete_covariates.iloc[i]
            self.continuous_covariates_new = self.continuous_covariates.iloc[i]
            self.dropped_index_X = np.delete(X, -1, 1)
           
            return self.model.fit(self.dropped_index_X, self.sites_new, self.discrete_covariates_new,self.continuous_covariates_new).transform(self.dropped_index_X, self.sites_new, self.discrete_covariates_new,self.continuous_covariates_new)

  
        
