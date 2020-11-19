#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import Constants as constants
import numpy as np
from DenseClassificationModel import DenseClassificationModel
from SVCModel import SVCModel


class GridSearch:
    def __init__( self, model, grid_result ):
        self.gdr_dict= {}
        self.model = model
        self.grid_result = grid_result
       
        if ( isinstance( self.model, SVCModel ) ):
            self.gdr_dict["best_params"] = self.grid_result.best_params_
            self.gdr_dict["classifier__C"] = self.grid_result.best_params_['classifier__C']
        else:
            self.gdr_dict["best_params"] = self.grid_result.best_params_
            self.gdr_dict['classifier__learn_rate'] = self.grid_result.best_params_['classifier__learn_rate']
            self.gdr_dict['classifier__epochs'] = self.grid_result.best_params_['classifier__epochs']
            self.gdr_dict['classifier__hidden_units_L1'] = self.grid_result.best_params_['classifier__hidden_units_L1']
            self.gdr_dict['classifier__hidden_units_L2'] = self.grid_result.best_params_['classifier__hidden_units_L2']
            self.gdr_dict['classifier__l2_reg_penalty'] = self.grid_result.best_params_['classifier__l2_reg_penalty']
            self.gdr_dict['classifier__drop_out_rate'] = self.grid_result.best_params_['classifier__drop_out_rate']
        

