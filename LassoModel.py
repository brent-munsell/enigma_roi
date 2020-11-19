from sklearn.feature_selection import SelectFromModel
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt

class LassoModel:
    
    def __init__( self ):
        
        self.grid_dict = { 'lasso__alpha': [ 0.01, 0.05, 0.1, 0.2, 0.3 ] }
        
        self.name = 'lasso'
        
    def get_name( self ):
        
        return self.name
        
    def construct( self, alpha_term=0.1, debug=False ):
        
        if debug:
            self.print( "Alpha = {0:.3f}".format( alpha_term ) )
        
        self.model = SelectFromModel( lm.Lasso( alpha=alpha_term ) )
        
        return self.model
    
    
    def set_grid_dict( self, grid_dict = { 'lasso__alpha': [ 0.01, 0.05, 0.1, 0.2, 0.3 ] } ):
        
        self.grid_dict = grid_dict
        
    def get_grid_dict( self ):
        
        return self.grid_dict

    def get( self ):
        
        return self.model
