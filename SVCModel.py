import sklearn.svm as svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SVCModel(BaseEstimator, ClassifierMixin):

	def __init__( self, name="classifier", C=0.1, max_iter=10000, debug=False ):
		self.name = name
		self.C = C
		self.debug = debug
		self.model = None
		self.max_iter = max_iter
		self.grid_dict = { 'classifier__C': [ 0.1, 0.25, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.0  ] }

	def get_name( self ):
		return self.name


	def construct( self ):
		if self.debug:
			self.print( "C = {0:.2f}, max_iter = {1:d}".format( self.C, self.max_iter ) )

		self.model = svm.LinearSVC( penalty="l2", dual=True, C=self.C, max_iter=self.max_iter )
		return self.model


	def fit( self, X, y ):

		X, y = check_X_y( X, y )

		if self.model is None:
			self.construct()
      
		self.model.fit(X, y) 
		return self
    
	def get_grid_dict( self ):
        
		return self.grid_dict

	def get_weights( self ):

		return self.model.coef_[0]

	def predict( self, X ):
		new_X = np.nan_to_num(X)          
		return self.model.predict( new_X )

	def predict_proba( self, X ):

		probs = self.model.predict_proba( X )

		if probs.shape[1] == 1:
			probs = np.hstack( [1 - probs, probs] )

		return probs

	def score( self, X, y ):

		outputs = self.model.evaluate( X, y )

		if not isinstance( outputs, list ):
			outputs = [ outputs ]

		for name, output in zip( self.model.metrics_names, outputs ):
			if name in ["accuracy", "acc"]:
				return output

		raise ValueError('The model is not configured to compute accuracy. '
                     'You should pass `metrics=["accuracy"]` to '
                     'the `model.compile()` method.')


	def set_grid_dict( self, grid_dict = { 'classifier__C': [ 1.0 ] } ):
		self.grid_dict = grid_dict

	def get_grid_dict( self ):
		return self.grid_dict

	
