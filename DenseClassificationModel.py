from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

class DenseClassificationModel(BaseEstimator, ClassifierMixin):

	def __init__( self, name="classifier", input_dimension=0, learn_rate=0.01, hidden_units_L1=3, hidden_units_L2=20, l2_reg_penalty=0.1, drop_out_rate=0.2, optimizer="Adadelta", checkpoint_folder="./chpt", checkpoint_file=None, monitor="val_accuracy", batch_size=20, validation_split=0.25, epochs=250, verbose=0, mode="max", debug=False):
		
		self.name = name
		self.input_dimension = input_dimension
		self.learn_rate = learn_rate
		self.hidden_units_L1 = hidden_units_L1
		self.hidden_units_L2 = hidden_units_L2
		self.l2_reg_penalty = l2_reg_penalty
		self.drop_out_rate = drop_out_rate
		self.optimizer = optimizer
		self.grid_dict = { 'classifier__learn_rate': [ 0.001, 0.01 ], 'classifier__epochs': [ 2 ],'classifier__hidden_units_L1': [ 10, 20 ], 'classifier__hidden_units_L2': [ 3, 5, 7 ], 'classifier__l2_reg_penalty': [ 0.1, 0.2 ], 'classifier__drop_out_rate': [ 0.2, 0.3 ] }
		self.checkpoint_folder = checkpoint_folder
		self.checkpoint_file = checkpoint_file
		self.monitor = monitor
		self.batch_size = batch_size
		self.validation_split = validation_split
		self.mode = mode
		self.epochs = epochs
		self.verbose = verbose
		self.model = None
		self.debug = debug

		print( "input_dimension = {0:d}".format( self.input_dimension ) )

	def get_name( self ):

		return self.name

	def construct( self ):
        
       
  #SGD

#  - Stochastic Gradient Descent 를 지칭, 다음과같은 인자를 취함

#  - Learning rate

#  - Momentum : local minima에 빠지지 않기위해 이전 단계에서의 가중치가 적용된 평균을 사용

#  - Nesterov Momentum : solution에 가까워 질 수록 gradient를 slow down시킴
#   -Moment는 global minimum optimal을 찾아도 관성때문에 계속 가려고함 
 
# -그래서 NAG를함   NAG는 moment를 한곳에서 gradient를 찾음 



#  · ADAM

#  - Adaptive Moment Estimation 을 지칭
#  -gradient값부분을 제곱해줌으로써 step size가 얼마나 큰지 알려줌
#  -각각의 stepsize마다 learning rate도 바꿔줌 (stepsize가 크면 learning rate를 작게해서 w값을 높여주고 그리고 반대)

#  - 이전 step에서의 평균뿐 아니라 분산까지 고려한 복잡한 지수 감쇠(exponential decay)를 사용
#  -문제점 : G(t)가 계속 제곱된 양수값을 더하니깐 계속 커지기만함 stepsize는 G(t)의 반비례하는 값이니 G(t)가 커지니간 나중에 0에로됨
#  그래서 해결해서 나온게 RMSProp




#  · RMSProp

#  - RMS : Root Mean Squeared Error

#  - 말그대로 지수 감쇠 squared gradients의 평균으로 나눔으로써 learning rate를 감소시킴


       

		if self.optimizer == "SGD":
			opt = SGD( learning_rate = self.learn_date ) ##stochastic gradient 
		elif self.optimizer == "Adam":
			opt = Adam( learning_rate = self.learn_rate ) ##Adaptive Moment Estimation 
		elif self.optimizer == "RMSprop":
			opt = RMSprop( learning_rate = self.learn_rate )
		elif self.optimizer == "Adagrad":
			opt = Adaggrad()
		else:
			opt = Adadelta()

		self.model = Sequential()        
		self.dense_layers = []
		self.arch = []
        
		if self.debug:
			self.print( "lr={0:.6f}, hu_L1={1:d}, hu_L2={2:d}, l2_pen={3:.3f}".format( self.learn_rate, self.hidden_units_L1, self.hidden_units_L2, self.l2_reg_penalty ) )    

		self.model.add( Dense( units=self.hidden_units_L1, activation='relu', input_shape=( self.input_dimension, ) ) )
		self.dense_layers.append( 0 )
		self.arch.append( self.input_dimension )
		self.arch.append( self.hidden_units_L1 )   
		self.model.add( Dropout( rate=self.drop_out_rate ) )    
		self.model.add( Dense( units=self.hidden_units_L2, kernel_regularizer=l2( self.l2_reg_penalty ), activation='relu' ) )                           
		self.dense_layers.append( 2 )
		self.arch.append( self.hidden_units_L2 )
		self.model.add( Dense( units=1, activation='sigmoid' ) )
		self.model.compile( optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'] )  

		return self.model
    
	def get_weights( self ):

		W = self.model.get_weights()[ 0 ]
		print(W)        
		idx = W < ( np.max( np.abs( W ) )*0.1 )
		W[ idx ] = 0
		return np.sum( np.abs( W ), axis=1 )

	def fit( self, X, y ):

		X, y = check_X_y( X, y )

		cb_list = None

		if self.model is None:
			self.construct()

		if not os.path.exists( self.checkpoint_folder ):
			os.makedirs( self.checkpoint_folder )

		if self.checkpoint_file is not None:
			checkpoint_path = self.checkpoint_folder + "/" + self.checkpoint_file
			checkpoint = ModelCheckpoint( checkpoint_path, monitor=self.monitor, verbose=self.verbose, save_best_only=True, mode=self.mode )
			cb_list = [checkpoint]

		self.model.fit( X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks=cb_list, verbose=self.verbose )

		#if cb_list is not None:
		scores = []
		for hdf in os.listdir( self.checkpoint_folder ):          
			if os.path.isfile( os.path.join( self.checkpoint_folder, hdf ) ):
				toks = hdf.split(".")
				scores.append( ( toks[0] + "." + toks[1] ) )
		
		scores.sort( reverse=True )

		print( scores )
		
		if len(scores) > 0:
			if float( scores[0] ) > 0:
				try:
					if ( len( scores ) > 2 ) and ( float( scores[1] ) > 0.8 ):
						self.model.load_weights( ( self.checkpoint_folder + "/" + scores[1] + ".hdf5" ) )
						print( "Loaded checkpoint file [ {0:s} ]".format( ( self.checkpoint_folder + "/" + scores[1] + ".hdf5" ) ) )
					elif ( float( scores[0] ) > 0.3 ):
						self.model.load_weights( ( self.checkpoint_folder + "/" + scores[0] + ".hdf5" ) )
						print( "Loaded checkpoint file [ {0:s} ]".format( ( self.checkpoint_folder + "/" + scores[0] + ".hdf5" ) ) )

				except:
					print( "Unable to load checkpoint file [ {0:s} ]".format( ( self.checkpoint_folder + "/" + scores[0] + ".hdf5" ) ) )
		
		for f in os.listdir( self.checkpoint_folder ):
			file_path = os.path.join( self.checkpoint_folder, f )
			if os.path.isfile( file_path ):
				os.unlink( file_path )

		os.rmdir( self.checkpoint_folder )

		return self

	def predict( self, X ):

		y_p = self.model.predict( X )
		gidx = y_p >= 0.5
		lidx = y_p < 0.5
		y_p[gidx]=1
		y_p[lidx]=0
		return np.int32( y_p[:,0] )

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

	def set_grid_dict( self, grid_dict = { 'classifer__learn_rate': [ 0.01, 0.1 ] } ):

		self.grid_dict = grid_dict
        
	def get_grid_dict( self ):

		return self.grid_dict
