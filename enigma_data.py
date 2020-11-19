import os
import logging
import numpy as np
import pandas as pd
import pickle
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from make_design_matrix import make_design_matrix,standardize_across_features


class enigma_data(object):

	def __init__( self, dfile, predict, predict_val, data_opt="AD" ):

		self.opt = data_opt
		self.data = {}

		if self.opt == "AD":
			self.feature_range = [ 'ACR_L', 'UNC_R' ] 
			self.remove_columns = [ "SubjID","Dx","JME", "Handedness","DURILL", "AO", "RESPONSE", "ENGEL", "TLEMTS", "TLEnonlesion", "patientsexunspecified" ]

		elif self.opt == "FA":
			self.feature_range = [ 'ACR_L', 'UNC_R' ] 
			self.remove_columns = [ "SubjID","Dx","JME", "Handedness","DURILL", "AO", "RESPONSE", "ENGEL", "TLEMTS", "TLEnonlesion", "patientsexunspecified" ]

		elif self.opt == "MD":
			self.feature_range = [ 'ACR_L', 'UNC_R' ] 
			self.remove_columns = [ "SubjID","Dx","JME", "Handedness","DURILL", "AO", "RESPONSE", "ENGEL", "TLEMTS", "TLEnonlesion", "patientsexunspecified" ]

		elif self.opt == "RD":
			self.feature_range = [ 'ACR_L', 'UNC_R' ] 
			self.remove_columns = [ "SubjID","Dx","JME", "Handedness","DURILL", "AO", "RESPONSE", "ENGEL", "TLEMTS", "TLEnonlesion", "patientsexunspecified" ]

		elif self.opt == "DD":
			self.feature_range = [ 'ACR_L_AD', 'UNC_R_RD' ] 
			self.remove_columns = [ "SubjID","Dx","JME", "Handedness","DURILL", "AO", "RESPONSE", "ENGEL", "TLEMTS", "TLEnonlesion", "patientsexunspecified" ]

		else:
			self.feature_range = [ 'L_LatVent', 'R_insula_surfavg' ] 
			self.remove_columns = ["SubjID","Dx","Handedness","DURILL", "AO"]

		self.DEBUG = True

		if os.path.exists( dfile ):

			self.dframe = pd.read_csv( dfile )
      
			if self.DEBUG:
				print( 'Found classes: ' + str( self.dframe.loc[:,predict].unique() ) )
			else:
				assert isinstance( dfile, object )

		self.predict = predict		# what are we predicting
		self.predict_val = predict_val	# what is the prediction value (or label)
		#self.harmonize = harmonize	# run combate (yes or no)

		'''if ( self.harmonize ) and ( batch is not None ) and ( covariates is not None ):
			self.covariates = covariates
			self.batch = batch
		elif self.harmonize:
			logging.warning("Batch or covariates is not defined ... unable to harmonize" )
			self.harmonize = False'''

	# --------------------------------------------------
	# These two methods are primarly used for debugging
	#	getFrame()
	# 	XY()
	# 
	# Future verisons, will most likely depreciate these
	# --------------------------------------------------
	def getFrame( self ):
		return self.dframe # this is primarly used for debugging

	def XY( self ):
		self.X = self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1] ]
		self.Y = self.dframe.loc[:, self.predict ]
		return self
	# --------------------------------------------------

	def save_dframe_as_csv( self, file_name ):
		self.dframe.to_csv( file_name )

	def save_data( self, file_name ):
		pickle.dump( self.data, open( file_name, "wb" ) ) 
		
	def load_data( self, file_name ):
		self.data = pickle.load( open( file_name, "rb" ) )
		

	def partition( self, train_validation_percent=0.75, perm_train_labels=False ):
		self.Covars = self.dframe.loc[:,["Site","Sex","Age",self.predict]]
		self.X = self.dframe.loc[ :, self.feature_range[0]:self.feature_range[1]]
        
		self.X = pd.concat([self.X, self.Covars], axis=1, sort=False)
		self.X['combined'] = self.X[self.predict].astype(str) + "_" + self.X['Site'].astype(str)
        
		vc = self.X["combined"].value_counts().loc[lambda x: x>9].index.tolist()
		self.X = self.X.loc[self.X["combined"].isin(vc)]   

		train, test = train_test_split( self.X, 
															 test_size=(1 - train_validation_percent), 
															 random_state = None, stratify = self.X["combined"] )

		train[[self.predict,'Site']] = train.combined.str.split("_",expand=True) 
		train["Site"] = train["Site"].astype(int)
		train[self.predict] = train[self.predict].astype(int)  
        
		self.data['Covars_train'] = train[["Site","Sex","Age"]]   
                                     
		self.data['X_train'] = train.drop(columns=['Site',self.predict, 'Sex', "Age","combined"])
		self.data["X_train"]["index"]=range(0,len(self.data['X_train']))
		self.data['Y_train'] = train[self.predict]
        
		if perm_train_labels:
			self.data['Y_train'] = shuffle( Y_train, random_state=None )

		self.data['X_test'] = test.drop(columns=['Site',self.predict, 'Sex', "Age","combined"])
		self.data["X_test"]["index"]=range(0,len(self.data['X_test']))
		self.data['Y_test'] = test[self.predict]

		return self.data


	def parse( self, balance=False ):

		# remove unwanted columns from the data
		self.dframe.drop( columns=self.remove_columns, axis=1, inplace=True )

		# remove subjects that have one (or more) NA or negative values 
		for i in range( 4, len( self.dframe.columns ) ):
			self.dframe.drop( self.dframe[ self.dframe.iloc[:,i].isnull() ].index, axis=0, inplace=True )
			self.dframe.drop( self.dframe[ self.dframe.iloc[:,i] <= 0 ].index, axis=0, inplace=True )

		# remove subjects from the dataframe that are not HC (0) or the SDx prediction value
		self.dframe = self.dframe[ ( self.dframe[ self.predict ] == 0 ) | ( self.dframe[ self.predict ] == self.predict_val ) ].copy()

		# probably not necessary, but just in case, replace SDx value with one. Now the labels are just 0 and 1 (and not 0 and 3 or 4 or 5 or 6)
		self.dframe[ self.predict ].replace( self.predict_val, 1, inplace=True )
        
		sites=self.dframe[ "Site" ].unique()
		id = 1
		for site in sites:
			idd = self.dframe[ "Site" ] == site
			self.dframe[ "Site" ].loc[idd] = id
			id = id + 1
            
		if balance:

			o_lab = self.dframe.loc[ self.dframe[ self.predict ] == 1 ]
			z_lab = self.dframe.loc[ self.dframe[ self.predict ] == 0 ].sample( n=len( o_lab ), random_state=None )
			
			self.dframe = pd.concat( [ o_lab, z_lab ] )


		return self	


	def parse_engel( self ):

		# remove unwanted columns from the data (but keep ENGEL column)
		self.remove_columns.remove( "ENGEL" )
		self.dframe.drop( columns=self.remove_columns, axis=1, inplace=True )

		# remove subjects that have one (or more) NA or negative values 
		for i in range( 4, len( self.dframe.columns ) ):
			self.dframe.drop( self.dframe[ self.dframe.iloc[:,i].isnull() ].index, axis=0, inplace=True )
			self.dframe.drop( self.dframe[ self.dframe.iloc[:,i] <= 0 ].index, axis=0, inplace=True )

		# probably not necessary, but just in case, replace SDx value with one. Now the labels are just 0 and 1 (and not 0 and 3 or 4 or 5 or 6)
		#self.dframe[ self.predict ].replace( self.predict_val, 1, inplace=True )

		# remove engel values > 4 (i.e. 5 and above )
		#self.dframe.drop( self.dframe[ self.predict ] > 4 ].index, axis=0, inplace=True )
		self.dframe.drop( self.dframe[ self.dframe[ self.predict ] > 4 ].index, axis=0, inplace=True )
		self.dframe[ self.predict ] = ( self.dframe[ self.predict ] > self.predict_val ).astype(int)



		return self	

		

