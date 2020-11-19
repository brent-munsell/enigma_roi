from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from scipy.interpolate import UnivariateSpline, splev, splrep, interp1d
from joblib import dump, load
import scipy.stats as stats
import Constants as constants
import numpy as np
import os
import pandas as pd
import warnings
from sklearn.utils.multiclass import unique_labels

import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class PyEnigmaPlots(object):

	def __init__( self, predictor, feature, root_dir="./", save_dir="./", img_format=".svg", debug=False, interp=True ):
	
		self.interp = interp
		self.n_interp = 20
		self.xs = np.linspace( 0, 1, self.n_interp )
		self.spline_degree = 2

		self.predictor = str( predictor )
		self.feature = feature
		self.root_dir = root_dir
		self.save_dir = save_dir

		self.sv_data_nprm = None
		self.sv_data_perm = None
		self.dl_data_nprm = None
		self.dl_data_perm = None

		self.features = None

		self.debug = debug

		self.sv_it_nprm = list()
		self.sv_it_prm = list()
		self.dl_it_nprm = list()
		self.dl_it_prm = list()

		self.labels = None
		self.sv_title = None
		self.dl_title = None

		self.img_format = img_format

		self.txtbx_properties = dict( boxstyle='round', facecolor='lightgray', alpha=1.0 )
		
		if self.predictor == "engel":
			self.sv_data_nprm = os.path.join( root_dir, "{0:s}_arch/sv/engel/nprm/".format( self.feature.lower() ) )
			self.sv_data_perm = os.path.join( root_dir, "{0:s}_arch/sv/engel/prm/".format( self.feature.lower() ) )
			self.dl_data_nprm = os.path.join( root_dir, "{0:s}_arch/dl/engel/nprm/".format( self.feature.lower() ) )
			self.dl_data_perm = os.path.join( root_dir, "{0:s}_arch/dl/engel/prm/".format( self.feature.lower() ) )
			self.labels = [ "Engel 1", "Engel 2-4" ]
			self.title = "SVC: Engel 1 -vs- Engel 2-4"
		else:
			self.sv_data_nprm = os.path.join( root_dir, "{0:s}_arch/sv/{1:s}/nprm/".format( self.feature.lower(), self.predictor ) )
			self.sv_data_perm = os.path.join( root_dir, "{0:s}_arch/sv/{1:s}/prm/".format( self.feature.lower(), self.predictor ) )
			self.dl_data_nprm = os.path.join( root_dir, "{0:s}_arch/dl/{1:s}/nprm/".format( self.feature.lower(), self.predictor ) )
			self.dl_data_perm = os.path.join( root_dir, "{0:s}_arch/dl/{1:s}/prm/".format( self.feature.lower(), self.predictor ) )
			self.labels = [ "HC", "SDx" ]
			self.title = "HC -vs- " + feature.upper() + " (SDx= " +  predictor + ")"

		# -------------------------------------------------------
		# process svc data -- no label permutation
		# -------------------------------------------------------
		for jobfile in os.listdir( self.sv_data_nprm ):
			file_path = os.path.join( self.sv_data_nprm, jobfile )
			it_result = None
			if os.path.isfile( file_path ):
				try:
					if self.debug:
						print( file_path )
					it_result = load( file_path )
					self.sv_it_nprm.extend( it_result["iteration"] )
					if self.features is None:
						self.features = it_result["features"]
				except:
					print( "[ {0:s} ] file failed to load".format( file_path ) )
		
		self.tmp_matrix = np.zeros( ( len( self.sv_it_nprm ), 10 ) )

		if self.interp is True:
			self.roc_fpr_matrix = np.zeros( ( len( self.sv_it_nprm ), self.n_interp ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.sv_it_nprm ), self.n_interp ) )
		else:
			fpr, tpr, _ = roc_curve( self.sv_it_nprm[0]["yt"], self.sv_it_nprm[0]["yp"], pos_label=1 )
			self.roc_fpr_matrix = np.zeros( ( len( self.sv_it_nprm ), len( fpr ) ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.sv_it_nprm ), len( tpr ) ) )

		self.weight_mtx = np.zeros( ( len( self.sv_it_nprm ), len( self.sv_it_nprm[0]["w"] ) ) )
		
		# -------------------------
		# tmp matrix column description
		# -------------------------
		# column[0] = TN
		# column[1] = FP
		# column[2] = FN
		# column[3] = TP
		# column[4] = AUC
		# column[5] = PPV
		# column[6] = NPV
		# column[7] = ACC
		# column[8] = SEN
		# column[9] = SPC

		error_idx = []

		for i in range( 0, len( self.sv_it_nprm ) ):
			self.tmp_matrix[i,0] = self.sv_it_nprm[i]['tn']
			self.tmp_matrix[i,1] = self.sv_it_nprm[i]['fp']
			self.tmp_matrix[i,2] = self.sv_it_nprm[i]['fn']
			self.tmp_matrix[i,3] = self.sv_it_nprm[i]['tp']
			self.tmp_matrix[i,4] = self.sv_it_nprm[i]['auc']
			self.tmp_matrix[i,5] = ( self.sv_it_nprm[i]['tp'] / (self.sv_it_nprm[i]['tp'] + self.sv_it_nprm[i]['fp'] ) ) 	# PPV
			self.tmp_matrix[i,6] = ( self.sv_it_nprm[i]['tn'] / (self.sv_it_nprm[i]['tn'] + self.sv_it_nprm[i]['fn'] ) )	# NPV
			self.tmp_matrix[i,7] = ( ( self.sv_it_nprm[i]['tp'] + self.sv_it_nprm[i]['tn'] ) / ( self.sv_it_nprm[i]['tp'] + self.sv_it_nprm[i]['tn'] + self.sv_it_nprm[i]['fp'] + self.sv_it_nprm[i]['fn'] ) )	#ACC
			self.tmp_matrix[i,8] = ( self.sv_it_nprm[i]['tp'] / (self.sv_it_nprm[i]['tp'] + self.sv_it_nprm[i]['fn'] ) )	# SEN
			self.tmp_matrix[i,9] = ( self.sv_it_nprm[i]['tn'] / (self.sv_it_nprm[i]['tn'] + self.sv_it_nprm[i]['fp'] ) )	# SPC
			
			fpr, tpr, _ = roc_curve( self.sv_it_nprm[i]["yt"], self.sv_it_nprm[i]["yp"], pos_label=1 )

			# if fpr or tpr only has two values, then this defines a linear line starting at 0 and ending at 1
			# we need at least 3 values, so simply average the endpoint values. 
			if len( fpr ) == 2:
				fpr = np.insert( fpr, 1, ( fpr[0]+fpr[1] / 2 ) )

			if len( tpr ) == 2:
				tpr = np.insert( tpr, 1, ( tpr[0]+tpr[1] / 2 ) )

			if self.interp is True:
				try:
					spl = interp1d( fpr, tpr )
					self.roc_tpr_matrix[i,:] = spl( self.xs )
					self.roc_fpr_matrix[i,:] = self.xs
				except:
					error_idx.append( i )
			else:
				self.roc_tpr_matrix[i,:] = tpr
				self.roc_fpr_matrix[i,:] = fpr

			self.weight_mtx[i,:] =  ( self.sv_it_nprm[i]['w'] / np.max( self.sv_it_nprm[i]['w'] ) )

		# if an NaN exists, then set to zero (divide by zero situtation)
		if np.count_nonzero( np.isnan( self.tmp_matrix ) ) > 0:
			np.nan_to_num( self.tmp_matrix, copy=False )

		# remove tpr or fpr errors (i.e. remove matrix rows with zero values)
		self.roc_tpr_matrix = np.delete( self.roc_tpr_matrix, error_idx, 0 )
		self.roc_fpr_matrix = np.delete( self.roc_fpr_matrix, error_idx, 0 )

		# compute the mean and stdev for each column in the tmp_matrix
		self.sv_nprm_res_xbar = np.mean( self.tmp_matrix, axis=0 )
		self.sv_nprm_res_std = np.std( self.tmp_matrix, axis=0 )
		self.sv_nprm_roc_fpr_xbar = np.mean( self.roc_fpr_matrix, axis=0 )
		self.sv_nprm_roc_fpr_std = np.std( self.roc_fpr_matrix, axis=0 )
		self.sv_nprm_roc_tpr_xbar = np.mean( self.roc_tpr_matrix, axis=0 )
		self.sv_nprm_roc_tpr_std = np.std( self.roc_tpr_matrix, axis=0 )
		self.sv_nprm_weight_mtx = np.sum( np.abs( self.weight_mtx ), axis=0 )

		filename = "svc_nprm_tm_" + self.predictor + "_" + self.feature + ".csv"
		np.savetxt( os.path.join( self.save_dir, filename ), self.tmp_matrix, delimiter=",")

		# -------------------------------------------------------
		# process svc data -- permutated labels
		# -------------------------------------------------------
		for jobfile in os.listdir( self.sv_data_perm ):
			file_path = os.path.join( self.sv_data_perm, jobfile )
			it_result = None
			if os.path.isfile( file_path ):
				try:
					if self.debug:
						print( file_path )
					it_result = load( file_path )
					self.sv_it_prm.extend( it_result["iteration"] )
				except:
					print( "[ {0:s} ] file failed to load".format( file_path ) )

		self.tmp_matrix = np.zeros( ( len( self.sv_it_prm ), 10 ) )

		if self.interp is True:
			self.roc_fpr_matrix = np.zeros( ( len( self.sv_it_prm ), self.n_interp ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.sv_it_prm ), self.n_interp ) )
		else:
			fpr, tpr, _ = roc_curve( self.sv_it_prm[0]["yt"], self.sv_it_prm[0]["yp"], pos_label=1 )
			self.roc_fpr_matrix = np.zeros( ( len( self.sv_it_prm ), len( fpr ) ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.sv_it_prm ), len( tpr ) ) )

		self.weight_mtx = np.zeros( ( len( self.sv_it_prm ), len( self.sv_it_prm[0]["w"] ) ) )

		error_idx = []

		for i in range( 0, len( self.sv_it_prm ) ):
			self.tmp_matrix[i,0] = self.sv_it_prm[i]['tn']
			self.tmp_matrix[i,1] = self.sv_it_prm[i]['fp']
			self.tmp_matrix[i,2] = self.sv_it_prm[i]['fn']
			self.tmp_matrix[i,3] = self.sv_it_prm[i]['tp']
			self.tmp_matrix[i,4] = self.sv_it_prm[i]['auc']
			self.tmp_matrix[i,5] = ( self.sv_it_prm[i]['tp'] / (self.sv_it_prm[i]['tp'] + self.sv_it_prm[i]['fp'] ) ) 	
			self.tmp_matrix[i,6] = ( self.sv_it_prm[i]['tn'] / (self.sv_it_prm[i]['tn'] + self.sv_it_prm[i]['fn'] ) )	
			self.tmp_matrix[i,7] = ( ( self.sv_it_prm[i]['tp'] + self.sv_it_prm[i]['tn'] ) / ( self.sv_it_prm[i]['tp'] + self.sv_it_prm[i]['tn'] + self.sv_it_prm[i]['fp'] + self.sv_it_prm[i]['fn'] ) )
			self.tmp_matrix[i,8] = ( self.sv_it_prm[i]['tp'] / (self.sv_it_prm[i]['tp'] + self.sv_it_prm[i]['fn'] ) )	
			self.tmp_matrix[i,9] = ( self.sv_it_prm[i]['tn'] / (self.sv_it_prm[i]['tn'] + self.sv_it_prm[i]['fp'] ) )	

			fpr, tpr, _ = roc_curve( self.sv_it_prm[i]["yt"], self.sv_it_prm[i]["yp"], pos_label=1 )

			if len( fpr ) == 2:
				fpr = np.insert( fpr, 1, ( fpr[0]+fpr[1] / 2 ) )

			if len( tpr ) == 2:
				tpr = np.insert( tpr, 1, ( tpr[0]+tpr[1] / 2 ) )

			if self.interp is True:
				try:
					spl = interp1d( fpr, tpr )
					self.roc_tpr_matrix[i,:] = spl( self.xs )
					self.roc_fpr_matrix[i,:] = self.xs
				except:
					error_idx.append( i )
			else:
				self.roc_tpr_matrix[i,:] = tpr
				self.roc_fpr_matrix[i,:] = fpr

			self.weight_mtx[i,:] =  ( self.sv_it_prm[i]['w'] / np.max( self.sv_it_prm[i]['w'] ) )

		if np.count_nonzero( np.isnan( self.tmp_matrix ) ) > 0:
			np.nan_to_num( self.tmp_matrix, copy=False )

		self.roc_tpr_matrix = np.delete( self.roc_tpr_matrix, error_idx, 0 )
		self.roc_fpr_matrix = np.delete( self.roc_fpr_matrix, error_idx, 0 )

		self.sv_prm_res_xbar = np.mean( self.tmp_matrix, axis=0 )
		self.sv_prm_res_std = np.std( self.tmp_matrix, axis=0 )
		self.sv_prm_roc_fpr_xbar = np.mean( self.roc_fpr_matrix, axis=0 )
		self.sv_prm_roc_fpr_std = np.std( self.roc_fpr_matrix, axis=0 )
		self.sv_prm_roc_tpr_xbar = np.mean( self.roc_tpr_matrix, axis=0 )
		self.sv_prm_roc_tpr_std = np.std( self.roc_tpr_matrix, axis=0 )
		self.sv_prm_weight_mtx = np.sum( np.abs( self.weight_mtx ), axis=0 )

		filename = "svc_prm_tm_" + self.predictor + "_" + self.feature + ".csv"
		np.savetxt( os.path.join( self.save_dir, filename ), self.tmp_matrix, delimiter=",")

		# -------------------------------------------------------
		# process dl data -- no label permutation
		# -------------------------------------------------------
		for jobfile in os.listdir( self.dl_data_nprm ):
			file_path = os.path.join( self.dl_data_nprm, jobfile )
			it_result = None
			if os.path.isfile( file_path ):
				try:
					if self.debug:
						print( file_path )
					it_result = load( file_path )
					self.dl_it_nprm.extend( it_result["iteration"] )
				except:
					print( "[ {0:s} ] file failed to load".format( file_path ) )

		self.tmp_matrix = np.zeros( ( len( self.dl_it_nprm ), 10 ) )

		if self.interp is True:
			self.roc_fpr_matrix = np.zeros( ( len( self.dl_it_nprm ), self.n_interp ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.dl_it_nprm ), self.n_interp ) )
		else:
			fpr, tpr, _ = roc_curve( self.dl_it_nprm[0]["yt"], self.dl_it_nprm[0]["yp"], pos_label=1 )
			self.roc_fpr_matrix = np.zeros( ( len( self.dl_it_nprm ), len( fpr ) ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.dl_it_nprm ), len( tpr ) ) )

		self.weight_mtx = np.zeros( ( len( self.dl_it_nprm ), len( self.dl_it_nprm[0]["w"] ) ) )

		error_idx = []

		for i in range( 0, len( self.dl_it_nprm ) ):
			self.tmp_matrix[i,0] = self.dl_it_nprm[i]['tn']
			self.tmp_matrix[i,1] = self.dl_it_nprm[i]['fp']
			self.tmp_matrix[i,2] = self.dl_it_nprm[i]['fn']
			self.tmp_matrix[i,3] = self.dl_it_nprm[i]['tp']
			self.tmp_matrix[i,4] = self.dl_it_nprm[i]['auc']
			self.tmp_matrix[i,5] = ( self.dl_it_nprm[i]['tp'] / (self.dl_it_nprm[i]['tp'] + self.dl_it_nprm[i]['fp'] ) ) 	
			self.tmp_matrix[i,6] = ( self.dl_it_nprm[i]['tn'] / (self.dl_it_nprm[i]['tn'] + self.dl_it_nprm[i]['fn'] ) )	
			self.tmp_matrix[i,7] = ( ( self.dl_it_nprm[i]['tp'] + self.dl_it_nprm[i]['tn'] ) / ( self.dl_it_nprm[i]['tp'] + self.dl_it_nprm[i]['tn'] + self.dl_it_nprm[i]['fp'] + self.dl_it_nprm[i]['fn'] ) )	
			self.tmp_matrix[i,8] = ( self.dl_it_nprm[i]['tp'] / (self.dl_it_nprm[i]['tp'] + self.dl_it_nprm[i]['fn'] ) )	
			self.tmp_matrix[i,9] = ( self.dl_it_nprm[i]['tn'] / (self.dl_it_nprm[i]['tn'] + self.dl_it_nprm[i]['fp'] ) )	

			fpr, tpr, _ = roc_curve( self.dl_it_nprm[i]["yt"], self.dl_it_nprm[i]["yp"], pos_label=1 )
 
			if len( fpr ) == 2:
				fpr = np.insert( fpr, 1, ( fpr[0]+fpr[1] / 2 ) )

			if len( tpr ) == 2:
				tpr = np.insert( tpr, 1, ( tpr[0]+tpr[1] / 2 ) )

			if self.interp is True:
				try:
					spl = interp1d( fpr, tpr ) 
					self.roc_tpr_matrix[i,:] = spl( self.xs )
					self.roc_fpr_matrix[i,:] = self.xs
				except:
					error_idx.append( i )
			else:
				self.roc_tpr_matrix[i,:] = tpr
				self.roc_fpr_matrix[i,:] = fpr

			self.weight_mtx[i,:] =  ( self.dl_it_nprm[i]['w'] / np.max( self.dl_it_nprm[i]['w'] ) )

		if np.count_nonzero( np.isnan( self.tmp_matrix ) ) > 0:
			np.nan_to_num( self.tmp_matrix, copy=False )

		self.roc_tpr_matrix = np.delete( self.roc_tpr_matrix, error_idx, 0 )
		self.roc_fpr_matrix = np.delete( self.roc_fpr_matrix, error_idx, 0 )

		self.dl_nprm_res_xbar = np.mean( self.tmp_matrix, axis=0 )
		self.dl_nprm_res_std = np.std( self.tmp_matrix, axis=0 )
		self.dl_nprm_roc_fpr_xbar = np.mean( self.roc_fpr_matrix, axis=0 )
		self.dl_nprm_roc_fpr_std = np.std( self.roc_fpr_matrix, axis=0 )
		self.dl_nprm_roc_tpr_xbar = np.mean( self.roc_tpr_matrix, axis=0 )
		self.dl_nprm_roc_tpr_std = np.std( self.roc_tpr_matrix, axis=0 )
		self.dl_nprm_weight_mtx = np.sum( self.weight_mtx, axis=0 )

		filename = "dl_nprm_tm_" + self.predictor + "_" + self.feature + ".csv"
		np.savetxt( os.path.join( self.save_dir, filename ), self.tmp_matrix, delimiter=",")

		# -------------------------------------------------------
		# process dl data -- permutated labels
		# -------------------------------------------------------
		for jobfile in os.listdir( self.dl_data_perm ):
			file_path = os.path.join( self.dl_data_perm, jobfile )
			it_result = None
			if os.path.isfile( file_path ):
				try:
					if self.debug:
						print( file_path )
					it_result = load( file_path )
					self.dl_it_prm.extend( it_result["iteration"] )
				except:
					print( "[ {0:s} ] file failed to load".format( file_path ) )


		self.tmp_matrix = np.zeros( ( len( self.dl_it_prm ), 10 ) )

		if self.interp is True:
			self.roc_fpr_matrix = np.zeros( ( len( self.dl_it_prm ), self.n_interp ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.dl_it_prm ), self.n_interp ) )
		else:
			fpr, tpr, _ = roc_curve( self.dl_it_prm[10]["yt"], self.dl_it_prm[10]["yp"], pos_label=1 )
			self.roc_fpr_matrix = np.zeros( ( len( self.dl_it_prm ), 3 ) )
			self.roc_tpr_matrix = np.zeros( ( len( self.dl_it_prm ), 3 ) )

		self.weight_mtx = np.zeros( ( len( self.dl_it_prm ), len( self.dl_it_prm[0]["w"] ) ) )

		error_idx = []

		for i in range( 0, len( self.dl_it_prm ) ):
			self.tmp_matrix[i,0] = self.dl_it_prm[i]['tn']
			self.tmp_matrix[i,1] = self.dl_it_prm[i]['fp']
			self.tmp_matrix[i,2] = self.dl_it_prm[i]['fn']
			self.tmp_matrix[i,3] = self.dl_it_prm[i]['tp']
			self.tmp_matrix[i,4] = self.dl_it_prm[i]['auc']
			self.tmp_matrix[i,5] = ( self.dl_it_prm[i]['tp'] / (self.dl_it_prm[i]['tp'] + self.dl_it_prm[i]['fp'] ) ) 
			self.tmp_matrix[i,6] = ( self.dl_it_prm[i]['tn'] / (self.dl_it_prm[i]['tn'] + self.dl_it_prm[i]['fn'] ) )
			self.tmp_matrix[i,7] = ( ( self.dl_it_prm[i]['tp'] + self.dl_it_prm[i]['tn'] ) / ( self.dl_it_prm[i]['tp'] + self.dl_it_prm[i]['tn'] + self.dl_it_prm[i]['fp'] + self.dl_it_prm[i]['fn'] ) )	
			self.tmp_matrix[i,8] = ( self.dl_it_prm[i]['tp'] / (self.dl_it_prm[i]['tp'] + self.dl_it_prm[i]['fn'] ) )
			self.tmp_matrix[i,9] = ( self.dl_it_prm[i]['tn'] / (self.dl_it_prm[i]['tn'] + self.dl_it_prm[i]['fp'] ) )

			fpr, tpr, _ = roc_curve( self.dl_it_prm[i]["yt"], self.dl_it_prm[i]["yp"], pos_label=1 )

			if len( fpr ) == 2:
				fpr = np.insert( fpr, 1, ( fpr[0]+fpr[1] / 2 ) )

			if len( tpr ) == 2:
				tpr = np.insert( tpr, 1, ( tpr[0]+tpr[1] / 2 ) )

			if self.interp is True:
				try:
					spl = interp1d( fpr, tpr ) 
					self.roc_tpr_matrix[i,:] = spl( self.xs )
					self.roc_fpr_matrix[i,:] = self.xs
				except:
					error_idx.append( i )
			else:
				self.roc_tpr_matrix[i,:] = tpr
				self.roc_fpr_matrix[i,:] = fpr

			self.weight_mtx[i,:] =  ( self.dl_it_prm[i]['w'] / np.max( self.dl_it_prm[i]['w'] ) )

		# print( self.tmp_matrix[:,7 ] )

		if np.count_nonzero( np.isnan( self.tmp_matrix ) ) > 0:
			np.nan_to_num( self.tmp_matrix, copy=False )

		# print( np.mean( self.tmp_matrix[:,7 ] ) )

		self.roc_tpr_matrix = np.delete( self.roc_tpr_matrix, error_idx, 0 )
		self.roc_fpr_matrix = np.delete( self.roc_fpr_matrix, error_idx, 0 )

		self.dl_prm_res_xbar = np.mean( self.tmp_matrix, axis=0 )
		self.dl_prm_res_std = np.std( self.tmp_matrix, axis=0 )
		self.dl_prm_roc_fpr_xbar = np.mean( self.roc_fpr_matrix, axis=0 )
		self.dl_prm_roc_fpr_std = np.std( self.roc_fpr_matrix, axis=0 )
		self.dl_prm_roc_tpr_xbar = np.mean( self.roc_tpr_matrix, axis=0 )
		self.dl_prm_roc_tpr_std = np.std( self.roc_tpr_matrix, axis=0 )
		self.dl_prm_weight_mtx = np.sum( self.weight_mtx, axis=0 )

		filename = "dl_prm_tm_" + self.predictor + "_" + self.feature + ".csv"
		np.savetxt( os.path.join( self.save_dir, filename ), self.tmp_matrix, delimiter=",")

		if self.debug:
			print( "len(sv_it_nprm) = {0:d}, len(sv_it_perm) = {1:d}".format( len(self.sv_it_nprm), len(self.sv_it_prm) ) )
			print( "len(dl_it_nprm) = {0:d}, len(dl_it_perm) = {1:d}".format( len(self.dl_it_nprm), len(self.dl_it_prm) ) )

	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def confusion_matrices( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		cm_sv_nprm = np.zeros( (2,2) )
		cm_dl_nprm = np.zeros( (2,2) )

		sd_sv_nprm = np.zeros( (2,2) )
		sd_dl_nprm = np.zeros( (2,2) )

		cm_sv_nprm[0,0] = self.sv_nprm_res_xbar[0]
		cm_sv_nprm[0,1] = self.sv_nprm_res_xbar[1]
		cm_sv_nprm[1,0] = self.sv_nprm_res_xbar[2]
		cm_sv_nprm[1,1] = self.sv_nprm_res_xbar[3]

		sd_sv_nprm[0,0] = self.sv_nprm_res_std[0]
		sd_sv_nprm[0,1] = self.sv_nprm_res_std[1]
		sd_sv_nprm[1,0] = self.sv_nprm_res_std[2]
		sd_sv_nprm[1,1] = self.sv_nprm_res_std[3]

		plt.rcParams.update({'font.size': 9})
		
		fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(8,4), dpi=120 )
		im = ax1.imshow(cm_sv_nprm, interpolation='nearest', cmap=plt.cm.Blues)
		# ax1.figure.colorbar( im, ax=ax1 )

		ax1.set( xticks=np.arange( cm_sv_nprm.shape[1]),
				 yticks=np.arange( cm_sv_nprm.shape[0]),
           		 xticklabels=self.labels, yticklabels=self.labels,
           		 title=("SVC: " + self.title),
           		 ylabel='True',
           		 xlabel='Predicted' ) 

		plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
		fmt = '.1f'
		thresh = cm_sv_nprm.max() / 2.

		for i in range( cm_sv_nprm.shape[0]):
			for j in range(cm_sv_nprm.shape[1]):
				ax1.text(j, i, ( format(cm_sv_nprm[i, j], fmt) + " +/- " + format(sd_sv_nprm[i, j], fmt) ), ha="center", va="center", color="white" if cm_sv_nprm[i, j] > thresh else "black")

		cm_dl_nprm[0,0] = self.dl_nprm_res_xbar[0]
		cm_dl_nprm[0,1] = self.dl_nprm_res_xbar[1]
		cm_dl_nprm[1,0] = self.dl_nprm_res_xbar[2]
		cm_dl_nprm[1,1] = self.dl_nprm_res_xbar[3]
		
		sd_dl_nprm[0,0] = self.dl_nprm_res_std[0]
		sd_dl_nprm[0,1] = self.dl_nprm_res_std[1]
		sd_dl_nprm[1,0] = self.dl_nprm_res_std[2]
		sd_dl_nprm[1,1] = self.dl_nprm_res_std[3]

		im = ax2.imshow(cm_sv_nprm, interpolation='nearest', cmap=plt.cm.Blues)
		# ax2.figure.colorbar(im, ax=ax2)

		ax2.set(	xticks=np.arange( cm_dl_nprm.shape[1]),
				yticks=np.arange( cm_dl_nprm.shape[0]),
           		xticklabels=self.labels, yticklabels=self.labels,
           		title=("DL: " + self.title),
           		ylabel='True',
           		xlabel='Predicted' ) 

		plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
		fmt = '.1f'
		thresh = cm_sv_nprm.max() / 2.

		for i in range( cm_dl_nprm.shape[0]):
			for j in range(cm_dl_nprm.shape[1]):
				ax2.text(j, i, ( format(cm_dl_nprm[i, j], fmt) + " +/- " + format(sd_dl_nprm[i, j], fmt) ), ha="center", va="center", color="white" if cm_dl_nprm[i, j] > thresh else "black")

		fig.tight_layout()
		plt.show()

		if save_fig:
			filename = "svc_dl_confusion_matrices_" + self.predictor + "_" + self.feature + "." + self.img_format
			fig.savefig( os.path.join( save_dir, filename ), format=self.img_format, dpi=120 )

	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def roc_curves( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		fig, (ax1, ax2) = plt.subplots( 1, 2, figsize=(8,4), dpi=120 )
		ax1.errorbar( self.sv_nprm_roc_fpr_xbar, self.sv_nprm_roc_tpr_xbar, yerr=self.sv_nprm_roc_tpr_std, fmt=".-k", ecolor='red', elinewidth=2, capsize=8);
		ax1.set( title=("SVC: " + self.title), ylabel='True Positive Rate (TPR)', xlabel='False Positive Rate (FPR)' )
		ax1.text( 0.5, 0.15, ( "AUC = " + format( self.sv_nprm_res_xbar[4]*100, '.1f' ) + " +/- " + format( self.sv_nprm_res_std[4]*100, '.1f' ) ), bbox=self.txtbx_properties )

		ax2.errorbar( self.dl_nprm_roc_fpr_xbar, self.dl_nprm_roc_tpr_xbar, yerr=self.dl_nprm_roc_tpr_std, fmt=".-k", ecolor='red', elinewidth=2, capsize=8);
		ax2.set( title=("DL: " + self.title ), ylabel='True Positive Rate (TPR)', xlabel='False Positive Rate (FPR)' )
		ax2.text( 0.5, 0.15, ( "AUC = " + format( self.dl_nprm_res_xbar[4]*100, '.1f' ) + " +/- " + format( self.dl_nprm_res_std[4]*100, '.1f' ) ), bbox=self.txtbx_properties )

		fig.tight_layout()
		plt.show()

		if save_fig:
			filename = "svc_dl_roc_curves_" + self.predictor + "_" + self.feature + "." + self.img_format
			fig.savefig( os.path.join( save_dir, filename ), format=self.img_format, dpi=120 )


	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def score_dists( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		fig, axes = plt.subplots( 3, 2, figsize=(8,10), dpi=120 )
		x = np.linspace( self.sv_nprm_res_xbar[5] - 3*self.sv_nprm_res_std[5], self.sv_nprm_res_xbar[5] + 3*self.sv_nprm_res_std[5], 100)
		axes[0,0].plot( x, stats.norm.pdf(x, self.sv_nprm_res_xbar[5], self.sv_nprm_res_std[5] ), label="SVC" )
		x = np.linspace( self.dl_nprm_res_xbar[5] - 3*self.dl_nprm_res_std[5], self.dl_nprm_res_xbar[5] + 3*self.dl_nprm_res_std[5], 100)
		axes[0,0].plot( x, stats.norm.pdf(x, self.dl_nprm_res_xbar[5], self.dl_nprm_res_std[5] ), label="DL" )
		axes[0,0].set( title=self.title, ylabel='Probability Density', xlabel='Positive Predictive Value (PPV)' )
		axes[0,0].legend()
		
		x = np.linspace( self.sv_nprm_res_xbar[6] - 3*self.sv_nprm_res_std[6], self.sv_nprm_res_xbar[6] + 3*self.sv_nprm_res_std[6], 100)
		axes[0,1].plot( x, stats.norm.pdf(x, self.sv_nprm_res_xbar[6], self.sv_nprm_res_std[6] ), label="SVC" )
		x = np.linspace( self.dl_nprm_res_xbar[6] - 3*self.dl_nprm_res_std[6], self.dl_nprm_res_xbar[6] + 3*self.dl_nprm_res_std[6], 100)
		axes[0,1].plot( x, stats.norm.pdf(x, self.dl_nprm_res_xbar[6], self.dl_nprm_res_std[6] ), label="DL" )
		axes[0,1].set( title=self.title, ylabel='Probability Density', xlabel='Negative Predictive Value (NPV)' )
		axes[0,1].legend()

		x = np.linspace( self.sv_nprm_res_xbar[7] - 3*self.sv_nprm_res_std[7], self.sv_nprm_res_xbar[7] + 3*self.sv_nprm_res_std[7], 100)
		axes[1,0].plot( x, stats.norm.pdf(x, self.sv_nprm_res_xbar[7], self.sv_nprm_res_std[7] ), label="SVC" )
		x = np.linspace( self.dl_nprm_res_xbar[7] - 3*self.dl_nprm_res_std[7], self.dl_nprm_res_xbar[7] + 3*self.dl_nprm_res_std[7], 100)
		axes[1,0].plot( x, stats.norm.pdf(x, self.dl_nprm_res_xbar[7], self.dl_nprm_res_std[7] ), label="DL" )
		axes[1,0].set( title=self.title, ylabel='Probability Density', xlabel='Accuracy (ACC)' )
		axes[1,0].legend()

		x = np.linspace( self.sv_nprm_res_xbar[8] - 3*self.sv_nprm_res_std[8], self.sv_nprm_res_xbar[8] + 3*self.sv_nprm_res_std[8], 100)
		axes[1,1].plot( x, stats.norm.pdf(x, self.sv_nprm_res_xbar[8], self.sv_nprm_res_std[8] ), label="SVC" )
		x = np.linspace( self.dl_nprm_res_xbar[8] - 3*self.dl_nprm_res_std[8], self.dl_nprm_res_xbar[8] + 3*self.dl_nprm_res_std[8], 100)
		axes[1,1].plot( x, stats.norm.pdf(x, self.dl_nprm_res_xbar[8], self.dl_nprm_res_std[8] ), label="DL" )
		axes[1,1].set( title=self.title, ylabel='Probability Density', xlabel='Sensitivity (SEN)' )
		axes[1,1].legend()

		x = np.linspace( self.sv_nprm_res_xbar[9] - 3*self.sv_nprm_res_std[9], self.sv_nprm_res_xbar[9] + 3*self.sv_nprm_res_std[9], 100)
		axes[2,0].plot( x, stats.norm.pdf(x, self.sv_nprm_res_xbar[9], self.sv_nprm_res_std[9] ), label="SVC" )
		x = np.linspace( self.dl_nprm_res_xbar[9] - 3*self.dl_nprm_res_std[9], self.dl_nprm_res_xbar[9] + 3*self.dl_nprm_res_std[9], 100)
		axes[2,0].plot( x, stats.norm.pdf(x, self.dl_nprm_res_xbar[9], self.dl_nprm_res_std[9] ), label="DL" )
		axes[2,0].set( title=self.title, ylabel='Probability Density', xlabel='Specitivity (SPC)' )
		axes[2,0].legend()

		fig.delaxes( axes[2,1] )
		
		fig.tight_layout( pad=1.5, w_pad=1.5, h_pad=1.5 )
		plt.show()

		if save_fig:
			filename = "svc_dl_score_dists_" + self.predictor + "_" + self.feature + "." + self.img_format
			fig.savefig( os.path.join( save_dir, filename ), format=self.img_format, dpi=120 )


	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def sv_prm_dists( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		fig, axes = plt.subplots( 3, 2, figsize=(8,10), dpi=120 )
		x = np.linspace( self.sv_prm_res_xbar[5] - 3*self.sv_prm_res_std[5], self.sv_prm_res_xbar[5] + 3*self.sv_prm_res_std[5], 100)
		yp = stats.norm.pdf(x, self.sv_prm_res_xbar[5], self.sv_prm_res_std[5] )
		axes[0,0].plot( x, yp, linestyle='-', label="SVC (Permutated Labels)" )
		x = np.linspace( self.sv_nprm_res_xbar[5] - 3*self.sv_nprm_res_std[5], self.sv_nprm_res_xbar[5] + 3*self.sv_nprm_res_std[5], 100)
		yn = stats.norm.pdf(x, self.sv_nprm_res_xbar[5], self.sv_nprm_res_std[5] ) 
		axes[0,0].plot( x, yn, linestyle='dotted', label="SVC (Correct Labels)" )
		axes[0,0].vlines( self.sv_nprm_res_xbar[5], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[0,0].set( title=self.title, ylabel='Probability Density', xlabel='Positive Predictive Value (PPV)' )
		axes[0,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.sv_prm_res_xbar[5], std1=self.sv_prm_res_std[5], nobs1=2, mean2=self.sv_nprm_res_xbar[5], std2=self.sv_nprm_res_std[5], nobs2=2, equal_var=False) #( yp, yn ) #equal_var=False )
		axes[0,0].text( (self.sv_prm_res_xbar[5]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.sv_prm_res_xbar[6] - 3*self.sv_prm_res_std[6], self.sv_prm_res_xbar[6] + 3*self.sv_prm_res_std[6], 100)
		yp = stats.norm.pdf(x, self.sv_prm_res_xbar[6], self.sv_prm_res_std[6] )
		axes[0,1].plot( x, yp, linestyle='-', label="SVC (Permutated Labels)" )
		x = np.linspace( self.sv_nprm_res_xbar[6] - 3*self.sv_nprm_res_std[6], self.sv_nprm_res_xbar[6] + 3*self.sv_nprm_res_std[6], 100)
		yn = stats.norm.pdf(x, self.sv_nprm_res_xbar[6], self.sv_nprm_res_std[6] )
		axes[0,1].plot( x, yn, linestyle='dotted', label="SVC (Correct Labels)" )
		axes[0,1].vlines( self.sv_nprm_res_xbar[6], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[0,1].set( title=self.title, ylabel='Probability Density', xlabel='Negative Predictive Value (NPV)' )
		axes[0,1].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.sv_prm_res_xbar[6], std1=self.sv_prm_res_std[6], nobs1=2, mean2=self.sv_nprm_res_xbar[6], std2=self.sv_nprm_res_std[6], nobs2=2, equal_var=False)
		axes[0,1].text( (self.sv_prm_res_xbar[6]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.sv_prm_res_xbar[7] - 3*self.sv_prm_res_std[7], self.sv_prm_res_xbar[7] + 3*self.sv_prm_res_std[7], 100)
		yp = stats.norm.pdf(x, self.sv_prm_res_xbar[7], self.sv_prm_res_std[7] )
		axes[1,0].plot( x, yp, linestyle='-', label="SVC (Permutated Labels)" )
		x = np.linspace( self.sv_nprm_res_xbar[7] - 3*self.sv_nprm_res_std[7], self.sv_nprm_res_xbar[7] + 3*self.sv_nprm_res_std[7], 100)
		yn = stats.norm.pdf(x, self.sv_nprm_res_xbar[7], self.sv_nprm_res_std[7] )
		axes[1,0].plot( x, yn, linestyle='dotted', label="SVC (Correct Labels)" )
		axes[1,0].vlines( self.sv_nprm_res_xbar[7], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[1,0].set( title=self.title, ylabel='Probability Density', xlabel='Accuracy (ACC)' )
		axes[1,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.sv_prm_res_xbar[7], std1=self.sv_prm_res_std[7], nobs1=2, mean2=self.sv_nprm_res_xbar[7], std2=self.sv_nprm_res_std[7], nobs2=2, equal_var=False)
		axes[1,0].text( (self.sv_prm_res_xbar[7]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.sv_prm_res_xbar[8] - 3*self.sv_prm_res_std[8], self.sv_prm_res_xbar[8] + 3*self.sv_prm_res_std[8], 100)
		yp = stats.norm.pdf(x, self.sv_prm_res_xbar[8], self.sv_prm_res_std[8] )
		axes[1,1].plot( x, yp, linestyle='-', label="SVC (Permutated Labels)" )
		x = np.linspace( self.sv_nprm_res_xbar[8] - 3*self.sv_nprm_res_std[8], self.sv_nprm_res_xbar[8] + 3*self.sv_nprm_res_std[8], 100)
		yn = stats.norm.pdf(x, self.sv_nprm_res_xbar[8], self.sv_nprm_res_std[8] )
		axes[1,1].plot( x, yn, linestyle='dotted', label="SVC (Correct Labels)" )
		axes[1,1].vlines( self.sv_nprm_res_xbar[8], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[1,1].set( title=self.title, ylabel='Probability Density', xlabel='Sensitivity (SEN)' )
		axes[1,1].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.sv_prm_res_xbar[8], std1=self.sv_prm_res_std[8], nobs1=2, mean2=self.sv_nprm_res_xbar[8], std2=self.sv_nprm_res_std[8], nobs2=2, equal_var=False)
		axes[1,1].text( (self.sv_prm_res_xbar[8]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.sv_prm_res_xbar[9] - 3*self.sv_prm_res_std[9], self.sv_prm_res_xbar[9] + 3*self.sv_prm_res_std[9], 100)
		yp = stats.norm.pdf(x, self.sv_prm_res_xbar[9], self.sv_prm_res_std[9] )
		axes[2,0].plot( x, yp, linestyle='-', label="SVC (Permutated Labels)" )
		x = np.linspace( self.sv_nprm_res_xbar[9] - 3*self.sv_nprm_res_std[9], self.sv_nprm_res_xbar[9] + 3*self.sv_nprm_res_std[9], 100)
		yn = stats.norm.pdf(x, self.sv_nprm_res_xbar[9], self.sv_nprm_res_std[9] )
		axes[2,0].plot( x, yn, linestyle='dotted', label="SVC (Correct Labels)" )
		axes[2,0].vlines( self.sv_nprm_res_xbar[9], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[2,0].set( title=self.title, ylabel='Probability Density', xlabel='Specitivity (SPC)' )
		axes[2,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.sv_prm_res_xbar[9], std1=self.sv_prm_res_std[9], nobs1=2, mean2=self.sv_nprm_res_xbar[9], std2=self.sv_nprm_res_std[9], nobs2=2, equal_var=False)
		axes[2,0].text( (self.sv_prm_res_xbar[9]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		fig.delaxes( axes[2,1] )
		
		fig.tight_layout( pad=1.5, w_pad=1.5, h_pad=1.5 )
		plt.show()

		if save_fig:
			filename = "svc_prm_dists_" + self.predictor + "_" + self.feature + "." + self.img_format
			fig.savefig( os.path.join( save_dir, filename ), format=self.img_format, dpi=120 )
			

	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def dl_prm_dists( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		fig, axes = plt.subplots( 3, 2, figsize=(8,10), dpi=120 )
		x = np.linspace( self.dl_prm_res_xbar[5] - 3*self.dl_prm_res_std[5], self.dl_prm_res_xbar[5] + 3*self.dl_prm_res_std[5], 100)
		yp = stats.norm.pdf(x, self.dl_prm_res_xbar[5], self.dl_prm_res_std[5] )
		axes[0,0].plot( x, yp, linestyle='-', label="DL (Permutated Labels)" )
		x = np.linspace( self.dl_nprm_res_xbar[5] - 3*self.dl_nprm_res_std[5], self.dl_nprm_res_xbar[5] + 3*self.dl_nprm_res_std[5], 100)
		yn = stats.norm.pdf(x, self.dl_nprm_res_xbar[5], self.dl_nprm_res_std[5] ) 
		axes[0,0].plot( x, yn, linestyle='dotted', label="DL (Correct Labels)" )
		axes[0,0].vlines( self.dl_nprm_res_xbar[5], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[0,0].set( title=self.title, ylabel='Probability Density', xlabel='Positive Predictive Value (PPV)' )
		axes[0,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.dl_prm_res_xbar[5], std1=self.dl_prm_res_std[5], nobs1=2, mean2=self.dl_nprm_res_xbar[5], std2=self.dl_nprm_res_std[5], nobs2=2, equal_var=False)
		axes[0,0].text( (self.dl_prm_res_xbar[5]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )
		
		x = np.linspace( self.dl_prm_res_xbar[6] - 3*self.dl_prm_res_std[6], self.dl_prm_res_xbar[6] + 3*self.dl_prm_res_std[6], 100)
		yp = stats.norm.pdf(x, self.dl_prm_res_xbar[6], self.dl_prm_res_std[6] )
		axes[0,1].plot( x, yp, linestyle='-', label="DL (Permutated Labels)" )
		x = np.linspace( self.dl_nprm_res_xbar[6] - 3*self.dl_nprm_res_std[6], self.dl_nprm_res_xbar[6] + 3*self.dl_nprm_res_std[6], 100)
		yn = stats.norm.pdf(x, self.dl_nprm_res_xbar[6], self.dl_nprm_res_std[6] )
		axes[0,1].plot( x, yn, linestyle='dotted', label="DL (Correct Labels)" )
		axes[0,1].vlines( self.dl_nprm_res_xbar[6], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[0,1].set( title=self.title, ylabel='Probability Density', xlabel='Negative Predictive Value (NPV)' )
		axes[0,1].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.dl_prm_res_xbar[6], std1=self.dl_prm_res_std[6], nobs1=2, mean2=self.dl_nprm_res_xbar[6], std2=self.dl_nprm_res_std[6], nobs2=2, equal_var=False)
		axes[0,1].text( (self.dl_prm_res_xbar[6]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.dl_prm_res_xbar[7] - 3*self.dl_prm_res_std[7], self.dl_prm_res_xbar[7] + 3*self.dl_prm_res_std[7], 100)
		yp = stats.norm.pdf(x, self.dl_prm_res_xbar[7], self.dl_prm_res_std[7] )
		axes[1,0].plot( x, yp, linestyle='-', label="DL (Permutated Labels)" )
		x = np.linspace( self.dl_nprm_res_xbar[7] - 3*self.dl_nprm_res_std[7], self.dl_nprm_res_xbar[7] + 3*self.dl_nprm_res_std[7], 100)
		yn = stats.norm.pdf(x, self.dl_nprm_res_xbar[7], self.dl_nprm_res_std[7] )
		axes[1,0].plot( x, yn, linestyle='dotted', label="DL (Correct Labels)" )
		axes[1,0].vlines( self.dl_nprm_res_xbar[7], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[1,0].set( title=self.title, ylabel='Probability Density', xlabel='Accuracy (ACC)' )
		axes[1,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.dl_prm_res_xbar[7], std1=self.dl_prm_res_std[7], nobs1=2, mean2=self.dl_nprm_res_xbar[7], std2=self.dl_nprm_res_std[7], nobs2=2, equal_var=False)
		axes[1,0].text( (self.dl_prm_res_xbar[7]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.dl_prm_res_xbar[8] - 3*self.dl_prm_res_std[8], self.dl_prm_res_xbar[8] + 3*self.dl_prm_res_std[8], 100)
		yp = stats.norm.pdf(x, self.dl_prm_res_xbar[8], self.dl_prm_res_std[8] )
		axes[1,1].plot( x, yp, linestyle='-', label="DL (Permutated Labels)" )
		x = np.linspace( self.dl_nprm_res_xbar[8] - 3*self.dl_nprm_res_std[8], self.dl_nprm_res_xbar[8] + 3*self.dl_nprm_res_std[8], 100)
		yn = stats.norm.pdf(x, self.dl_nprm_res_xbar[8], self.dl_nprm_res_std[8] )
		axes[1,1].plot( x, yn, linestyle='dotted', label="DL (Correct Labels)" )
		axes[1,1].vlines( self.dl_nprm_res_xbar[8], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[1,1].set( title=self.title, ylabel='Probability Density', xlabel='Sensitivity (SEN)' )
		axes[1,1].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.dl_prm_res_xbar[8], std1=self.dl_prm_res_std[8], nobs1=2, mean2=self.dl_nprm_res_xbar[8], std2=self.dl_nprm_res_std[8], nobs2=2, equal_var=False)
		axes[1,1].text( (self.dl_prm_res_xbar[8]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		x = np.linspace( self.dl_prm_res_xbar[9] - 3*self.dl_prm_res_std[9], self.dl_prm_res_xbar[9] + 3*self.dl_prm_res_std[9], 100)
		yp = stats.norm.pdf(x, self.dl_prm_res_xbar[9], self.dl_prm_res_std[9] )
		axes[2,0].plot( x, yp, linestyle='-', label="DL (Permutated Labels)" )
		x = np.linspace( self.dl_nprm_res_xbar[9] - 3*self.dl_nprm_res_std[9], self.dl_nprm_res_xbar[9] + 3*self.dl_nprm_res_std[9], 100)
		yn = stats.norm.pdf(x, self.dl_nprm_res_xbar[9], self.dl_nprm_res_std[9] )
		axes[2,0].plot( x, yn, linestyle='dotted', label="DL (Correct Labels)" )
		axes[2,0].vlines( self.dl_nprm_res_xbar[9], ymin=np.min(yn), ymax=np.max(yn), color="#ff7f0e" )
		axes[2,0].set( title=self.title, ylabel='Probability Density', xlabel='Specitivity (SPC)' )
		axes[2,0].legend( loc=8, ncol=2, fontsize=7, fancybox=True, shadow=True )
		t,p=stats.ttest_ind_from_stats( mean1=self.dl_prm_res_xbar[9], std1=self.dl_prm_res_std[9], nobs1=2, mean2=self.dl_nprm_res_xbar[9], std2=self.dl_nprm_res_std[9], nobs2=2, equal_var=False)
		axes[2,0].text( (self.dl_prm_res_xbar[9]), (0.90*np.max(yp)), ( "p={0:.5f}".format( p ) ), bbox=self.txtbx_properties )

		fig.delaxes( axes[2,1] )
		
		fig.tight_layout( pad=1.5, w_pad=1.5, h_pad=1.5 )
		plt.show()

		if save_fig:
			filename = "dl_prm_dists_" + self.predictor + "_" + self.feature + "." + self.img_format
			fig.savefig( os.path.join( save_dir, filename ), format=self.img_format, dpi=120 )

	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def model_stat_to_csv( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		filename = "sv_dl_" + self.predictor + "_" + self.feature + ".csv"

		data = dict()
		data['sv_nprm_xbar'] = self.sv_nprm_res_xbar
		data['sv_nprm_std'] = self.sv_nprm_res_std
		data['sv_prm_xbar'] = self.sv_prm_res_xbar
		data['sv_prm_std'] = self.sv_prm_res_std
		data['dl_nprm_xbar'] = self.dl_nprm_res_xbar
		data['dl_nprm_std'] = self.dl_nprm_res_std
		data['dl_prm_xbar'] = self.dl_prm_res_xbar
		data['dl_prm_std'] = self.dl_prm_res_std

		df = pd.DataFrame( data )

		if save_fig:
			df.to_csv( os.path.join( save_dir, filename ) )

	# -----------------------------------------------------------
	#
	#
	#
	# -----------------------------------------------------------
	def predictor_weights( self, save_fig=False, save_dir="./" ):

		if save_fig:
			if not os.path.exists( save_dir ):
				print( "figure directory does not exist, making one [{0:s}]".format( save_dir ) )
				os.makedirs( save_dir )

		top_num_variables = 5
		filename = "sv_dl_predictor_weights_" + self.predictor + "_" + self.feature + ".csv"

		data = dict()
		data['regions'] = self.features
		data['sv_weights'] = self.sv_nprm_weight_mtx
		data['dl_weights'] = self.dl_nprm_weight_mtx
		df = pd.DataFrame( data )

		print( "Showing top {0:d} predictor variables. Entire set available in [{1:s}]".format( top_num_variables, os.path.join( save_dir, filename ) ) ) 

		idx = np.argsort( self.sv_nprm_weight_mtx * -1 )

		print("\n===========================================")
		print("SVC: {0:s}".format( self.title ) )
		print("===========================================")
		for i in range( 0, top_num_variables ):
			print( "{0:s}\t\t(w={1:.1f})".format( self.features[idx[i]], self.sv_nprm_weight_mtx[idx[i]] ) )

		idx = np.argsort( self.dl_nprm_weight_mtx * -1 )

		print("\n===========================================")
		print("DL: {0:s}".format( self.title ) )
		print("===========================================")
		for i in range( 0, top_num_variables ):
			print( "{0:s}\t\t(w={1:.1f})".format( self.features[idx[i]], self.dl_nprm_weight_mtx[idx[i]] ) )

		if save_fig:
			df.to_csv( os.path.join( save_dir, filename ) )
			# fig.savefig( os.path.join( save_dir, filename ), format="svg", dpi=120 )

