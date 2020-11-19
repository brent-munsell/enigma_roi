from enum import Enum
from DenseClassificationModel import DenseClassificationModel
from LassoModel import LassoModel
from SVCModel import SVCModel
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import Constants as constants
from enigma_data import enigma_data
from neurocombat_sklearn import CombatModel
from NeuroCombatModel import NeuroCombatModel


class Model_Factory:
    
    def build( self, model_type=constants.MODELS.DL_CLASSIFY, **sk_params ):
        
        if model_type == constants.MODELS.SV_CLASSIFY:
            model = SVCModel( **sk_params )
            return model
        
        elif model_type == constants.MODELS.DL_CLASSIFY:
            return DenseClassificationModel( **sk_params )

        elif model_type == constants.MODELS.LASSO:
            model = LassoModel()
            return model
        
        elif model_type == constants.MODELS.SCALER:
            return MinMaxScaler()
        
        elif model_type == constants.MODELS.NEUROCOMBAT:
            model = NeuroCombatModel( **sk_params )
            return model
            
        else:
            return SVMSMOTE()
    
