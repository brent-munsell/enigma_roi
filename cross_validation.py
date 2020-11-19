#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc

class CrossValidation:
    def __init__(self, ttv_dict, cv_result, cv_folds):
        self.iteration = []
        self.ttv_dict = ttv_dict
        self.cv_result = cv_result
        self.cv_folds = cv_folds
       
        for idx in range( 0, self.cv_folds ):

            res = {}
            res["yt"] = self.ttv_dict['Y_test']
            res["yp"] = self.cv_result['estimator'][idx].predict( self.ttv_dict['X_test'] )
            fpr, tpr, _ = roc_curve( res["yt"], res["yp"], pos_label=1 )
            res["auc"] = auc( fpr, tpr )
            tn, fp, fn, tp = confusion_matrix( res["yt"], res["yp"] ).ravel()
            res["tn"] = tn
            res["fp"] = fp
            res["fn"] = fn
            res["tp"] = tp
            res["w"] = self.cv_result['estimator'][idx]['classifier'].get_weights()
            self.cv_result['estimator'][idx]['classifier'].get_weights()       
            self.iteration.append( res )
            print( "CV Fold {0:d} AUC {1:f}".format( idx+1, res["auc"] ) )
        

