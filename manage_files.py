#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from datetime import datetime
import numpy as np
class ManageFiles:
    def __init__( self, root_folder ):
        _uuid = str( np.int32( datetime.timestamp( datetime.now() ) ) )
        
        self.root_folder = root_folder
        self.sdx = int(input("sdx :"))
        self.egl = int(input("egl :"))
        self.hmz = int(input("hmz :"))
        self.prm = int(input("prm :"))
        self.grd = int(input("grd :"))
        self.mdl = input("mdl :")
        self.csv_opt = input("csv_opt :")
        
        self.csv_file = self.root_folder + self.csv_opt + ".csv"
        self.predict, self.predict_val = ("ENGEL",1) if self.egl == 1 else ("SDx",self.sdx) 
        
        _archPath = ""
        _gridPath = ""
        
        if ( self.egl == 0 ):
            _gridPath = "_sdx_" + str( self.sdx ) + "_" 
            if ( self.prm == 1 ):
                _archPath = str( self.sdx ) + "/prm"
            else:
                _archPath = str( self.sdx ) + "/nprm"
        else:
            _gridPath = "_engel_"
            if ( self.prm == 1 ):
                _archPath = "engle/prm"
            else:
                _archPath = "engle/nprm"
                
                
        self.grid_result_file = self.root_folder + self.csv_opt.lower() + "_arch/" + self.mdl.lower() + _gridPath + self.csv_opt + ".joblib"  
        self.arch = ( self.root_folder + self.csv_opt.lower() + "_arch/" + self.mdl + "/" + _archPath )
        self.chpt_folder = self.arch + "/chpt_" + _uuid + "/"
        self.arch_file = os.path.join ( self.arch, "it_result_" + _uuid + ".joblib" )
        
        if not os.path.exists( self.arch ) :
               os.makedirs( self.arch )
        

