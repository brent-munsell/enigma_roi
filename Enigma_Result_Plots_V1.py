#!/usr/bin/env python
# coding: utf-8

from PyEnigmaPlots import PyEnigmaPlots
import argparse
import os
import shutil

NB_DEBUG = 1

parser = argparse.ArgumentParser()
parser.add_argument( "-sdx", default="3", help="Sdx ( 3, engel )" )
parser.add_argument( "-dir", default="./figs", help="directory name" )
parser.add_argument( "-fea", default="all", help="feature ( all, AD, FA, MD, or RD )" )
parser.add_argument( "-fmt", default="svg", help="image format ( svg, jpg, png, gif )" )

args = parser.parse_args()

sdx = args.sdx
folder = args.dir
feature = args.fea
fmt = args.fmt

# --------------------------------------------------------------------------------
# create the directory (if needed)
# --------------------------------------------------------------------------------

if not os.path.exists( folder ):
    if NB_DEBUG:
        print( "making folder [{0:s}]".format( folder ) )
    os.makedirs( folder )


# feature: can be { FA, RD, MD, AD, or all }
# predictor: can be { 3, 4, 5, 6, 7, 8, or engel }
# note: predictor="engel" does not exists for feature="all"

plots = PyEnigmaPlots( predictor=sdx, save_dir=folder, feature=feature, img_format=fmt )

plots.confusion_matrices( save_fig=True, save_dir=folder )
plots.roc_curves( save_fig=True, save_dir=folder )
plots.score_dists( save_fig=True, save_dir=folder )
plots.sv_prm_dists( save_fig=True, save_dir=folder )
plots.dl_prm_dists( save_fig=True, save_dir=folder )
plots.predictor_weights( save_fig=True, save_dir=folder )
plots.model_stat_to_csv( save_fig=True, save_dir=folder )





