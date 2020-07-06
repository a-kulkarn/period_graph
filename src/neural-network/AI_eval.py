# Python 3.7.3.
## THIS FILE saves only to TestingOutputs

import os, sys, scipy.io, scipy.linalg, random, numpy as np
from time import time

###
# In CONFIG
# -- paths
# -- balance
# -- PCA (how many components
# -- number cohomology mats
# -- max-data-size : Read files until file sizes exceeds max-data-size
# -- output : Saved models
# -- output : Saved predictions
# -- hyperparameter config.
from NNCONFIG import *

#######
# Keras import
# We need to do sketchy path stuff when called from sage.

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # don't display warnings; only errors
try:
    from keras.models import load_model
except ModuleNotFoundError:
    sys.path.insert(0, PYTHON3_LOCAL_SITE_PKG)
    from keras.models import load_model

from util import *
from model_bundle import *
from AI_functions import *
from data_handling import *

#**************************************************
# Setup input parameters.

# File containing pretrained networks.
# ModelNum = '_newest' if ReadNewest else UseModel


#**************************************************
# Read in evaluation data.

fnames = sorted(list(dataShape.keys()))
sampler = BasicSampler

data_set = DataSet(SAGE_INPUT_DIR, dataShape)

data_gruppe, is_data_labelled = data_set.read_all()
test_all             = ReformatData(data_gruppe,is_data_labelled, NumMats)
test_x,test_y,test_M = test_all # test_y is 'None' if the data is unlabelled.



#**************************************************
# load and evaluate models.

MB = fetch_model(NN_PATH, ReadNewest, UseModel)
dataout = MB.evaluate_models(test_all)

WritePredictionsToSagePipe(NN_PATH, dataout)
