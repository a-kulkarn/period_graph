
# Python 3.7.3.

# SET CURRENT WORKING DIRECTORY.
import os, sys

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

## THIS FILE saves only to TestingOutputs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import os, sys, scipy.io, scipy.linalg, time, random, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #don't display warnings; only errors
import numpy as np, tensorflow as tf, matplotlib.pylab as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from util import *
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from keras.models import load_model
from time import time

from model_bundle import *
from data_handling import *


#**************************************************
# Setup input parameters.

np.random.seed(random_seed)

fnames = sorted(list(dataShape.keys()))

if EVALS_DIR == "/Users/Heal/Dropbox/Research/EAK/4-monomial-complete/":
    sampler = BasicSampler
else:
    sampler = RandomBalancedSampler
print("Using sampler: ", sampler.__name__)

# File containing pretrained networks.
ModelNum = '_newest' if ReadNewest else UseModel

print("***\n\nTHESE WILL BE INPUT TO AI_analyze.py:\n")
print("   Using trained model:    ", ModelNum, "\n\n***")

#**************************************************
# Main script.

## read the core-indices that define the train dataset.
#csvfile = NN_PATH+'SavedModels/train_indices'+ModelNum+'.csv'
#indices = np.loadtxt(csvfile)

test_all = ReadDataAndFormat(EVALS_DIR, dataShape, NumMats, "testing", ttratio, Sampler=sampler, verbose=False)
#test_all = KH_circumvent(EVALS_DIR, dataShape, NumMats, "testing", ttratio, Sampler=sampler, verbose=False)
test_x, test_y, test_M = test_all
print(sum(test_y))

#***************************************************
### PRINT, SAVE, AND VISUALIZE RESULTS FOR TEST DATA

MB = load_model_bundle(NN_PATH+'SavedModels',ModelNum)
pCN, rCN, pNN, rNN, pEN, rEN = MB.evaluate_models(test_all)

PlotsOn = True                 #broken for now TODO: Investigate?
paramsNN    = [ModelNum]
paramsCN    = [ModelNum]

## write the core-indices that define the test dataset.
#csvfile = NN_PATH+'EvalOutputs/test_indices'+ModelNum+'.csv'
#np.savetxt(csvfile, indices_out, delimiter=",")

opt_th_pNN = OptimalROCSup(pNN, test_y, "NN")
opt_th_pCN = OptimalROCSup(pCN, test_y, "CN")
print("opt_th_pNN:", opt_th_pNN)
print("opt_th_pCN:", opt_th_pCN)

yNN = ThresholdProbs(pNN, opt_th_pNN)
yCN = ThresholdProbs(pCN, opt_th_pCN)

opt_th_pEN = OptimalROCSup(pEN, test_y, "EN")
print("opt_th_pEN:", opt_th_pEN)


plt.show()


yEN = yCN*yNN # EN -- "Ensemble of networks"

print(sum(yEN),yEN.shape)

# Note: We are aiming for a 'large lower right' and 'very small upper right' of the confusion matrices.
argsin = [NN_PATH,"",paramsNN,paramsCN,test_y,yNN,yCN,yEN]
WriteConfusion(*argsin)
PrintConfusion(test_y, yNN, yCN, yEN) #prints confusion matrices per filter



yEN_opt_th_pEN = ThresholdProbs(pEN, opt_th_pEN) #not equivalent to yEN = yCN*yNN.
PrintConfusion(test_y, yNN, yCN, yEN_opt_th_pEN) #prints confusion matrices per filter


if PlotsOn and sum(test_y)>0 and sum(test_y)<len(test_y):
    WritePlots(NN_PATH,"",pNN,pCN,pEN,test_y)
