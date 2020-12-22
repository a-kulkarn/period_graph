
# Python 3.7.3.

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


# Suppress warnings from tensorflow; only display errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Remaining dependencies
import numpy as np, matplotlib.pylab as plt
from util import *
from model_bundle import *
from data_handling import *


#**************************************************
# Setup input parameters.

np.random.seed(random_seed)

if INPUT_DIR == "/Users/Heal/Dropbox/Research/EAK/4-monomial-complete/":
    sampler = BasicSampler
else:
    sampler = RandomBalancedSampler
print("Using sampler: ", sampler.__name__)


#**************************************************
# Main script.

## read the core-indices that define the train dataset.
#csvfile = NN_PATH+'SavedModels/train_indices'+ModelNum+'.csv'
#indices = np.loadtxt(csvfile)

test_all = ReadDataAndFormat(INPUT_DIR, dataShape, NumMats, "testing", ttratio, Sampler=sampler, verbose=False)
#test_all = KH_circumvent(EVALS_DIR, dataShape, NumMats, "testing", ttratio, Sampler=sampler, verbose=False)
test_x, test_y, test_M = test_all
print(sum(test_y))

#***************************************************
### PRINT, SAVE, AND VISUALIZE RESULTS FOR TEST DATA

# File containing pretrained networks.
ModelNum = '_newest' if ReadNewest else UseModel
MB = fetch_model(NN_PATH, ReadNewest, ModelNum)
pCN, rCN, pNN, rNN, pEN, rEN = MB.evaluate_models(test_all)
ModelNum = MB.name()


print("***\n\nTHESE WILL BE INPUT TO AI_analyze.py:\n")
print("   Using trained model:    ", ModelNum, "\n\n***")


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

# NOTE: We are aiming for a 'large lower right' and 'very small upper right'
#       of the confusion matrices.

for val in ["_newest", ModelNum]:
    argsin = [NN_PATH, val, paramsNN, paramsCN, test_y, yNN, yCN, yEN]
    WriteConfusion(*argsin)

PrintConfusion(test_y, yNN, yCN, yEN) #prints confusion matrices per filter



yEN_opt_th_pEN = ThresholdProbs(pEN, opt_th_pEN) #not equivalent to yEN = yCN*yNN.
PrintConfusion(test_y, yNN, yCN, yEN_opt_th_pEN) #prints confusion matrices per filter


if PlotsOn and sum(test_y)>0 and sum(test_y)<len(test_y):
    WritePlots(NN_PATH,"",pNN,pCN,pEN,test_y)
