########################################################################################
#
# Script that combines AI_train and AI_analyze in multiple rounds to generate table
# data for the article. Not part of the main software package.
#
########################################################################################

# Python 3.7.3.


import os, sys, scipy.io, scipy.linalg, random
from time import time

# Suppress warnings from tensorflow; only display errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Remaining dependencies
import numpy as np, matplotlib.pylab as plt


###
# In CONFIG
# -- paths
# -- balance
# -- PCA (how many components)
# -- number cohomology mats
# -- max-data-size : Read files until file sizes exceeds max-data-size
# -- output : Saved models
# -- output : Saved predictions
# -- hyperparameter config.
from NNCONFIG import *
from util import *
from data_handling import *
from model_bundle import *
from AI_functions import *

#**************************************************
# Functions. TODO: create a separate module, or move to AI_functions.

def save_training_info(NN_PATH, BM, INPUT_DIR, NumMats, random_seed,
                       MAX_INPUT_DATA_SIZE, ttratio, sampler, train_y):
    
    BM.save(os.path.join(NN_PATH, 'SavedModels/',''), also_to_newest=False)
    reference_network = str(BM.base_network_name())


    # Parameters that the network hasn't yet remembered about itself.
    setup_params = {"Num cohomology matrices / pair" : NumMats,
                    "Total time elapsed" : 0,
                    "Random seed" : random_seed,
                    "Training set filename" : '"{}"'.format(INPUT_DIR)}

    model_save_path = os.path.join(NN_PATH, "SavedModels", '')
    BM.save_parameters(model_save_path, setup_dic=setup_params,
                       params_dic=network_architecture_hyperparameters, also_to_newest=False)


    ## Save the information about the training set.

    success_percent_str = "{} / {}".format(np.sum(train_y), train_y.shape[0])

    data_info = {"MAX_INPUT_DATA_SIZE" : MAX_INPUT_DATA_SIZE,
                 "Train/test ratio" : ttratio,
                 "Random seed" : random_seed,
                 "Sampler name" : sampler.__name__,
                 "Percentage successes in training set" : success_percent_str,
                 "Training set filename" : '"{}"'.format(INPUT_DIR)}

    BM.save_training_data_info(model_save_path, data_info)
    return
##

def WriteTrainingConfusionStats(ModelNum, pCN, pNN, pEN, test_y,
                                print_matrices=True, write_table9=False):
    """
    Write the confusion matrices using the optimal threshold for the ensemble network.
    (Note: we **do not** also write the confusion matrices using the individually optimal thresholds.)
    
    If print_matrices=True, also print **both** types of confusion matrix to standard out.
    """

    # Determine optimal thresholds.
    opt_th_pNN = OptimalROCSup(pNN, test_y, "NN")
    opt_th_pCN = OptimalROCSup(pCN, test_y, "CN")
    opt_th_pEN = OptimalROCSup(pEN, test_y, "EN")

    # Threshold.
    yNN = ThresholdProbs(pNN, opt_th_pNN)
    yCN = ThresholdProbs(pCN, opt_th_pCN)
    yEN = yCN*yNN
    yEN_opt_th_pEN = ThresholdProbs(pEN, opt_th_pEN) #not equivalent to yEN = yCN*yNN.


    # TODO: 'params' are kept for legacy, but ultimately serve no purpose. Should be removed.
    paramsNN, paramsCN = [ModelNum], [ModelNum]

    argsin = [NN_PATH, ModelNum, paramsNN, paramsCN, test_y, yNN, yCN, yEN]
    WriteConfusion(*argsin)

    if print_matrices:
        print("Optimal thresholds:")
        print("opt_th_pNN:", opt_th_pNN)
        print("opt_th_pCN:", opt_th_pCN)
        print("opt_th_pEN:", opt_th_pEN)

        PrintConfusion(test_y, yNN, yCN, yEN) #prints confusion matrices per filter
        PrintConfusion(test_y, yNN, yCN, yEN_opt_th_pEN) #prints confusion matrices per filter

    
    if write_table9:
        util._WriteTable9Data(os.path.join(NN_PATH, 'EvalOutputs', 'table9data.txt'),
                              MB, ttratio, confusion_matrix(test_y, yEN))

    return
##

# Imports to print table9
from sklearn.metrics import confusion_matrix
import util


###################################################
# Main script.

# 
# NOTE: We are aiming for a 'large lower right' and 'very small upper right'
#       of the confusion matrices for the trained network.
#
# Terminology: EN -- "Ensemble of networks"

####################

######
## Data setup.

sampler = RandomSampler if dataStream==1 else RandomBalancedSampler
print("Using sampler: ", sampler.__name__)

# Read and process the data.
train_data = ReadDataAndFormat(INPUT_DIR, dataShape, NumMats, "training", ttratio,
                               Sampler=sampler, verbose=False)

test_data = ReadDataAndFormat(INPUT_DIR, dataShape, NumMats, "testing", ttratio,
                             Sampler=sampler, verbose=False)

train_x, train_y, train_M = train_data
test_x, test_y, test_M = test_data

## Display training data stats
if train_y is None:
    error_msg = "Data in input directory is unlabelled. Directory: {}".format(INPUT_DIR)
    raise RuntimeError(error_msg)

print("\n\n# successes in original training set: ",
      np.sum(train_y), " / ", train_y.shape[0],
      " total training samples.")


######
## Training/Testing loop

for dummy_var in range(5):

    ######
    ## Run training protocol.

    # ** Actually training/loading the network.
    if not FineTuneInTraining:
        BM, paramsNN, paramsCN = train_model_bundle(train_data, NumMats,
                                                    **network_architecture_hyperparameters)
    else:
        old_model_bundle = fetch_model(NN_PATH, ReadNewest, UseModel)
        BM = finetune_bundle(old_model_bundle, train_data, **finetune_hyperparameters)


    ## Save the training run info.
    save_training_info(NN_PATH=NN_PATH, BM=BM, INPUT_DIR=INPUT_DIR, NumMats=NumMats,
                       random_seed=random_seed, MAX_INPUT_DATA_SIZE=MAX_INPUT_DATA_SIZE,
                       ttratio=ttratio, sampler=sampler, train_y=train_y)

    ######
    ## Run testing protocol.

    # Load file containing pretrained networks and evaluate on testing data.
    MB = fetch_model(NN_PATH, ReadNewest, UseModel)
    pCN, rCN, pNN, rNN, pEN, rEN = MB.evaluate_models(test_data)

    ModelNum = MB.name()
    print("   Using trained model:    ", ModelNum, "\n\n***")

    WriteTrainingConfusionStats(ModelNum, pCN, pNN, pEN, test_y,
                                print_matrices=True, write_table9=False)

