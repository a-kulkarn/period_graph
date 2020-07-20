# Python 3.7.3.

import os, sys, scipy.io, scipy.linalg, random
from time import time

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
# Main script.

sampler = RandomSampler if dataStream==1 else RandomBalancedSampler
print("Using sampler: ", sampler.__name__)

# Read and process the data.
data = ReadDataAndFormat(INPUT_DIR, dataShape, NumMats, "training", ttratio, Sampler=sampler, verbose=False)
#data = KH_circumvent(INPUT_DIR, dataShape, NumMats, "training", Sampler=sampler, ttratio, verbose=False)
train_x, train_y, train_M = data
print(len(train_y))

if train_y is None:
    raise RuntimeError("Data in input directory is unlabelled. Directory: {}".format(INPUT_DIR))
    
print("\n\n# successes in original training set: ",
      np.sum(train_y)," / ",train_y.shape[0],
      " total training samples.")
    

# ** Actually training/loading the network.
if not FineTuneInTraining:
    BM, paramsNN, paramsCN = train_model_bundle(data, NumMats, **network_architecture_hyperparameters)
    
else: #load pre-trained models from computer

    old_model_id = '_newest' if ReadNewest else OldModel
    old_model_bundle = load_model_bundle(os.path.join(NN_PATH, 'SavedModels', ''), old_model_id)

    BM = finetune_bundle(old_model_bundle, data, **finetune_hyperparameters)
    paramsNN,paramsCN   = [OldModel],[OldModel]


#**************************************************
### PRINT, SAVE, AND VISUALIZE RESULTS

## write the core-indices that define the train dataset.
#csvfile = NN_PATH+'SavedModels/train_indices'+BM.name()+'.csv'
#csv_newest = NN_PATH+'SavedModels/train_indices_newest.csv'
#np.savetxt(csvfile, indices_out, delimiter=",")
#np.savetxt(csv_newest, indices_out, delimiter=",")

BM.save(os.path.join(NN_PATH, 'SavedModels/',''), also_to_newest=True)

print("***\n\nTHIS WILL BE INPUT TO AI_analyze.py:\n")
print("   Naming this training: ", BM.name(), "\n\n***")

reference_network = "None" if not FineTuneInTraining else OldModel

# Note: We are aiming for a 'large lower right' & 'very small upper right' of the confusion matrices.

# Parameters that the network hasn't yet remembered about itself.
setup_params = {"Num cohomology matrices / pair" : NumMats,
                "Total time elapsed" : 0,
                "Random seed" : random_seed,
                "Training set filename" : '"{}"'.format(INPUT_DIR)}

BM.save_parameters(os.path.join(NN_PATH, "SavedModels", ''), setup_dic=setup_params,
                   params_dic=network_architecture_hyperparameters, also_to_newest=True)
