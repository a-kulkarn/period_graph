import os, sys, scipy.io, scipy.linalg, time, random, pickle
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
from util import *
from AI_functions import *

#**************************************************
# Main script.

start_time = time()

sampler = BasicSampler if dataStream==1 else RandomBalancedSampler
print("Using sampler: ", sampler)

# Read and process the data.
data_gruppe, is_data_labelled = ReadData(INPUT_DIR, None, dataShape, Sampler=sampler, verbose=False)
data = ReformatData(data_gruppe, is_data_labelled, NumMats)
train_x,train_y,train_M = data

print("\n\n# successes in original training set: ",
      np.sum(train_y)," / ",train_y.shape[0],
      " total training samples.")

old_model_id = '_newest' if ReadNewest else OldModel
old_model_bundle = load_model_bundle(os.path.join(NN_PATH, 'SavedModels', ''), old_model_id)

BM = finetune_bundle(old_model_bundle, BatchSize, EpochNum, data, Balancing=Balancing)

### SAVE MODEL ITSELF TO FILE
network_name = BM.model_id
BM.save(os.path.join(NN_PATH, 'SavedModels/',''), also_to_newest=True)    
paramsNN,paramsCN   = [OldModel],[OldModel]

elapsed_time = time() - start_time



#**************************************************
### WRITE MODEL PARAMETERS TO FILE

reference_network = old_model_id
paramsNN,paramsCN = [OldModel],[OldModel]

# Note: We are aiming for a 'large lower right' & 'very small upper right' of the confusion matrices.
ParamsSetup = [IsNewData, network_name, not FineTuneInTraining, NumMats,Balancing,DoPCA,
               INPUT_DIR,elapsed_time, reference_network, random_seed]

WriteParameters(os.path.join(NN_PATH, "SavedModels"), network_name, ParamsSetup)
WriteParameters(os.path.join(NN_PATH, "SavedModels"), "_newest", ParamsSetup)
