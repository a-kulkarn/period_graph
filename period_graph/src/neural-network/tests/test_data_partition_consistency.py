
##############################################################################################
#
# Test for data partition consistency.
#
##############################################################################################
#
# This tests will ONLY work on a particular developer machine ('doob', on the Dartmouth math
# department cluster). 
#
# This script needs to be run in the /neural_network/ directory.
#
# IMPORTANT NOTE: The RandomSampler object uses its own **hard-coded** random seed, in order
#                 To force consistency between the data-sets, in case the train/analyze setup
#                 is run at completely different times. Thus, the 'random_seed' parameter
#                 really does nothing in effect. This design fault might be corrected in a
#                 future iteration.
#
##############################################################################################
# Imports

import os, sys, scipy.io, scipy.linalg, random
from time import time
import numpy

# Adjust the path to find the config files
NN_PATH = "/home/akulkarn/period_graph/period_graph/src/neural-network/"
sys.path.insert(1, NN_PATH)

##############################################################################################
#
# Testing config setup. (Selectively change the config file variables for the test.)

from NNCONFIG import *

INPUT_DIR = "/home/akulkarn/Gauss-Manin-data"

# Data management parameters.
MAX_INPUT_DATA_SIZE = "1MB"
ttratio = 0.3
dataShape = {"edgesX-*.csv":2*35+1, "timingsY-*.csv":3+1, "DCM01-*.csv":21**2+1, "DCM10-*.csv":21**2+1}
NumMats = 2
random_seed = 132456789


##############################################################################################

# Secondary imports
from util import *
from data_handling import *
from model_bundle import *
from AI_functions import *

# Force this inside the submodule
import data_handling
data_handling.MAX_INPUT_DATA_SIZE = "1MB"


##############################################################################################


## First, check that numpy.random seed reset is consistent.

numpy.random.seed(random_seed)
first_rand = [numpy.random.rand() for i in range(100)]

numpy.random.seed(random_seed)
second_rand = [numpy.random.rand() for i in range(100)]

assert first_rand == second_rand


##############################################################################################

# Start main test. The idea of the test is to detect that running the program twice with
# the same random initial conditions produces the same DataSet. 


data_sets = []

for i in range(2):

    # Set the seed to conduct the main test.
    numpy.random.seed(random_seed)

    
    sampler = RandomSampler
    print("Using sampler: ", sampler.__name__)

    # Read and process the data.
    data = ReadDataAndFormat(INPUT_DIR, dataShape, NumMats, "training", ttratio, Sampler=sampler, verbose=False)
    #data = KH_circumvent(INPUT_DIR, dataShape, NumMats, "training", Sampler=sampler, ttratio, verbose=False)
    train_x, train_y, train_M = data
    print(len(train_y))

    print("\n\n# successes in original training set: ",
          np.sum(train_y)," / ",train_y.shape[0],
          " total training samples.")

    data_sets += [data]


# Compare the outputs.

for ds in data_sets:
    train_x, train_y, train_M = ds
    assert all(numpy.array_equal(a,b) for a,b in zip(data_sets[0], ds))

