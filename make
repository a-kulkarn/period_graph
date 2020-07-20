#!/bin/bash

##############################################
# Initialize Paths
##############################################

currentDir=$(pwd)
pathToRepo=$(pwd)
cd src


cd $(dirname $0)
pathToSrc=$(pwd)

# Set the path to PeriodSuite
pathToSuite="$pathToSrc/suite"

# Set the path to the neural network library
pathToNNLib="$pathToSrc/neural-network/SpecialModels"



##############################################
# Initialize PeriodSuite and other submodules
##############################################

if ! [ -f $pathToSuite/suite.mag ]
then
   git submodule update --init
fi

# Ensure PeriodSuite is on the sage9 branch.
cd $pathToSuite
git checkout sage9
./make



##############################################
# Initialize Data and logging directories
##############################################

cd $pathToRepo
mkdir training-data
mkdir neural_network_input
mkdir process-status
mkdir archive

cd training-data
mkdir edge-data
mkdir failed-edges
mkdir DifferentiateCohomology-failed


cd $pathToSrc
mkdir vertex-data
mkdir root-quartics
mkdir user_input
mkdir quartics-cache


# Integration phase
mkdir ode-data
mkdir periods

# Neural network directories
mkdir edge-data-unlabelled
mkdir neural-network/__pycache__
mkdir neural-network/SavedModels
mkdir neural-network/TrainingOutputs
mkdir neural-network/EvalOutputs



##############################################
# Create config files.
##############################################

cd $pathToSrc


###################
# SAGE_CONFIG     #
###################

# set directory names in sage
cat > SAGE_CONFIG.py << EOF
SELF_PATH = "$pathToRepo/"
SRC_ABS_PATH = "$pathToSrc/"
pathToSuite = "$pathToSuite/"
TRAINING_PATH = "$pathToRepo/training-data/"
PYTHON3_BIN = "$(which python3)"
PHASE_I_ALARM  = 30;
PHASE_II_ALARM = 21*(5*60)
INTEGRATION_ALARM = 2*8*60
DCM_ALARM = 10
DIGIT_PRECISION = 300
EOF


###################
# MAGMA_CONFIG    #
###################

# set directory names in magma
# set alarm times for magma computations
cat > magma/MAGMA_CONFIG << EOF
SRC_ABS_PATH := "$pathToSrc/";
SUITE_FILE := "$pathToSuite/suite.mag";
EOF


###################
# NNCONFIG        #
###################

cat > neural-network/NNCONFIG.py << EOF
NN_PATH = "$pathToSrc/neural-network/"
INPUT_DIR = "$pathToRepo/training-data/"
SAGE_INPUT_DIR = "$pathToRepo/neural_network_input/"
PYTHON3_LOCAL_SITE_PKG = "$HOME/.local/lib/python3.7/site-packages/"

# Data management parameters.
MAX_INPUT_DATA_SIZE = "100MB"
ttratio = 0.9
dataShape = {"edgesX-*.csv":2*35+1, "timingsY-*.csv":3+1, "DCM01-*.csv":21**2+1, "DCM10-*.csv":21**2+1}
NumMats = 2
random_seed = 132456789

# Model selection parameters.
FineTuneInTraining = False
ReadNewest = False
OldModel = 'Bly'          # for train_AI. what pretrained network to fine-tune.
UseModel = OldModel       # what model to evaluate and analyze.
UseEvalu = OldModel       # what evaluation to analyze.


#######################################
# Network architecture hyperparameters.

network_architecture_hyperparameters = {
    'DoPCA' : True,
    'PCAk'  : 23,    
    'EpochNum'    : 50,       ## how many epochs to train MLP&CNN networks
    'BatchSize'   : 1000,     ## batch size for training MLP&CNN
    'StepSizeMLP' : 1e-3,     ## MLP step size
    'StepSizeCNN' : 1e-3,     ## CNN step size
    'Balancing'   : True
}

finetune_hyperparameters = {
    'EpochNum'  : 100,      ## how many epochs to train MLP&CNN networks
    'BatchSize' : 1000,     ## batch size for training MLP&CNN
    'Balancing' : True
}


#######################################
# Legacy parameters (for old branches).

dataStream = 1 # Sampler selector.
DoPCA = True

EOF


##############################################
# Initialize suite's Fermat directory
##############################################

cd $pathToSrc
# Run magma to initialize the Fermat directory
magma -b magma/initialize-fermat.m

##############################################
# Cleanup
##############################################

cd $currentDir
