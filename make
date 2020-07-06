#!/bin/bash

cd src
currentDir=$(pwd)

cd $(dirname $0)
pathToSrc=$(pwd)
cd ..
pathToRepo=$(pwd)

# Set the path to PeriodSuite
pathToSuite="$pathToSrc/suite"

if ! [ -f $pathToSuite/suite.mag ]
then
   git submodule update --init
fi

# updating PeriodSuite in submodule
git submodule update --remote --rebase
cd $pathToSuite
git checkout master
./make

cd $pathToSrc
mkdir process-status
mkdir edge-data
mkdir vertex-data
mkdir output-files
mkdir failed-edges
mkdir DifferentiateCohomology-failed
mkdir root-quartics
mkdir user_input
mkdir quartics-cache
mkdir archive

# Integration phase
mkdir ode-data
mkdir periods

# Neural network directories
mkdir edge-data-unlabelled
mkdir neural-network/__pycache__
mkdir neural-network/SavedModels
mkdir neural-network/SpecialModels
mkdir neural-network/TrainingOutputs
mkdir neural-network/EvalOutputs

# set directory names in sage
cat > SAGE_CONFIG.py << EOF
SRC_ABS_PATH = "$pathToSrc/"
pathToSuite = "$pathToSuite/"
PHASE_I_ALARM  = 30;
PHASE_II_ALARM = 21*(5*60)
INTEGRATION_ALARM = 2*8*60
DCM_ALARM = 10
DIGIT_PRECISION = 300
EOF

# set directory names in magma
# set alarm times for magma computations
cat > magma/MAGMA_CONFIG << EOF
SRC_ABS_PATH := "$pathToSrc/";
SUITE_FILE := "$pathToSuite/suite.mag";
ONLY_FIRST := false;
EOF

cat > neural-network/NNCONFIG.py << EOF
NN_PATH = "$pathToSrc/neural-network/"
INPUT_DIR = "$pathToSrc/edge-data-unlabelled/"
SAGE_INPUT_DIR = "$pathToSrc/edge-data-unlabelled/"
PYTHON3_LOCAL_SITE_PKG = "$HOME/.local/lib/python3.7/site-packages/"

# Data management parameters.
MAX_INPUT_DATA_SIZE = "100MB"
TRAINING_PERCENTAGE = 0.9
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
    'EpochNum'  : 50,       ## how many epochs to train MLP&CNN networks
    'BatchSize' : 1000,     ## batch size for training MLP&CNN
    'StepSize'  : 1e-3,     ## MLP step size
    'Balancing' : True
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

# fix the name for the interface.sage
cd $pathToRepo

cat > __init__.sage << EOF
# Dumb version of python module structure. Upgrade to pure python
# is a goal in the future release.
SELF_PATH   = "$pathToRepo/"
PYTHON3_BIN = "$(which python3)"
load(SELF_PATH + "interface.sage")
EOF

cat > __init__.py << EOF
from sage.all import *
SELF_PATH   = "$pathToRepo/"
PYTHON3_BIN = "$(which python3)"
load(SELF_PATH + "interface.sage")
EOF


# sed -i.backup "1,2d" interface.sage
# sed -i.backup "1i\\
# PYTHON3_BIN    = \"$(which python3)\"
# " ${pathToRepo}"/interface.sage"
# sed -i.backup "1i\\
# SELF_PATH    = \"${pathToRepo}"/"\"
# " ${pathToRepo}"/interface.sage"

cd $pathToSrc
# Run magma to initialize the Fermat directory
magma -b magma/initialize-fermat.m

cd $currentDir
cd ..
