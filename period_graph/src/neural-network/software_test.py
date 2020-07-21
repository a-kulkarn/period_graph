#
# This file tests basic usage of training and evaluation.
# NOTE: The training tests must be run beforehand to generate testing data.
#
# NOTE: This test *must* be run in the current directory with python3.
#

import os, subprocess

assert subprocess.call(["python3", "AI_train.py"],
                       cwd=os.path.join(SRC_ABS_PATH, "neural-network",''))

assert subprocess.call(["python3", "AI_analyze.py"],
                       cwd=os.path.join(SRC_ABS_PATH, "neural-network",''))

