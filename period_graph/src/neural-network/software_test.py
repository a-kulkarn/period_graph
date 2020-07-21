#
# This file tests basic usage of training and evaluation.
# NOTE: The training tests must be run beforehand to generate testing data.
#
# NOTE: This test *must* be run in the current directory with python3.
#

import os, subprocess

assert subprocess.call(["python3", "AI_train.py"]) == 0
assert subprocess.call(["python3", "AI_analyze.py"]) == 0

