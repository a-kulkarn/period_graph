#
# This file tests generating training data with the generator option
# WARNING: This test takes several hours.
#

import os, subprocess
from sage.all import *
from __init__ import *
TEST_PATH = os.path.join(os.path.join(SELF_PATH, "tests", ""))

# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

create_training_data(opts={'generator':'complete4', 'generate-quartics':None})
