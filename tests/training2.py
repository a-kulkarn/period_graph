#
# This file tests generating training data with the generator option
# WARNING: This test takes several hours.
#

import os, subprocess
from sage.all import *
from period_graph import *

# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

create_training_data(opts={'generator':'complete4', 'generate-quartics':None})
