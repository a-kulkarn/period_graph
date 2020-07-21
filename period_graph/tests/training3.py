#
# This file tests generating training data with the generator option and total jobs option.
# 
#

import os, subprocess
from sage.all import *
from period_graph import *


# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

create_training_data(opts={'generator':'complete4', 'total-jobs':10})
