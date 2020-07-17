#
# This file tests basic usage of the neural network eval function.
#

import os, subprocess
from sage.all import *
from period_graph import *


# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

E = [(x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + z*w**3),
     (x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**3*w + w**4),
     (x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + x*w**3),
     (x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + y*w**3),
     (x**4 + y**4 + z**4 + w**4,x**4 + y**4 + x*z**3 + w**4)]

# Run the training program
sE, _, _ = nn_sort(E)

assert Set(sE) == Set(E)
