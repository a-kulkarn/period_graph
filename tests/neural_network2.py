#
# This file tests large input sets, involking parallelization.
#

# TODO: We need to figure out how to fix paths with regard to the tests.
import os, subprocess
from sage.all import *
from period_graph import SELF_PATH, nn_sort
TEST_PATH = os.path.join(os.path.join(SELF_PATH, "tests", ""))

# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

E = 20*[[x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + z*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**3*w + w**4],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + x*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + y*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + x*z**3 + w**4]]

# Run the training program
sE, _, _ = nn_sort(E)

