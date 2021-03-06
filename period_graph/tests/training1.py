#
# This file tests basic usage of the training data creator.
#

# TODO: We need to figure out how to fix paths with regard to the tests.
import os, subprocess
from sage.all import *
from period_graph import *

# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

E = [[x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + z*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**3*w + w**4],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + x*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + y*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + x*z**3 + w**4]]

# Run the training program
write_user_edges_to_file(E)
create_training_data()
