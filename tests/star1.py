#
# This file tests whether the periods are correctly computed for a small star
# based around the Fermat vertex.
#

# TODO: We need to figure out how to fix paths with regard to the tests.
import os, subprocess
from sage.all import *
from __init__ import *
TEST_PATH = os.path.join(os.path.join(SELF_PATH, "tests", ""))

# Setup test edges.
R = PolynomialRing(QQ, 4, "xyzw")
(x,y,z,w) = R.gens()

E = [[x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + z*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**3*w + w**4],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + x*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + z**4 + y*w**3],
     [x**4 + y**4 + z**4 + w**4,x**4 + y**4 + x*z**3 + w**4]]

# Run the program
ivps(E)
integrate_edge_odes()
G = load_phase_III_graph()
initialize_fermat_directory()
carry_periods(G=G)

# Verify the results.
# assert len(os.listdir("../src/periods/")) == 6

# Run this afterward to check the period matrices.
res = subprocess.call(["magma", "-b", "verify-periods-are-correct.m"], cwd=TEST_PATH)
assert res == 0
