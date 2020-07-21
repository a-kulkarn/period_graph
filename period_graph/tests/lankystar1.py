#
# This file tests whether the periods are correctly computed for a small graph
# based around the Fermat vertex. Some vertices are of distance 2 away from Fermat.
#

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


# Run the program
ivps(E)
integrate_edge_odes()

# Build a graph with some permutation links thrown in.
G = load_phase_III_graph()


# Compute the periods.
initialize_fermat_directory()

# Add a permutation edge.
u = quartic_data(x**4 + y**4 + z**4 + z*w**3)
v = quartic_data(u.s4label)
G.add_vertex(v)
G.add_edge(u, v, EdgeLabel(None, 'forward', 'permutation'))

# Compute the periods.
carry_periods(G=G)

# Verify the results.
# assert len(os.listdir("../src/periods/")) == 6

# Run this afterward to check the period matrices.
res = subprocess.call(["magma", "-b", "verify-periods-are-correct.m"], cwd=TEST_PATH)
assert res == 0
