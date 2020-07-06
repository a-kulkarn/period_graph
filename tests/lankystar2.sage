#
# This file tests whether the periods are correctly computed for a small graph
# based around the Fermat vertex. Some vertices are of distance 2 away from Fermat.
#

# TODO: We need to figure out how to fix paths with regard to the tests.
import os, subprocess

# Use the dumb import system to import from the super directory.
load("../__init__.sage")

# Setup test edges.
R.<x,y,z,w> = PolynomialRing(QQ, 4)

E = [[x^4 + y^4 + z^4 + w^4, x^4 + y^4 + z^4 + z*w^3],
     [x^3*w + y^4 + z^4 + w^4, x^4 + x^3*w + y^4 + z^4 + w^4]]


# Run the program
ivps(E)
integrate_edge_odes()

# Build a graph with some permutation links thrown in.
G = load_phase_III_graph()


# Compute the periods.
initialize_fermat_directory()

# Add a permutation edge.
u = quartic_data(x^4 + y^4 + z^4 + z*w^3)
v = quartic_data(u.s4label)
G.add_vertex(v)
G.add_edge(u, v, EdgeLabel(None, 'forward', 'permutation'))


# Compute the periods.
carry_periods(G=G)

# Verify the results.
# assert len(os.listdir("../src/periods/")) == 6

# Run this afterward to check the period matrices.
res = subprocess.call(["magma", "-b", "verify-periods-are-correct.m"])
assert res == 0
