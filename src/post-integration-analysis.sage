
from SAGE_CONFIG import *

# Load the ARBMatrixWrap class
load(pathToSuite+"arb_matrix_cereal_wrap.sage")

###################################################
# Construction of Graph

# Here we construct the graph from the return code object and the
# vertex manifest.

def parse_vtx_string(s):
    return s.strip('vtx').strip('()')

def parse_ret_code(c):
    return c[0][0][0].strip('ode-data').strip('/')

import os
def reconstruct_return_codes():
    retc = []    
    for dirname in os.listdir(SRC_ABS_PATH + "ode-data"):
        if os.path.exists("{}ode-data/{}/transition_mat.sobj".format(SRC_ABS_PATH, dirname)):
            retc.append(((("ode-data/" + dirname,), {}), 0))
        else:
            retc.append(((("ode-data/" + dirname,), {}), 1))

    save(retc, SRC_ABS_PATH + "Z-integration-return-codes.sobj")
    return


def load_phase_III_graph(directed=True, allow_weak_edges=True):

    manifest = load_manifest()
    success_list = []
    for dirname in os.listdir(SRC_ABS_PATH + "ode-data"):
        ode_safe_write = os.path.exists(
            "{}ode-data/{}/safe_write_flag".format(SRC_ABS_PATH, dirname))

        transition_matrix_exists = os.path.exists(
            "{}ode-data/{}/transition_mat.sobj".format(SRC_ABS_PATH, dirname))

        if ode_safe_write and transition_matrix_exists:
            success_list += [dirname]
    
    if directed:
        G = DiGraph({})
    else:
        G = Graph()

    
    with open(SRC_ABS_PATH + "edge-manifest",'r') as F:
        for line in F:
            v,w,dirid = line.rstrip().split(",")
            G.add_vertex(quartic_data(v))
            G.add_vertex(quartic_data(w))

    for c in success_list:
        # Load the transition matrix to attach necessary data to the edge.
        tm = load(SRC_ABS_PATH + 'ode-data/' + c + '/' + 'transition_mat.sobj')
        label = EdgeLabel(c, 'forward', tm)

        try:
            if allow_weak_edges or not label.is_weak():
                G.add_edge(manifest[c], label=label)

                if not label.is_weak():
                    backward_label = EdgeLabel(c, 'backward', tm)
                    G.add_edge((manifest[c][1], manifest[c][0]), label=backward_label)
        except KeyError as e:
            print("WARNING: Manifest key error: ", e)
                    
    return G


def _my_phaseIII(weak=True, isolated_vertices=True):
    H3 = load_phase_III_graph(directed=True, allow_weak_edges=weak)

    if isolated_vertices:
        # Function defined in first stage analysis.
        add_isolated_vertices(G,[4,5])

    insert_S4_links(H3)
    return H3
