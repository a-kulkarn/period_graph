
from SAGE_CONFIG import *

load(SRC_ABS_PATH + "first-stage-analysis.sage")
load(SRC_ABS_PATH + "post-integration-analysis.sage")

# Load the ARBMatrixWrap class
load(pathToSuite+"arb_matrix_cereal_wrap.sage")


###################################################
# Functions for getting transition matrices

def path_edges(G, pth):
    return [(pth[i], pth[i+1], G.edge_label(pth[i], pth[i+1])) for i in range(len(pth)-1)]


def get_transition_mat(e):
    edata = e[2]
    etype = edata.edge_type()
    if etype == 'permutation':
        tm = compute_permutation_transition(e)
    elif etype == 'normal' or etype == 'weak':
        tm = load_transition_mat(e)
    else:
        raise ValueError("Invalid edge type given.")

    if edata.direction() == 'forward':
        return tm
    else:
        return tm^(-1)

def permutation_matrix_auf_list(perm_as_list):
    A = zero_matrix(ZZ, len(perm_as_list))
    for i in range(len(perm_as_list)):
        A[i,perm_as_list[i]-1] = 1 # Permutations are on symbols [x,y,z,w]
    return A
    
def compute_permutation_transition(e):
    """
    WARNING: Assumes that the S4_transition function has been loaded into the magma
    interpreter.
    """
    assert e[2].edge_type() == 'permutation'
    
    A = e[0].perm_as_matrix()
    B = e[1].perm_as_matrix().transpose()
    
    #A = permutation_matrix_auf_list(e[0].perm)
    #B = permutation_matrix_auf_list(e[1].perm).transpose()
    perm = (B*A).list()
    magma_output = magma.eval('S4_transition("{}","{}");'.format(perm, e[0].quartic()))
    return sage_eval("matrix(QQ, {})".format(magma_output))

    
def load_transition_mat(e):
    label = e[2]
    tm = load(SRC_ABS_PATH + 'ode-data/' + label.dirname() + '/' + 'transition_mat.sobj')
    tm = tm.arb_matrix()
    return tm

def edge_precision(e):
    if e[2].edge_type() == 'permutation':
        return math.inf
    else:
        try:
            tm = get_transition_mat(e)
            return -log(max(tm.apply_map(lambda x : x.diameter()).list()), 10)
        except:
            return 0
            
###################################################
# Functions for periods

def load_periods(v):
    per_mat = load(SRC_ABS_PATH + 'periods/' + str(v) + '/' + 'periods.sobj')
    return per_mat.arb_matrix()

import os
def save_periods(v, per):
    if not os.path.exists(SRC_ABS_PATH + 'periods/' + v + '/'):
        os.mkdir(SRC_ABS_PATH + 'periods/' + v + '/')
        save(ARBMatrixCerealWrap(per), SRC_ABS_PATH + 'periods/' + v + '/' + 'periods.sobj')
    return


# Constant to increase Magma's supply of working digits. Creates a safety buffer for
# Magma's ARB arithmetic.
MAGMA_WORKING_PRECISION_FACTOR = 1.2
def save_periods_magma(v, periods):
    """
    Save the periods in a magma-readable format. 
    Adapted from output_to_file function in integrator.sage.
    """
    ivpdir   = SRC_ABS_PATH + 'periods/' + v + '/'
    filename = 'periods-magma'

    # Create the periods file if it is not there.
    if not os.path.exists(ivpdir + filename):
        #os.mkdir('periods/' + v + '/')
    
        with open(ivpdir+filename,'w') as output_file:
            
            maximal_error = max(periods.apply_map(lambda x : x.diameter()).list());
            periods_mid   = periods.apply_map(lambda x : x.mid());

            print("Accumulated maximal error:", maximal_error)
            if maximal_error == 0:
                # For a default precision, use the value  stored in the base ring of the arb_matrix
                bit_precision      = periods.base_ring().precision()
                attained_precision = floor(log(2^bit_precision, 10))
            else:
                attained_precision = -maximal_error.log(10).round()

            # Magma first reads the complex ball precision and number of digits.
            output_file.write(str(attained_precision)+"\n")

            digits = ceil(attained_precision*MAGMA_WORKING_PRECISION_FACTOR);
            output_file.write(str(digits)+"\n")

            print("Writing the periods to file.")
            numrows = periods_mid.nrows()
            numcols = periods_mid.ncols()
            for i in [1..numrows]:
                output_file.write(str(periods_mid[i-1].list()))
                if i < numrows: output_file.write("\n")
####

# TODO: Abolish this global scope/load nonsense. This is *horrible* design.
# The values defined below are usually defined via the meta file or are constants
# in integrator.sage.
d = 4
fermat_type = [1,1,1,1]
bit_precision = ceil(log(10^DIGIT_PRECISION, 2))
ncpus = 8


# Basic initialization
def initialize_fermat_directory():
    R = ComplexBallField(bit_precision)
    fermat_string = quartic_data('x^4 + y^4 + z^4 + w^4').quartic_string()
    
    locals_dict = {'d' : 4,
                   'fermat_type' : [1,1,1,1],
                   'bit_precision' : 466,
                   'ncpus' : 8}

    # TODO: Replace with a save_eval with locals dict. Then test.
    load(pathToSuite+"fermat_periods.sage")
    
    fermat_period_data = periods_of_fermat(fermat_type)
    print("Fermat periods computed.")
    fpm_rows=fermat_period_data.nrows()
    fpm_cols=fermat_period_data.ncols()

    fermat_periods = MatrixSpace(R,fpm_rows,fpm_cols)(fermat_period_data)
    
    save_periods(fermat_string, fermat_periods)
    save_periods_magma(fermat_string, fermat_periods)

    
###########################################################
# Main functions

# The code below is inefficient and meant for testing.
# Also, we have Sage 8.6, and not 8.8 with the fancy "breadth_first_search" options.
#
# TODO: Since we are upgrading to Sage 9.0, we can optimize this part.

def carry_periods(G=None, start_vtx=quartic_data('x^4 + y^4 + z^4 + w^4'), verbose=False):
    """
    Move the period data from Fermat to all the other connected hypersurfaces.
    Optional parameters are:

    G         -- Directed graph of quartic data edges. Default is to load from file.
    start_vtx -- The starting vertex, as a quartic_data object.
    """

    # Ensure that Fermat is initialized.
    if start_vtx == quartic_data('x^4 + y^4 + z^4 + w^4'):
        try:
            load_periods(start_vtx.quartic_string())
        except IOError:
            initialize_fermat_directory()


    old_dir = magma.eval("GetCurrentDirectory()")
    magma.eval('ChangeDirectory("{}")'.format(SRC_ABS_PATH))
    magma.load(SRC_ABS_PATH + "magma/S4-transition.m")

    if verbose:
        print("Constructing graph...")
        
    if G == None:
        G = load_phase_III_graph()
        if verbose:
            print("Graph built. Inserting permutation links...")
        insert_S4_links(G)

    # This determines what route we take if there is ambiguity.
    weight_func = (lambda e : e[2].weight())
    short_paths = G.shortest_paths(start_vtx, by_weight=True, weight_function = weight_func)
    
    for v in G.vertices():
        print("Vertex: ", v) if verbose else None
            
        #path_verts = G.shortest_path(start_vtx, v, weight_function = weight_func)
        try:
            path_verts = short_paths[v]
        except KeyError:
            # There is no path linking the starting vertex to v.
            continue
            
        if len(path_verts) < 2:
            continue
        else:
            print(path_edges(G, path_verts)) if verbose else None
            carry_periods_along_path(path_edges(G, path_verts))

    #Cleanup
    magma.eval('ChangeDirectory("{}")'.format(old_dir))
###

def carry_periods_along_path(pe):

    weak_link_crossed = False
    
    for e in pe:
        source_per = load_periods(e[0].quartic_string())

        if weak_link_crossed:
            if e[2].edge_type() == 'permutation':

                # We can move the holomorphic periods, but nothing else.
                new_per = matrix(source_per.rows()[0])
            else:
                # We do not have enough periods to compute anything else.
                return
        else:
            tm = get_transition_mat(e)
            if e[2].edge_type() == 'weak':
                weak_link_crossed = True

            # Apply the transformation.
            new_per = tm*source_per
        ## end if 
            
        # Save the new periods to the file.
        save_periods(e[1].quartic_string(), new_per)
        save_periods_magma(e[1].quartic_string(), new_per)

    
####################
