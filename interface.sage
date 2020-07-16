import os, sys, subprocess
sys.path.insert(1, SELF_PATH + "src/")
sys.path.insert(2, SELF_PATH + "src/suite/")
from SAGE_CONFIG import *
import numpy as np

# Stupid imports (should be pure python in the future).
load(SRC_ABS_PATH + "sage/phase_I_util.py")  # Needed for nn_sort.
load(SRC_ABS_PATH + "first-stage-analysis.sage")
import src.integrator
from src.carry_periods import *
load(SRC_ABS_PATH + "sage/to_AI_pipe.sage")
load(SRC_ABS_PATH + "sage/sage_data_handling.sage")


############################################################################################
# Testing
############################################################################################

def load_test(testfile=SRC_ABS_PATH+"user_input/test_edges"):
    """
    Loads a file containing a list of polynomials. Returns a list
    of homogeneous quartics in 4 variables.
    """
    R.<x,y,z,w> = PolynomialRing(QQ,4)
    lst = []
    with open(testfile) as F:
        for line in F:
            tmp = sage_eval(line, locals={'x':x, 'y':y, 'z':z, 'w':w})
            lst.append(tmp)
    return lst

############################################################################################
# Util / misc
############################################################################################

def lex_order_mons_of_degree(R,d):
    # Note: Only works for characteristic 0.    
    P = R.change_ring(order='lex')
    mons = (sum(P.gens()) ** d).monomials()
    return [R(m) for m in mons]

def isSmooth(g):
    return ProjectiveHypersurface(g).is_smooth()

def get_simple_polys(n,degree=4, R=PolynomialRing(QQ,'x,y,z,w')):
    """
    Return smooth polynomials with n terms with each coefficient equal to 1.
    """
    mons = lex_order_mons_of_degree(R,degree)
    if n > len(mons):
        raise ValueError("The number of terms may not exceed the number of monomials.")

    return [sum(p) for p in Subsets(mons,n) if isSmooth(sum(p))]


def convert_folder_to_edge(folder, R=PolynomialRing(QQ,'x,y,z,w')):
    pols = folder.split('__')
    f0 = parse_compressed_pol(R, pols[1])
    f1 = parse_compressed_pol(R, pols[2])
    return (f0,f1)

def _quartics_to_file(filename, quartics):
    """
    Print the list of quartics to a file in a human readable way.
    """
    with open(filename, 'w') as F:
        lststr = str(quartics).replace(',', ',\n')
        F.write(lststr)
        F.write('\n')

############################################################################################
# Neural network sorting (nn_sort)
############################################################################################

def read_nn_results(parent, probabilities=True):
    """
    Read the results output by the neural network. The polynomials are returned
    in the parent ring specified by the first argument.
    """
    mons = lex_order_mons_of_degree(parent,4)
    output = []
    with open(SRC_ABS_PATH + "ai_output", 'r') as F:
        for line in F:
            e = eval(line)
            v0 = e[0:len(mons)]
            v1 = e[len(mons):2*len(mons)]

            q0 = sum(mons[i] * v0[i] for i in range(len(mons)))
            q1 = sum(mons[i] * v1[i] for i in range(len(mons)))
            output.append((q0,q1))

    if probabilities:
        probs = np.loadtxt(SRC_ABS_PATH + "ai_probabilities", dtype=float, ndmin=2)

        # sort list so highest probability comes first
        # the polynomials are already sorted accordingly
        probs_sorted = probs[(-probs[:,0]).argsort()]

        # pEN preserves the order of the input list, this too can be valuable so is preserved
        return output, probs_sorted, probs
    else:
        return output



def nn_sort(edges, probabilities=True):
    """
    Given a list of edges `edges` (each edge of the form (f,g), where `f,g` are homogeneous
    in 4 variables), return a sorted list of edges in descending order of AI-score.

    If probabilities=True, also return the 2D array of all neural network probabilities.

    This function also modifes the internal `ai_file`. 
    """

    if len(edges) == 0:
        return edges

    R = edges[0][0].parent()
    mons = lex_order_mons_of_degree(R,4)

    convert = (lambda a : [a.monomial_coefficient(m) for m in mons])
    E_list_form = map((lambda a : (convert(a[0]), convert(a[1]))), edges)

    # Launch the AI ranking
    send_jobs_to_AI(E_list_form)
    run_ai_eval()
    return read_nn_results(R, probabilities=probabilities)
###


def rerun_nn(parent=PolynomialRing(QQ, 'x,y,z,w'), probabilities=True):
    """
    Rerun nn_sort on the unlabelled edge data. 
    """
    run_ai_eval()
    return read_nn_results(parent, probabilities=probabilities)



def write_user_edges_to_file(edges):
    """
    Given a list of edges `edges` (each edge of the form (f,g), where `f,g` are homogeneous
    in 4 variables), save this list to the `user_edges` file to be read by the
    main programs. The names of the variables are changed to `x,y,z,w`.
    """
    R.<x,y,z,w> = PolynomialRing(QQ,4)
    with open(SRC_ABS_PATH + "user_input/" + "user_edges", 'w+') as F:
        for e in edges:
            F.write("[{},{}]\n".format(R(e[0]), R(e[1])))
    return

# Aliases
edges_to_file = write_user_edges_to_file
to_user_file  = write_user_edges_to_file


############################################################################################
# Compute transition matrices.
############################################################################################

def _raise_exit_function_not_implemented_error():
    help_string = (
            """
            The Interface for automatic parsing of the exit function has not been
            designed. Please encode you desired function in the file

            {}user_input/user_exit_functions.sage

            and remember to set `CONSTRUCT_GRAPH=True`. I am aware it is possible to
            use function pickling, but as this interface is a prototype, I am not
            committing to a framework at this time.
            """)

    print(help_string)
    raise NotImplementedError


# TODO: Need to somehow feed in the exit function and construct graph variables.
def compute_transition_matrices(edges, exit_function=None):
    """
    Given a list of edges `edges` (each edge of the form (f,g), where `f,g` are homogeneous
    in 4 variables), attempt to compute the transition matrices. 
    """
    if len(edges) == 0:
        return
    
    write_user_edges_to_file(edges)

    if not exit_function == None:
        _raise_exit_function_not_implemented_error()    
        # Pickle exit_function (to be loaded later)
        #with open(SRC_ABS_PATH + "user_input/" + "pickled_user_function", 'w+') as F:
        #    pass
        
    construct_edge_odes()
    integrate_edge_odes()
    return


def first_ivps(edges, exit_function=None):
    """
    Given a list of edges `edges` (each edge of the form (f,g), where `f,g` are homogeneous
    in 4 variables), attempt to compute the first ODE associated to eade edge. 
    """
    if len(edges) == 0:
        return
    
    write_user_edges_to_file(edges)

    if not exit_function == None:
        _raise_exit_function_not_implemented_error()

    opts = {'generator':'file', 'forgo-manifest':None, "only-first":None}
    construct_edge_odes(opts=opts)
    return


def ivps(edges, opts={'generator':'file', 'forgo-manifest':None}):
    """
    Given a list of edges `edges` (each edge of the form (f,g), where `f,g` are homogeneous
    in 4 variables), attempt to compute the initial value problems (IVPs). 
    """

    if len(edges) == 0:
        return
    
    write_user_edges_to_file(edges)
    construct_edge_odes(opts)
    return


def load_transition_matrix(e):
    """
    Loads the transition matrix associated to `e = (f,g)`, (where f,g are homogeneous
    quartics), provided it exists. Raises an error otherwise.
    """

    R.<x,y,z,w> = PolynomialRing(QQ,4)
    e0str = str(R(e[0]))
    e1str = str(R(e[1]))
    G = construct_phase_III_graph()

    e_G = [ed for ed in G.edges() if ed[0]==e0str and ed[1]==e1str][0]
    return load_transition_mat(e_G)


############################################################################################
# Help / Info
############################################################################################

def interface_help():
    help_string = (
        """
        COMMANDS:

        compute_transition_matrices
        write_user_edges_to_file
        nn_sort

        ivps
        first_ivps

        create_training_data
        construct_edge_odes
        integrate_edge_odes

        load_test
        load_transition_matrix
        carry_periods

        insert_S4_links
        add_isolated_vertices
        """)
    print(help_string)


############################################################################################
# Core wrappings for shell utilities.
############################################################################################


def create_training_data(opts={'generator':'file'}):
    subprocess.call(["sage", "create-training-data.sage"] + format_subproc_opts(opts),
                    cwd=SRC_ABS_PATH)


def construct_edge_odes(opts={'generator':'file', 'forgo-manifest':None}):
    subprocess.call(["sage", "construct-edge-odes.sage"] + format_subproc_opts(opts),
                    cwd=SRC_ABS_PATH)


def integrate_edge_odes(opts={'generator':'file'}):
    src.integrator._integrate_edge_odes(**opts)
    #subprocess.call(["sage", "integrate-edge-odes.sage"] + format_subproc_opts(opts),
    #                cwd=SRC_ABS_PATH)


def run_ai_eval():
    subprocess.check_call([PYTHON3_BIN, "neural-network/AI_eval.py"],
                          cwd=SRC_ABS_PATH)


def format_subproc_opts(opts):
    opt_list = list(opts)
    pass_to_subproc = []

    for opt in opt_list:
        arg = opts[opt]
        if arg == None:
            pass_to_subproc += ["--{}".format(str(opt))]
        else:
            pass_to_subproc += ["--{}={}".format(str(opt), str(arg))]
    return pass_to_subproc



############################################################################################
# Cleanup utilities.
############################################################################################

import shutil
def manifest_size():
    return subprocess.check_output(["wc","-l", "edge-manifest"],
                                   cwd=SRC_ABS_PATH)

def clean_ode_failures():
    for dirname in os.listdir(SRC_ABS_PATH + "ode-data"):
        if not os.path.exists("{}ode-data/{}/safe_write_flag".format(SRC_ABS_PATH, dirname)):
            vic = "{}ode-data/{}".format(SRC_ABS_PATH, dirname)
            dest = os.path.join(SRC_ABS_PATH,"failed-ode-step", dirname)
            shutil.move(vic, dest)
    return


############################################################################################
# Timings data
############################################################################################

def timings_from_training_successes(folder=os.path.join(TRAINING_PATH, "edge-data"), verbose=True):
    """
    Returns a list of pairs (edge, timings), where each edge is given as a
    pair of quartic_data objects and each timings is a list [time, order, degree].

    Note that only successes are returned, as failed edges just have placeholder data for
    [time, order, degree].
    """

    dataShape = {'edgesX-*.csv':2*35+1, 'timingsY-*.csv':3+1}

    data_gruppe, is_data_labelled = ReadData(folder, dataShape, verbose=verbose)
    assert is_data_labelled
    
    # Format and return the result.
    edges_vec = [[ZZ(x) for x in e] for e in data_gruppe['edgesX-*.csv']]
    edges = map(vector_to_edge, edges_vec)

    timings = [x.astype(float) for x in data_gruppe['timingsY-*.csv']]
    timings = [list(x) for x in timings]
        
    return zip(edges, timings)



class TimingsData(object):

    def __init__(self, filename):
        """
        Example timings data format:

        AIStream: GroebnerInit:  0.000
        AIStream: 0.000
        AIStream: 0.000
        AIStream: total-PF: 1, 1: 0.000
        AIStream: GroebnerInit:  0.000
        AIStream: 0.000
        AIStream: 0.000
        AIStream: total-PF: 1, 1: 0.000
        AIStream: 0.000
        AIStream: 0.000
        AIStream: 0.010
        AIStream: total-PF: 2, 1: 0.010
        ...

        """
        self.PF_time = {}
        
        with open(filename, 'r') as timings:
            for line in timings:
                split_line=line.split(":")
                
                if len(split_line) == 1:  # Ignore empty lines.
                    continue
                
                if split_line[1].strip() == "GroebnerInit":
                    self._groebner_time = float(split_line[2])
                    
                elif split_line[1].strip() == "total-PF":
                    label = int(split_line[2].split(",")[0]) # Specific to our computations.
                    self.PF_time[label] = float(split_line[3])

    def number_completed_odes(self):
        return len(self.PF_time.keys())

    def total_PF_time(self):
        return sum(t for k,t in self.PF_time.items())

    def groebner_time(self):
        return self._groebner_time

    def total_time(self):
        return self.groebner_time() + self.total_PF_time()

    def PF_timings(self):
        return self.PF_time
    
#### end class


def get_ivp_timings(odedata=SRC_ABS_PATH + "ode-data", only_first=False, only_completed=True):
    edge_and_time=[]
    for folder in os.listdir(odedata):
        for filename in os.listdir(os.path.join(odedata,folder)):
            if filename.endswith("timings"):
                timings = TimingsData(os.path.join(odedata,folder,filename))            
                e = convert_folder_to_edge(folder)
                
                if only_first == True:
                    try:
                        t = timings.PF_timings()[1]
                        edge_and_time.append((e,t))
                    except KeyError:
                        edge_and_time.append((e, math.inf))
                        
                elif not only_completed or timings.number_completed_odes() == 21: # for K3 surfaces
                    t = timings.total_PF_time()
                    edge_and_time.append((e,t))
                        
    edge_and_time.sort(key=lambda x:-x[1])
    return edge_and_time


def get_first_ivp_timing(odedata=SRC_ABS_PATH + "ode-data"):
    return get_ivp_timings(odedata=odedata, only_first=True)

