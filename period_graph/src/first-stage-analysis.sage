
import re, os, time
from SAGE_CONFIG import *

# Quartic data object.
class quartic_data:

    # Static variables
    R.<x,y,z,w> = PolynomialRing(QQ, 4, order='lex')
    mons = ((x+y+z+w)^4).monomials()
    Rvars = [x,y,z,w]

    def __init__(self, *args, **kwds):

        if len(args) == 3: # (s4label, perm, neighbours)
            s4label = args[0]
            perm = args[1]
            neighbours = args[2]
            
        elif len(args) == 1: # Case of a list, polynomial, string, or compressed string
            if isinstance(args[0], list):
                mons = self.mons
                f = sum(mons[i] * args[0][i] for i in range(len(mons)))
                return self.__init__(f)
                
            elif isinstance(args[0], str):
                s = args[0]
                if s[0:3] == "vtx":
                    s = s.strip("vtx").strip('()')
                    
                f = sage_eval(s, locals={str(a):a for a in self.Rvars})
                return self.__init__(f)
                
            else:
                s4label, perm = s4act_poly(args[0])
                neighbours = None
            
        self.s4label = sum(mul(x) for x in s4label)
        self.neighbours = neighbours
        self.perm = [self.R.gens()[j-1] for j in perm]
        
    def __hash__(self):
        return str(self.s4label).__hash__()

    def __eq__(self, other):
        return self.s4label == other.s4label and self.perm == other.perm

    def __lt__(self, other):
        return self.quartic() < other.quartic()

    def __repr__(self):
        return "vtx({})".format(self.quartic())

    def quartic(self):
        return self.s4label(self.perm)

    def quartic_string(self):
        return str(self.s4label(self.perm))

    def quartic_long_list(self):
        q = self.quartic()
        return [q.monomial_coefficient(m) for m in mons]

    def perm_as_matrix(self):
        A = zero_matrix(ZZ, len(self.perm))
        for i,m in enumerate(self.Rvars):
            A[i, self.perm.index(m)] = 1
        return A

    
## end class

class EdgeLabel(object):

    def __init__(self, dirname, direction, tm):
        self._directory = dirname
        self._direction = direction

        if tm == 'permutation':
            self._is_weak = False
        else:
            self._is_weak = (not tm.nrows == 21)

        # Determine the type based on the data
        if self._is_weak:
            self._edge_type = 'weak'
        elif tm == 'permutation':
            self._edge_type = 'permutation'
        elif tm == 'S4':
            raise ValueError("'S4' is not a valid edge type. Perhaps you meant 'permutation'?")
        else:
            self._edge_type = 'normal'
            
        
    def __repr__(self):
        return str([self.edge_type(), self.direction()])
    
    def direction(self):
        return self._direction

    def dirname(self):
        return self._directory

    def is_weak(self):
        return self._is_weak

    def edge_type(self):
        return self._edge_type

    def weight(self):
        # TODO: Lots of wacky hardcoded magic numbers. This is deeply unsafe...
        weights = {'weak' : 10000, 'normal' : 100, 'permutation' : 1}
        return weights[self.edge_type()]
    

## end class

#############################################################################################
#
# Parsing, conversion, and misc.

def vector_to_edge(v):
    return (quartic_data(v[0:35]), quartic_data(v[35:71]))

def condensed_vtx_string(v):
    vstr = str(v).replace('^','').replace(' ','').replace('*','')
    vpost = ""
    
    if len(vstr) >= 100:
        vpost = '__' + str(hash(str(v)))
        terms = vstr.split('+')

        if len(terms) != 1:
            vstr = terms[0] + "+...+" + terms[-1]

        # If the string is still too long, bail out and use a hash value.
        if len(vstr) >= 100:
            vstr = "vtxhash="+str(hash(str(v)))
    return vstr, vpost


def parse_compressed_pol(R, f0):
    x,y,z,w = R.gens()
    var_dic = {'x':x, 'y':y, 'z':z, 'w':w}
    
    # Get terms separated by signs (with +/-'s kept in the list).
    terms = re.split(re.compile('([-+])'), f0)

    # Detect an initial '-'
    if terms[0] == '':
        terms = terms[1:]
    
    # Main polynomial building loop.
    res = R(0)
    current_sign = R(1)
    for term in terms:
        
        # Detect if the term is actually an operand.
        if term == '+':
            current_sign = R(1)
            continue
        elif term == '-':
            current_sign = R(-1)
            continue

        # Build the term by parsing the appropriate string.
        i = 0
        T = current_sign
        while i < len(term):
            var = sage_eval(term[i], locals=var_dic)

            if i+1==len(term) or not (term[i+1] in map(str,range(5))):
                T *= var
                i += 1
            else:
                T *= prod(var for j in range(eval(term[i+1])))    
                i += 2
        res += T
    return res


def edge_ivp_label(e):
    if isinstance(e[0], quartic_data) and isinstance(e[1], quartic_data):
        v = e[0].quartic()
        w = e[1].quartic()
    else:
        v = e[0]
        w = e[1]
        
    vstr, vpost = condensed_vtx_string(v)
    wstr, wpost = condensed_vtx_string(w)
    return "edge-ivp__"+vstr+'__'+wstr+vpost+wpost


def parse_ivp_dirname(dirname):
    R = quartic_data.R
    pols = dirname.split('__')
    f0 = parse_compressed_pol(R, pols[1])
    f1 = parse_compressed_pol(R, pols[2])
    return f0, f1


def ccsizes(G):
    return [len(x) for x in G.connected_components()]

#####################################################################################
#
# S4 action.

# This function needs to be associated to the quartic data object
def s4act(q_as_list):
    ## NOTE: Timeit output: "100 loops, best of 3: 4.74 ms per loop".
    #        According to the docs, each loop consists of 10^6 tests, so it is unlikely we will
    #        notice any problems.

    ## A way to choose a particular element in the s4 equivalence class of quartics.
    #  s4 is small enough that we can just enumerate through all the elements.

    mons = quartic_data.mons
    q = sum([ q_as_list[i]*mons[i] for i in range(35)])
    s4rep, perm = s4act_poly(q)
    return list(s4rep), perm


def s4act_poly(q):
    Rvars = quartic_data.Rvars
    
    # Need to use Permutations([1,2,3,4]) because of stupid type reasons.    
    S4elts  = list(Permutations([1,2,3,4]))
    S4orbit = [q([Rvars[i-1] for i in sigma.inverse()]) for sigma in S4elts]
    s4rep = min(S4orbit)
    
    # We rely on python's choice of internal ordering of objects.
    # As long as the choice of min is consistent, everything will work.
    #
    # Note that by listing over the inverses, we have that the returned
    # data `(s4rep, perm)` satisfies `s4rep(perm) = q`.
    return s4rep, S4elts[S4orbit.index(s4rep)]
    

###################################################
# Construction of Graph

# Override the add_edge to use the equiv. class. rep.
class MyQuotientGraph(Graph):
    def add_edge(self, v,w,label):
        Graph.add_edge(self, v.s4label, w.s4label, label=None)

def load_phase_I_graph(dirname=SRC_ABS_PATH + "vertex-data"):
    t0 = time.time()
    G = Graph({})
    x,y,z,w = quartic_data.Rvars

    for filename in os.listdir(dirname):
        with open(dirname+'/'+filename,'r') as F:
            for lines in F:
                q, perm, neighbours, a, b = sage_eval(lines, locals={'x':x, 'y':y, 'z':z, 'w':w})
                qd = quartic_data(q, perm, neighbours)
                neighbour_data = neighbours[0]
                qn = quartic_data(neighbour_data[0], neighbour_data[1], None)
                G.add_edge((qd, qn))
                
    t1 = time.time()
    print("Done. Time to construct graph: " + str(t1-t0))
    return G


def load_phase_II_graph():
    G = Graph()
    insert_manifest_edges(G)
    return G

def add_isolated_vertices(G,d):
    
    if isinstance(d, list):
        for elt in d:
            add_isolated_vertices(G,elt)
        return
            
    filename = SRC_ABS_PATH + "quartics-cache/smquartics"+str(d)
    with open(filename) as F:
        for line in F:
            coeffs = eval(line)
            fs4, perm = s4act(coeffs)
            G.add_vertex(quartic_data(fs4, perm, None))
    return


def insert_S4_links(G):
    # Note: This doesn't add all possible S4 links, but
    # rather a spanning tree of each S4-cluster.
    old_verts = G.vertices()
    for v in old_verts:
        if not v.quartic() == v.s4label:
            label = EdgeLabel(None, 'forward', 'permutation')
            w = quartic_data(v.s4label)
            G.add_edge(w, v, label=label)
            G.add_edge(v, w, label=label)


def quotient_graph(G):
    GmodS  = Graph()
    II_dic = {} 

    for v in G.vertices():
        GmodS.add_vertex(v.s4label)

    for e in G.edges():
        v = e[0].s4label
        w = e[1].s4label

        if not v == w:
            GmodS.add_edge(v,w)
            try:
                II_dic[(v,w)] = II_dic[(v,w)] + [e]
            except KeyError:
                II_dic[(v,w)] = [e]

    print("Quotient graph constructed.")
    return GmodS, II_dic

def rand_lifts(H, II_dic):
    return [choice(II_dic[(e[0],e[1])]) for e in H.edges()]
        



############################################################################################
# Manifest functions
    
# Recover the edge manifest in case there is a crash.
def reconstruct_edge_manifest():
    R = quartic_data.R
    with open(SRC_ABS_PATH + "edge-manifest", 'w') as F:
        for dirname in os.listdir(SRC_ABS_PATH + "ode-data"):
            if os.path.exists("{}ode-data/{}/safe_write_flag".format(SRC_ABS_PATH, dirname)):
                f0, f1 = parse_ivp_dirname(dirname)
                F.write("vtx({}),vtx({}),{}\n".format(f0,f1,dirname))
    return
##

def load_manifest():
    manifest = {}
    with open(SRC_ABS_PATH + "edge-manifest",'r') as F:
        for line in F:
            v,w,dirid = line.rstrip().split(",")
            manifest[dirid] = (quartic_data(v), quartic_data(w))

    return manifest


def insert_manifest_edges(G):
    try:
        with open(SRC_ABS_PATH + "edge-manifest",'r') as F:
            for line in F:
                a, b, dirname = line.rstrip().split(',')
                f = quartic_data(a)
                g = quartic_data(b)
                G.add_edge(f,g)
    except IOError:
        pass
    return

            
###################################################
# Feedback graph

def parse_ret_code(c):
    return c[0][0][0].strip('ode-data').strip('/')

#TODO: Remove duplication with "construct_phase_III_graph"
def construct_phase_III_feedback():
    retc = load(SRC_ABS_PATH + "Z-integration-return-codes.sobj")
    success_list = [parse_ret_code(c) for c in retc if c[1]==0]

    v_list = []
    e_dict = {}

    with open(SRC_ABS_PATH + "edge-manifest",'r') as F:
        for line in F:
            v,w,dirid = line.rstrip().split(",")
            v_list += [quartic_data(v), quartic_data(w)]
            e_dict[dirid] = (quartic_data(v), quartic_data(w))

    G = Graph()

    for v in Set(v_list):
        G.add_vertex(v)

    for c in success_list:
        # Add the edge and its inverse to the graph.
        G.add_edge(e_dict[c])
        G.add_edge((e_dict[c][1], e_dict[c][0]))

    return G


###################################################
# Edge iterators.

def rand_lift_iterator(G):
    H, II_dic = quotient_graph(G)
    return iter(map((lambda e : [e[0],e[1]]), rand_lifts(H,II_dic)))


# Generally, ignore if the edge is either already in the manifest,
# the edge connects two things in the exit graph,
# or the edge connects two things in the post-integration graph.
#
class special_iterator:
    def __init__(self, G, exit_graph):
        self.exit_graph = exit_graph
        self.base_iter = G.edge_iterator()

    def __iter__(self):
        return self
        
    def next(self):
        while True:
            e = self.base_iter.next()

            if e[0] not in self.exit_graph or e[1] not in self.exit_graph:
                return e
            elif self.exit_graph.shortest_path(e[0],e[1]) == 0:
                return e


    
# Find directories without a certain file.
#find base_dir -mindepth 2 -maxdepth 2 -type d '!' -exec test -e "{}/cover.jpg" ';' -print
# find base_dir -type d '!' -exec test -e "{}/safe_write_flag" ';' -print
