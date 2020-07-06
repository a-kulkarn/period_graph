
# Exit functions should be a function on a graph whose vertices are quartics.
# There are also two exit functions, one for each of the ode-based computation steps.

# Decide whether to keep track of an explored graph.
# This *must* be set to true in order for the exit
# functions to work.
CONSTRUCT_GRAPH = True

####################################################
#
#  Stage I exit function.
#
####################################################

my_verts = [quartic_data('x^4+y^4+z^4+w^4'), 
            quartic_data('x^4+y^4+z*w^3+z^3*w')]

def user_stop_function(G):
    if all(v in G for v in my_verts):
        return True
    else:
        return False


####################################################
#
#  Stage II exit function.
#
####################################################

def user_stop_function_II(G):
    #print G.edges()
    #if len(G.cycle_basis()) > 0:
    if False:
        return True
    else:
        return False

