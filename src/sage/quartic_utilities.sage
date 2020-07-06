
# Function to generate the list of quartics.
#
# Instead, generate the smooth monomials, and sample pairs randomly.
# Storing the poly-pairs is much too expensive.
def generate_k_mon_quartics(k):
    R.<x0,x1,x2,x3> = PolynomialRing(QQ,4, order='lex')
    P3 = ProjectiveSpace(R)
    S = Subsets(range(35), k)
    mons = (sum(P3.gens())^4).monomials()    
    poly = [ sum( mons[a] for a in A) for A in S ]    
    smooth_polys = {p for p in poly if P3.subscheme(p).is_smooth() }          

    return smooth_polys, mons
##

def generate_small4_quartics():
    R.<x0,x1,x2,x3> = PolynomialRing(QQ,4, order='lex')
    P3 = ProjectiveSpace(R)
    S = Subsets(range(35), 4)
    mons = (sum(P3.gens())^4).monomials()    

    polys = [x0^4 + x1^4 + x2^4 + x3^4,
             x0^3*x2 + x1^4 + x2^4 + x3^4,
             x0^4 + x0*x3^3 + x1^4 + x2^4];
    
    # x0^4 + x1^4 + x2^3*x3 + x2*x3^3,
    # x0^3*x3 + x1^4 + x2^4 + x3^4,
    # x0^3*x1 + x1^4 + x2^4 + x2*x3^3,
    # x0^4 + x1^3*x3 + x1*x3^3 + x2^4]
    return polys, mons
##

# TODO: save_quartics and load_quartics should be inverses.
#
# Save the pairs of smooth quartics for faster debugging.
def save_quartics(filename, quartics, mons):    
    qfile = open(filename,'w')
    for qq in quartics:        
        f0 = [qq.monomial_coefficient(mon) for mon in mons]
        qfile.write( str(f0) + '\n')
        
    qfile.close()
    return
     

def load_quartics(filename):
    qfile = open(filename,'r')
    quartic_list = []  
    for line in qfile:
        quartic_list += [eval(line)]
    qfile.close()    
    return quartic_list
##

