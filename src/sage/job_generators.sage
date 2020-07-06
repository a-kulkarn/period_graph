
###################################################################################
#  Job generators

# Generate `m` random 2-element subsets of quartics
class random_edge_generator(object):
    def __init__(self, quartics, max_jobs="default"):
        self.state = 0
        self.quartics = quartics

        if max_jobs == "default":
            max_jobs = float('inf')
        else:
            self.max_jobs = max_jobs

    def __iter__(self):
        return self

    def next(self):
        if self.state < self.max_jobs:
            i=0
            j=0
            while i >= j:
                i = randint(0, len(self.quartics)-1)
                j = randint(0, len(self.quartics)-1)
                
            self.state += 1
            return [self.quartics[i],self.quartics[j]]                                        
        else:
            raise StopIteration()

    def empty(self):
        return self.state == self.max_jobs

    def remaining(self):
        return self.max_jobs - self.state


# Enumerate through all quartics sequentially
class complete_edge_generator(object):
    def __init__(self, quartics, max_jobs="default"):
        self.state = 0
        self.quartics = quartics

        if max_jobs == "default":
            self.max_jobs = binomial(len(quartics),2)
        else:
            self.max_jobs = max_jobs
            
        self.i = 1
        self.j = 0

    def __iter__(self):
        return self

    def next(self):
        if self.state < self.max_jobs:

            out_elem = [self.quartics[self.i], self.quartics[self.j]]

            # Increment. Logic in the increment step is to 
            # only consider pairs (i,j) with i > j.
            self.state += 1
            self.j += 1
            if self.i <= self.j:
                self.i += 1
                self.j  = 0

            return out_elem
        else:
            raise StopIteration()

    def set_job_interval(self, job_interval):

        self.state = job_interval[0]
        self.max_jobs = min(job_interval[1], binomial(len(self.quartics), 2))

        a = 0
        while binomial(a,2) < self.state:
            a += 1
        a = a - 1

        self.i = a
        self.j = self.state - binomial(a,2)
                    
        return
        
    def empty(self):
        return self.state == self.max_jobs

    def remaining(self):
        return self.max_jobs - self.state
    
###

# Used to generate the complete bipartite graph between 4/5 monomial quartics.
class complete_bipartite_edge_generator(object):
    def __init__(self, quartics0, quartics1, max_jobs="default"):
        self.state = 0
        self.quartics0 = quartics0
        self.quartics1 = quartics1
        
        self.m0 = len(quartics0)
        self.m1 = len(quartics1)

        if max_jobs == "default":            
            self.max_jobs = len(quartics0)*len(quartics1)
        else:
            self.max_jobs = max_jobs
       

        self.i = 0
        self.j = 0

    def __iter__(self):
        return self

    def next(self):
        if self.state < self.max_jobs:

            out_elem = [self.quartics0[self.i], self.quartics1[self.j]]

            # Increment. Logic in the increment step is to 
            # only consider pairs (i,j) with i > j.
            self.state += 1
            self.j += 1
            if self.m1 <= self.j:
                self.i += 1
                self.j  = 0

            return out_elem
        else:
            raise StopIteration()

    def empty(self):
        return self.state == self.max_jobs

    def remaining(self):
        return self.max_jobs - self.state


# Read a list of quartics from a file.
class file_edge_generator(object):
    def __init__(self, filename, max_jobs="default"):
        
        if max_jobs == "default":
            max_jobs = sum(1 for line in open(filename, 'r'))

        # TODO: Remove hard-coding of the number of variables.
        R.<x,y,z,w> = PolynomialRing(QQ, 4, order='lex')

        self.R = R
        self.mons = (sum(R.gens())^4).monomials()
        self.state = 0
        self.vars = {'x':x, 'y':y, 'z':z, 'w':w}
        self.file = open(filename, 'r')
        self.max_jobs = max_jobs

    def __iter__(self):
        return self

    def next(self):
        try:
            a = self.file.readline()

            # Terminate when a blank line is encountered.
            if a == '':
                raise StopIteration
            
            self.state += 1

            e = (sage_eval(a, locals=self.vars))
            assert e[0].parent() == self.R

            v0 = [e[0].monomial_coefficient(mon) for mon in self.mons]
            v1 = [e[1].monomial_coefficient(mon) for mon in self.mons]
            return [v0,v1]

        except StopIteration:
            raise StopIteration
        
        except Exception as err:
            traceback.print_exc()
            raise StopIteration

    def empty(self):
        return self.state == self.max_jobs

    def remaining(self):
        return self.max_jobs - self.state

# Read from a file of coefficient vectors.
class coeff_file_edge_generator(file_edge_generator):

    def __init__(self, filename, max_jobs="default"):            
        self.state = 0
        self.file = open(filename, 'r')

        if max_jobs == "default":
            self.max_jobs = sum(1 for line in open(filename, 'r'))
        else:
            self.max_jobs = max_jobs

    def next(self):
        try:
            a = self.file.readline()
            self.state += 1

            # TODO: Remove hardcoding of variable size.
            e = sage_eval(a)

            v0 = e[0:35]
            v1 = e[35:70]
            return [v0,v1]
        
        except StopIteration:
            raise StopIteration
        
        except Exception as err:
            traceback.print_exc()            
            raise StopIteration


# TODO: Work in proper inheritance construct.
class basic_job_generator(object):
    
    def empty(self):
        self.state == self.max_jobs

    def remaining(self):
        return self.max_jobs - self.state

        
class graph_edge_generator(basic_job_generator):
    
    def __init__(self, G):
        self.base_iter = ([e[0], e[1]] for e in G.edge_iterator())
        self.max_jobs = sum(1 for e in G.edge_iterator())
        self.state = 0

    def __iter__(self):
        return self
    
    def next(self):
        e = next(self.base_iter)
        self.state += 1
        return e
