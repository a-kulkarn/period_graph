

#############################
# EDGE WORKER PROCESS
#######

def edge_check_worker():

    while True:
        try:
            # Check for the terminate signal.
            if quitEvent.is_set():
                return "Quit!"

            # Retrieve an available job from the queue.
            if job_queue.empty():
                return "Quit!"
            
            pair = job_queue.get()

            # Retrieve an available filename from the queue.
            suffix = suffix_queue.get()

            # Check if the edge is traversable and write to the appropriate file.
            f0 = pair[0]
            f1 = pair[1]
            try:
                if edge_traversable(suffix, f0,f1):
                    write_quartic_data_to_file(suffix,f0,f1)

            except Exception as e:
                traceback.print_exc()
                print()
                raise e
            finally:
                # Put the file suffix back in the queue.
                suffix_queue.put(suffix)

        except Exception as e:
            traceback.print_exc()
            print()
            raise e    
#####


def write_quartic_data_to_file(suffix, f0, f1):
    with open("vertex-data/new-"+suffix, 'a') as F:
        label0, perm0 = s4act(f0)
        label1, perm1 = s4act(f1)
        F.write("{},{},[{}],None,None\n".format(label0, perm0, (label1,perm1)))
        F.write("{},{},[{}],None,None\n".format(label1, perm1, (label0,perm0)))
    return

def s4act(q_as_list):
    ## NOTE: Timeit output: "100 loops, best of 3: 4.74 ms per loop".
    #        According to the docs, each loop consists of 10^6 tests, so it is unlikely we will
    #        notice any problems.

    ## A way to choose a particular element in the s4 equivalence class of quartics.
    #  s4 is small enough that we can just enumerate through all the elements.
    R.<x,y,z,w> = PolynomialRing(QQ,4, order='lex')
    mons = ((x+y+z+w)^4).monomials()
    Rvars = [x,y,z,w]
    
    q = sum([ q_as_list[i]*mons[i] for i in range(35)] )

    # Need to use Permutations([1,2,3,4]) because of stupid type reasons.    
    S4elts  = list(Permutations([1,2,3,4]))
    S4orbit = [ q([Rvars[i-1] for i in sigma.inverse()]) for sigma in S4elts]
    

    # We rely on python's choice of internal ordering of objects.
    # As long as the choice of min is consistent, everything will work.
    #
    # Note that by listing over the inverses, we have that the returned
    # data `(s4rep, perm)` satisfies `s4rep(perm) = q`.
    s4rep = min(S4orbit)
    return list(s4rep), S4elts[S4orbit.index(s4rep)]
    
