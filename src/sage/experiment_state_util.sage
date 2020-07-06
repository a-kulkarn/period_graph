
def top_up_queue(job_generator):
    while job_queue.qsize() <= BUFFERSIZE and not job_generator.empty():
        job_queue.put(job_generator.next())
    return

## Creates the job Queue and job iterator.
import platform
def initialize_experiment_data(experiment='complete4',
                               generate_quartics=False,
                               max_jobs="default",
                               job_interval="default"):

    quartics_dirname = "quartics-cache/"
    
    ## Step 1 is to generate the list of smooth quartics
    if experiment == 'complete4':
        if generate_quartics:
            quartics, mons = generate_k_mon_quartics(4)
            save_quartics(quartics_dirname+'smquartics4', quartics, mons)
            print("Generation finished")
            
        quartics = load_quartics(quartics_dirname+'smquartics4')
        job_generator = complete_edge_generator(quartics, max_jobs)
        
    elif experiment == 'complete5':
        if generate_quartics:
            quartics, mons = generate_k_mon_quartics(5)
            save_quartics(quartics_dirname+'smquartics5', quartics, mons)
            print("Generation finished")
            
        quartics = load_quartics(quartics_dirname+'smquartics5')
        job_generator = complete_edge_generator(quartics, max_jobs)

    elif experiment == 'random5':
        if generate_quartics:
            quartics, mons = generate_k_mon_quartics(5)
            save_quartics(quartics_dirname+'smquartics5', quartics, mons)
            print("Generation finished")
            
        quartics = load_quartics(quartics_dirname+'smquartics5')
        job_generator = random_edge_generator(quartics)
        
    elif experiment == 'complete4-5':
        if generate_quartics:
            quartics4, mons4 = generate_k_mon_quartics(4)
            quartics5, mons5 = generate_k_mon_quartics(5)
            save_quartics(quartics_dirname+'smquartics4', quartics4, mons4)
            save_quartics(quartics_dirname+'smquartics5', quartics5, mons5)
            print("Generation finished")
            
        quartics4 = load_quartics(quartics_dirname+'smquartics4')
        quartics5 = load_quartics(quartics_dirname+'smquartics5')
        job_generator = complete_bipartite_edge_generator(quartics4, quartics5, max_jobs)

    elif experiment == 'small4':
        if generate_quartics:
            quartics, mons4 = generate_small4_quartics()
            save_quartics(quartics_dirname+'small4quartics', quartics, mons4)
            print("Generation finished.")

        quartics = load_quartics(quartics_dirname+'small4quartics')
        job_generator = complete_edge_generator(quartics, max_jobs)

    elif experiment == 'file':
        job_generator = file_edge_generator(USER_EDGES_FILE, max_jobs)

    elif experiment == 'ai_file':
        job_generator = coeff_file_edge_generator(AI_EDGES_FILE, max_jobs)

            
    else:
        raise NotImplementedError
    ##
    if platform.system() == "Darwin":
        from mac_mp_queue import MacQueue
        job_queue = MacQueue()
    else:
        job_queue = mp.Queue()

    return job_generator, job_queue
####

def save_experiment_state():

    time_str = time.asctime().replace(' ', '_')
    namen = ['', time_str]

    lst = []
    while not job_queue.empty():
        lst += [job_queue.get()]

    print("Saving to state files.")
    
    for name in namen:
        save(job_generator, 'process-status/state.job_generator'+name)
        
        #with open('process-status/state.job_generator'+name, 'wb') as F:
        #    pickle.dump(job_generator, F)
        
        with open('process-status/state.job_queue'+name, 'wb') as F:
            pickle.dump(lst, F)

    # Restore the job queue to the original state.
    for x in lst:
        job_queue.put(x)
    
    with open('process-status/state.status', 'w') as F:
        F.write("PAUSED: "+time_str)
        
    return


def send_joblist_to_AI():
    # TODO: This seems to have been superceeded by the "to_AI_pipe" functionality.
    # This function will likely be removed in some future update.
    
    suffix = 'AI'
    time_str = time.asctime().replace(' ', '_')
    namen = ['', time_str]

    lst = []
    while not job_queue.empty():
        pair = job_queue.get()
        f0 = pair[0]
        f1 = pair[1]

        print(job_queue.qsize())
        create_nn_data(suffix, f0, f1)
        lst += [pair]                

    while not job_generator.empty():
        pair = job_generator.next()
        f0 = pair[0]
        f1 = pair[1]

        print (f0,f1)
        nn_data = create_nn_data(suffix, f0, f1)
        write_unlabelled_nn_data(suffix, nn_data)
        # TODO: restore generator state.

    
    # for name in namen:
    #     with open('process-status/state.job_generator'+name, 'wb') as F:
    #         pickle.dump(job_generator, F)
        
    #     with open('process-status/state.job_queue'+name, 'wb') as F:
    #         pickle.dump(lst, F)

    # Restore the job queue to the original state.
    for x in lst:
        job_queue.put(x)
    
    # with open('process-status/state.status', 'w') as F:
    #     F.write("PAUSED: "+time_str)
        
    return


def load_experiment_state():

    job_generator = load('process-status/state.job_generator')
    
    #with open('process-status/state.job_generator', 'r') as F:
    #    job_generator = pickle.load(F)
            
    with open('process-status/state.job_queue', 'rb') as F:
        lst = pickle.load(F)

        if platform.system() == "Darwin":
            from mac_mp_queue.py import MacQueue
            job_queue = MacQueue()
        else:
            job_queue = mp.Queue()

        for x in lst:
            job_queue.put(x)        

    return job_generator, job_queue
##
