
#######################
## Sage ver. 8.6

# SET CURRENT WORKING DIRECTORY.
import os, sys, getopt
os.chdir(sys.path[0])

# CONSTANTS

# Decide whether to randomize roots vertices for trees.
RANDOM_ROOT_INIT = True
TIMEOUT      = 5
UPDATETIME   = 10
BUFFERSIZE   = 500
LOGNAME      = "process-status/process.log"
STATUSFILE   = 'process-status/state.status'
USER_EDGES_FILE = "user_input/user_edges"

HELP_STRING = (
        """        sage construct-edge-odes.sage [--MODE-options] [--help] [--version]
        
        MODE OPTIONS:
        sage construct-edge-odes.sage --generator=<name> [--total-jobs=<num>]

        Generator OPTIONS:
        'file'
        """)


##########################################################
# JOB CONFIG 

# Parse the input configuration.
myargv = sys.argv[1:]
opts, args = getopt.getopt(myargv, "h", ["generator=", "total-jobs=", "help", "version",
                                         "forgo-manifest", "only-first"])
    
# Check to make sure nothing bad happened.
if not args == []:
    print("ERROR: options misinterpreted as arguments. Please check the input.")
    sys.exit(1)

# Check for the high priority options
opt_names = [o[0] for o in opts]

if '--help' in opt_names or '-h' in opt_names:
    print(HELP_STRING)
    sys.exit()

elif '--version' in opt_names:
    print("construct-edge-odes.sage version 0.1, Release date: Never!")
    sys.exit()


# Set the job config via parsing the other options.
job_config = {'max_jobs':"default"}

# Set the default option.
job_config['generator'] = 'default'
job_config['manifest'] = True
construct_odes_kwds = {}

for opt, arg in opts:
    if opt == "--total-jobs":
        job_config['max_jobs'] = int(arg)

    elif opt == "--generator":
        job_config['generator'] = arg

    elif opt == "--forgo-manifest":
        job_config['manifest'] = False

    elif opt == "--only-first":
        construct_odes_kwds['only_first'] = True
        
#############################

import re
import multiprocessing as mp
import time
import pickle

# Make sure that errors in the subprocess are raised properly.
import traceback
import logging
logging.basicConfig(filename='error.log', level=logging.ERROR)


## Load dependencies.
load("sage/user_interface.py")
load("sage/period-tree.py")
load("sage/phase_I_util.py")
load("sage/quartic_utilities.sage")
load("sage/job_generators.sage")
load("sage/experiment_state_util.sage")
load("sage/worker_functions.sage")
load("sage/arg_saver.py")

## Load user exit functions
load("first-stage-analysis.sage")
load("user_input/user_exit_functions.sage")

# Load main worker functions.
load("sage/phase_II_util.py")


#### MAIN PROCESS START #####
                           
if __name__ == '__main__':

    print("Beginning computation...")

    with open(STATUSFILE, 'w') as F:
        F.write('STARTED...\n' + time.asctime() + '\n')

    with open(LOGNAME, 'a') as F:
        F.write(time.asctime() + "\nJOB STARTED [All odes]...\n")


    ###########################################
    # Set up the job queue
    
    # Initialize exit condition graph.
    exit_graph = DiGraph({})

    if job_config['manifest']:
        insert_manifest_edges(exit_graph)

    gen_arg = job_config['generator']
    if gen_arg == 'default':
        # Load the list of potential jobs.
        G = load_phase_I_graph()
        #G = load('phaseIgraphcache')

        job_generator = graph_edge_generator(G)
        
    elif gen_arg == 'file':
        
        # Wrap the job generator so that next() returns quartic data objects.
        class phaseII_gen_wrapper(file_edge_generator):
            def next(self):
                e = super(phaseII_gen_wrapper, self).next()
                return [quartic_data(e[0]), quartic_data(e[1])]

        job_generator = phaseII_gen_wrapper("user_input/user_edges")
        
    elif gen_arg == 'special':
        # Load the list of potential jobs.
        G = load_phase_I_graph()
        #G = load('phaseIgraphcache')
        
        job_generator = special_iterator(G, exit_graph)
        
    else:
        raise ValueError("Invalid arg for generator. Given: {}".format(gen_arg))

    # Check the operating system and import a portable queue class if needed.
    import platform
    if platform.system() == "Darwin":
        load("sage/mac_mp_queue.py")
        job_queue = MacQueue()
        job_done_queue = MacQueue()
    else:
        job_queue = mp.Queue()
        job_done_queue = mp.Queue()

    # job_generator, job_queue = initialize_experiment_data(**job_config)
    # job_generator, job_queue = load_experiment_state()

    ###########################################
    # Configure parallelization
    
    quitEvent = mp.Event()
    num_cores = mp.cpu_count() - 4   
    # num_cores = 1 # Limiter for debugging.

    # Queue structure for the file names.
    suffix_queue = mp.Queue()
    for x in range(num_cores+2):
        suffix_queue.put(str(x))
        
    worker_processes = [mp.Process(target=construct_all_odes, kwargs=construct_odes_kwds)
                        for suffix in range(num_cores)]

    # HACK: A different top-up function to prevent computing extra edges.
    def top_up_queue(job_generator):
        while job_queue.qsize() <= 1000:
            try:
                e = job_generator.next()
                job_queue.put(e)
            except StopIteration as e:
                return
            
    top_up_queue(job_generator)
    
    # Launch!
    for WP in worker_processes:
        WP.start()
    
    ############################################
    # Display input for user while waiting for children processes to terminate. 
    time.sleep(2)
    print("All cores running. Awaiting data...")
    print("Press enter to terminate process: \n")

    tstart = time.time()
    try:
        while (not user_stop_function_II(exit_graph) and
               any(worker.is_alive() for worker in worker_processes)):
            
            top_up_queue(job_generator)
            if not input_with_timeout() == "TIMEOUT":
                break

            # Print at regular intervals
            if time.time()-tstart >= UPDATETIME:
                remaining_jobs = job_queue.qsize() + job_generator.remaining()
                print("Estimated number of remaining jobs: {}.".format(remaining_jobs) +
                      " Press enter to terminate process.")

                with open(LOGNAME, 'a') as F:
                    F.write(time.asctime() + '\n')
                    F.write("Jobs remaining: {}\n".format(remaining_jobs))

                # Add to the exit graph and check the exit condition.
                while CONSTRUCT_GRAPH and not job_done_queue.empty():
                    res = job_done_queue.get()

                    if res[0] == True:
                        e = res[1]
                        dirname = res[2]
                        exit_graph.add_edge(e[0], e[1], dirname)

                # The early-exit condition
                # if user_stop_function_II(exit_graph):
                #     for WP in worker_processes:
                #         WP.terminate()
                #     break
                
                tstart = time.time()
            ##

    except Exception as e:
        traceback.print_exc()
        logging.error(str(traceback.print_exc()))

        with open(STATUSFILE, 'w') as F:
            F.write(time.asctime() + '\nERROR\n')

        raise e
    finally:
        print("Sending terminate signal to children. Please wait about 30 seconds...")
        quitEvent.set()

    wait_count = 0
    while any(worker.is_alive() for worker in worker_processes):
        print("waiting...")
        time.sleep(10)
        wait_count += 1

        # Force quit.
        if wait_count > 3:
            for WP in worker_processes:
                WP.terminate()
                

    ############################################
    ## Save the output of the process if needed.
    
    remaining_jobs = job_queue.qsize() #+ job_generator.remaining()
    if not remaining_jobs ==0:
        pass
        # TODO: Enable experiment state saving.
        #save_experiment_state()
    else:
        with open(STATUSFILE, 'w') as F:
            F.write("FINISHED\n" + time.asctime() + "\n")


        with open(LOGNAME, 'a') as F:
            F.write(time.asctime() + "\nJOB FINISHED\n")
        
    print("Children processes finished. Exiting...")

    # NOTE: The empty() method should be fine with just one core running, but there
    #        might be something safer.

    # Final construction of the exit graph.
    while not job_done_queue.empty():
        res = job_done_queue.get()
        
        if res[0] == True:
            e = res[1]
            dirname = res[2]
            exit_graph.add_edge(e[0], e[1], dirname)

    # Print an awk-able vertex manifest as a comma separated list (Use FS=',' option in awk).
    with open("edge-manifest", "a") as F:
        for e in exit_graph.edges():
            if not e[2] == None:
                F.write(str(e[0]) + ',' + str(e[1]) + ',' + e[2] + '\n')


# Force Sage to die. Painfully.
# (If the interupt feature is used, the script will not exit for some reason.)
os._exit(0)
####
