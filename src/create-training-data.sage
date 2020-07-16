
#######################
## Sage ver. 8.6

# SET CURRENT WORKING DIRECTORY.
import os, sys, getopt
absdirname = os.path.dirname(os.path.abspath(__file__))
os.chdir(absdirname)

from SAGE_CONFIG import *

# CONSTANTS

# Decide whether to randomize roots vertices for trees.
HOUR = 60*60
RANDOM_ROOT_INIT = True
TIMEOUT      = 5
UPDATETIME   = 10
CHECKPTIME   = 10*HOUR
BUFFERSIZE   = 500
LOGNAME      = "process-status/process.log"
STATUSFILE   = "process-status/state.status"
USER_EDGES_FILE = "user_input/user_edges"
AI_EDGES_FILE   = "ai_output"

# Mode constants
MODE_AI_TRAIN = 'AI_train'
MODE_AI_EVAL = 'AI_eval'

HELP_STRING = (
        """        sage graph-manager.sage <MODE=None, AI_train, AI_eval> [--MODE-options] [--help] [--version] [--resume]
        
        MODE OPTIONS:
        sage graph-manager.sage --generator=<name> [--total-jobs=<num>] [--generate-quartics]
        sage graph-manager.sage AI_train --generator=<name> [--training-jobs=<num>] [--total-jobs=<num>] [--generate-quartics]
        sage graph-manager.sage AI_eval  [--total-jobs=<num>]

        Generator OPTIONS:
        'complete4', 'complete5', 'complete4-5', 'small4', 'file', 'ai_file'
        """)


#############################
# JOB CONFIG (should perhaps be user input)


# Default options
myargv = sys.argv[1:]
MODE = None
MODE_ALLOWED_OPTS = ['-h', "--generator", "--total-jobs", "--generate-quartics", "--help", "--version", "--resume", "--job-interval"]

# Check for a special mode.
if len(sys.argv) == 1:
    pass

elif sys.argv[1] == 'AI_train':
    myargv = sys.argv[2:]
    MODE = MODE_AI_TRAIN
    MODE_ALLOWED_OPTS = ['-h', "--generator", "--training-jobs", "--total-jobs", "--generate-quartics", "--help", "--version", "--resume"]
    
elif sys.argv[1] == 'AI_eval':
    myargv = sys.argv[2:]
    MODE = MODE_AI_EVAL
    MODE_ALLOWED_OPTS = ['-h', "--total-jobs", "--help", "--version", "--resume"]

elif not (sys.argv[1][0:2] == '--' or sys.argv[1][0] == '-'):
    print(sys.argv[1][0])
    print("ERROR: MODE '{}' not supported.".format(sys.argv[1]))
    sys.exit(1)
else:
    myargv = sys.argv[1:]

# Parse the input configuration.
opts, args = getopt.getopt(myargv, "h", ["generator=", "training-jobs=", "total-jobs=", "generate-quartics", "help", "version", "resume", "job-interval="])


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
    print("graph-manager version 0.1, Release date: Never!")
    sys.exit()

# --resume overrides other config options.
if '--resume' in opt_names:
    resume = True
else:
    resume = False


# Set the job config via parsing the other options.
job_config = {'max_jobs':"default"}

# Set the default option.
if MODE == MODE_AI_EVAL:
    job_config['experiment'] = 'ai_file'
    job_config['generate_quartics'] = False
else:
    job_config['experiment'] = 'file'
    job_config['generate_quartics'] = False

job_config['job_interval'] = "default"


def parse_job_interval_string(arg):
    items = arg.strip('[]').split(',')
    int_list = [int(x) for x in items]

    if not len(int_list) == 2:
        raise ValueError("Invalid length for job interval.")
    if int_list[0] < 0:
        raise ValueError("First index for job cannot be less than 0.")
    if int_list[0] > int_list[1]:
        raise ValueError("Last index for job cannot be less than first index.")

    return int_list
####


for opt, arg in opts:
    if not opt in MODE_ALLOWED_OPTS:
        if MODE == None:
            MODE = "<None>"
            print("Option '{}' not allowed in mode '{}'.".format(opt, MODE))
            sys.exit(1)

    if opt == "--training-jobs":
        print("WARNING: right now training jobs is just an alias for max-jobs. However, "+
              "more sophisticated logic may be implemented in the future.")

        job_config['max_jobs'] = int(arg)

    elif opt == "--total-jobs":
        job_config['max_jobs'] = int(arg)

    elif opt == "--generate-quartics":
        job_config['generate_quartics'] = True

    elif opt == "--generator":
        job_config['experiment'] = arg

    elif opt == "--job-interval":
        job_config['job_interval'] = parse_job_interval_string(arg)

# load the user terminate function. This is a function of one variable (a graph).
# The function can depend on user-defined constants.
load("first-stage-analysis.sage")
load("user_input/user_exit_functions.sage")


#############################

import re
import multiprocessing as mp
import queue
import time
import pickle

# Make sure that errors in the subprocess are raised properly.
import traceback
import logging
logging.basicConfig(filename='process-status/error.log', level=logging.INFO)


## Load dependencies.
load("sage/user_interface.py")
#load("sage/period-tree.py")
load("sage/phase_I_util.py")
load("sage/quartic_utilities.sage")
load("sage/job_generators.sage")
load("sage/experiment_state_util.sage")
load("sage/worker_functions.sage")


#### MAIN PROCESS START #####

if __name__ == '__main__':

    print("Beginning computation...")

    with open(STATUSFILE, 'w') as F:
        F.write('STARTED...\n' + time.asctime() + '\n')

    with open(LOGNAME, 'a') as F:
        F.write(time.asctime() + "\nJOB STARTED...\n")


    ###########################################
    # Set up the job queue
    
    if resume:
        job_generator, job_queue = load_experiment_state()
    else:
        job_generator, job_queue = initialize_experiment_data(**job_config)


    # Activate the hack. I make no guarantees regarding sefe usage.
    if not job_config['job_interval'] == "default":
        job_generator.set_job_interval(job_config['job_interval'])
        
    top_up_queue(job_generator)

    ###########################################
    # Configure parallelization
    
    quitEvent = mp.Event()
    num_cores = mp.cpu_count() - 1 # one for running an external process.

    explored_graph = Graph()

    # Queue structure for the file names.
    suffix_queue = mp.Queue()
    for x in range(num_cores+2):
        suffix_queue.put(str(x))
        
    worker_processes = [ mp.Process(target=edge_check_worker) for suffix in range(num_cores)]
    # AI_process = mp.Process(target="whatever function")

    # Launch!
    for WP in worker_processes:
        WP.start()
    
    ############################################
    # Display input for user while waiting for children processes to terminate. 
    time.sleep(2)
    print("All cores running. Awaiting data...")
    print("Press enter to terminate process: \n")
    
    tstart  = time.time()
    tcheckp = tstart
    try:
        while not user_stop_function(explored_graph) and (
                any(worker.is_alive() for worker in worker_processes)):
            
            # Possibly change job generator, or insert specific jobs into the queue first.

            top_up_queue(job_generator) # Techically a race condition.
            if not input_with_timeout() == "TIMEOUT":
                break

            # Add edges to the explored graph
            if CONSTRUCT_GRAPH == True:
                # NOTE: this could lead to the same file being open in
                # two separate processes. The code below is read-only, so
                # nothing too horrible should occur...maybe...
                load("first-stage-analysis.sage")

                # User defined exit condition, given as a function.
                # if user_stop_function(G):
                #     break


            # Launch subprocess to train AI based on data.
            # AI_process.start()

            # Print at regular intervals
            if time.time()-tstart >= UPDATETIME:
                remaining_jobs = job_queue.qsize() + job_generator.remaining()
                print("Estimated number of remaining jobs: {}.".format(remaining_jobs) +
                      " Press enter to terminate process.")

                with open(LOGNAME, 'a') as F:
                    F.write(time.asctime() + '\n')
                    F.write("Jobs remaining: {}\n".format(remaining_jobs))

                tstart = time.time()
            ##

            # Save a checkpoint every once in a while.
            if time.time()-tcheckp >= CHECKPTIME:
                save_experiment_state()
                tcheckp = time.time()
            
    except Exception as e:
        traceback.print_exc()
        logging.error(str(traceback.print_exc()))

        with open(STATUSFILE, 'w') as F:
            F.write(time.asctime() + '\nERROR\n')

        raise e
    finally:
        print("Sending terminate signal to children. Please wait about 30 seconds...")
        quitEvent.set()

    while any( worker.is_alive() for worker in worker_processes):
        print("waiting...")
        time.sleep(10)

    print("Children processes finished. Exiting...")
    
    ############################################
    ## Save the output of the process if needed.
    
    remaining_jobs = job_queue.qsize() + job_generator.remaining()
    if not remaining_jobs == 0:
        save_experiment_state()

        if MODE == MODE_AI_TRAIN:
            send_joblist_to_AI()

    else:
        with open(STATUSFILE, 'w') as F:
            F.write("FINISHED\n" + time.asctime() + "\n")

        with open(LOGNAME, 'a') as F:
            F.write(time.asctime() + "\nJOB FINISHED\n")

        
    #TODO: Actually parse the training jobs option.
    print("All data saved.")

    
# Force Sage to die. Painfully.
# (If the interupt feature is used, the script will not exit for some reason.)
os._exit(0)

####
