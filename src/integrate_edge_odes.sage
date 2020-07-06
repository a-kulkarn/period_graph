
import subprocess
from SAGE_CONFIG import *
load(SRC_ABS_PATH + "sage/arg_saver.py")

USER_EDGES_FILE = "user_input/user_edges"


## Note: Sage's timeout mechanism + subprocess + decorator = fail. Work will be done, but
##       no return codes produced. (The failure occurs in the decorator's cleanup).
##       This is why we pass the timeout duties to the subprocess.


# Load dependency
load(SRC_ABS_PATH + "first-stage-analysis.sage")

@parallel(ncpus=60)
def integrate_odes_in_directory(dirname):

    # There is a very annoying issue with the "load" call and variable scope.
    # Basically, load only looks for identifiers defined in global scope.
    #
    # Our current fix is to call sage as a subprocess so that it can have its own "global"
    # scope without subprocesses interfereing with each other.
    #
    
    # Check if this job or something similar has been attempted.
    abs_dirname = SRC_ABS_PATH + dirname
    args = {'timeout':INTEGRATION_ALARM, 'digit_precision':DIGIT_PRECISION}

    if attempt_already_made('integrate_odes_in_directory', abs_dirname, args):
        return 0
    
    try:
        timeout_opt = '--timeout={}'.format(INTEGRATION_ALARM)
        ivpdir_opt  = '--ivpdir={}'.format(dirname)
        prec_opt    = '--digit-precision={}'.format(DIGIT_PRECISION)

        ret_code = subprocess.call(['sage', pathToSuite + 'transition-integrator.sage',
                                    timeout_opt, ivpdir_opt, prec_opt], cwd=SRC_ABS_PATH)

        return ret_code
    except subprocess.CalledProcessError as err:
        # Uncomment to debug sage:
        print(err.output)
        return 1
####

def integration_job_list(**job_config):

    ODE_DATA_DIR = os.path.join(SRC_ABS_PATH, "ode-data", "")
    
    # Create the job list.
    if job_config['generator'] == 'default':
        joblist = ['ode-data/'+dirname+'/' for dirname in os.listdir(ODE_DATA_DIR)]

        # Only take jobs that have the safe write indicator
        joblist = [pth for pth in joblist if os.path.exists(pth+'safe_write_flag')]

    elif job_config['generator'] == 'file':
        joblist = []
        with open(os.path.join(SRC_ABS_PATH, USER_EDGES_FILE)) as F:
            for line in F:
                v,w = line.strip().lstrip('[').rstrip(']').split(',')

                # Need to cast the strings as polynomials to get the correctly
                # sorted terms in the directory name.
                vq = quartic_data(v)
                wq = quartic_data(w)
                dirname = 'ode-data/{}/'.format(edge_ivp_label((vq,wq)))

                if os.path.exists(SRC_ABS_PATH + dirname):
                    joblist.append(dirname)
    else:
        raise ValueError("Invalid option for 'generator': {}".format(job_config['generator']))
                     
    return joblist
###

def _integrate_edge_odes(**job_config):
    # Integrate
    joblist = integration_job_list(**job_config)
    results = integrate_odes_in_directory(joblist)

    old_retc = _load_integration_return_codes()
    _save_integration_return_codes(results, old_retc)
    return


#################################################################
# OSBOLETE: Return code functionality.
#################################################################

def _load_integration_return_codes():
    try:
        retc = load('Z-integration-return-codes.sobj')
        prev_args = [x[0][0][0] for x in retc]
    except IOError:
        retc = []
        prev_args = []
    return retc
    

def _save_integration_return_codes(results, retc):
    # Save the return codes for analysis
    results_list = list(results) + retc
    save(results_list, "Z-integration-return-codes")

