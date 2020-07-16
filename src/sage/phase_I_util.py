
import subprocess

# Constants
zero_vec = [0 for i in range(35)]
dim_coeff_space = 35
fail_data_string = "30000, 1000, 1000"

# Error codes 
ERROR_CODE = 1
ALARM_CLOCK_CODE = -14

# Data
TRAINING_PATH = os.path.join(SELF_PATH, "training-data", "")

#################################################################################################
## Neural Network data / magma parsing.


# Format the output according to Kat's specification. This being:
#
# QUOTE
# I'd like the data to be formatted in the following way.
# 2+k CSV files representing M data points, where k is the # of cohomology matrices used

# coeff pairs: (Mx70)  X.csv
# times: (1xM) Y.csv
# matrices 1: (Mx441) M1.csv
# ...
# matrices k: (Mx441) Mk.csv

# with each file comma-separated with no extra brackets, like such:
# id, *, *, *, *, ... , *
# id, *, *, *, *, ... , *
# id, *, *, *, *, ... , *
# id, *, *, *, *, ... , *
#
# END QUOTE.

def parse_magma_nn_output(magma_output):
    # NOTE: Every "load" command in magma produces an extra line of print to capture.

    if isinstance(magma_output, bytes):
        magma_output = magma_output.decode()
        
    data_lines = magma_output.replace('[','').replace(']','').split('\n')

    data = []
    for line in data_lines:
        if len(line) < 4:
            continue
        elif line[0:4] == "Load":
            continue
        else:
            data.append(line)

    return data

def parse_edge_traverse_output(magma_output):
    data_lines = magma_output.decode().split('\n')

    timings_data = []
    data_label = ""
    for line in data_lines:
        if line[0:9] == 'AIStream:':
            timings_data += [line[9:]]
        if line[0:10] == 'DataLabel:':
            data_label = line[10:].replace('[','').replace(']','')
    
    return data_label, ','.join(timings_data)

def attach_timings_data(nn_data, timingsX, timingsY):
    return nn_data + [timingsX, timingsY]
    

def write_nn_data(suffix, data, issuccess):

    # 0. Decide on the writing mode (success, fail, None). Note a 'Differentiate Cohomology fail'
    #    is handled separately. `None` represents unlabelled data.

    dirname = TRAINING_PATH
    filenames = ["edgesX-"+suffix+".csv",
                 "DCM01-"+suffix+".csv",
                 "DCM10-"+suffix+".csv"]
    
    if issuccess == None:
        dirname += "edge-data-unlabelled/"
    elif issuccess:
        dirname += "edge-data/"
        filenames.append("partial-timingsX-"+suffix+".csv")
        filenames.append("timingsY-"+suffix+".csv")
    else:
        dirname += "failed-edges/"
        filenames.append("partial-timingsX-"+suffix+".csv")
        filenames.append("timingsY-"+suffix+".csv")

    data_label = str(hash(data[0]))

    assert len(data) == len(filenames)

    for i in range(len(filenames)):
        with open(dirname+filenames[i], 'a') as F:
            F.write(data_label + ', ' + data[i] + '\n')
    return
####

# Alias function.
def write_unlabelled_nn_data(suffix, nn_data):
    write_nn_data(suffix, nn_data, None)


def create_nn_data(suffix, v, w, entropy_bias=None):
    """
    One-off creation of neural network data associated to an edge.
    Note that this function always starts a magma subprocess.

    To batch write several things to the AI pipe, use send_jobs_to_AI
    instead.
    """
    ## Launch magma to create the data for neural-network evaluation.
    ## Magma will write success data into a special file.
    ## Python writes failure data if the Magma alarm is triggered.
    if entropy_bias == None:
        magma_bias_param = '"None"'
    else:
        magma_bias_param = entropy_bias

    magma_output = subprocess.check_output(['magma', '-b',
                                            'suffix:='+"par-run-" + str(suffix),
                                            'f0_vec:='+str(v),
                                            'f1_vec:='+str(w),
                                            'bias:='+str(magma_bias_param),
                                            "magma/create-nn-data-III.m"])

    return parse_magma_nn_output(magma_output)


#################################################################################################
## Main worker function.

def edge_traversable(suffix, v, w, entropy_bias=None):

    ## Launch magma to check if the edge should be added to the tree.
    ## Magma will write success data into a special file.
    ## Python writes failure data if the Magma alarm is triggered.

    #############
    ## Attempt to create neural network data

    try:
        nn_data = create_nn_data(suffix, v, w, entropy_bias=entropy_bias)

    except subprocess.CalledProcessError as e:
        # Uncomment to view raw magma output:
        # print(e.output)

        if e.returncode == ERROR_CODE:
            # The relevant hypersurface was singular.
            # OR, the entropy was larger than the threshold.

            # Log the error for inspection.
            print(e.output)
            logging.info(time.asctime() + '\nCORE: ' + str(suffix) + '\n' + e.output + '\n')

        elif e.returncode == ALARM_CLOCK_CODE:
            # The X-label for the Neural network data failed to compute.
            # This goes into the bin of terrible inputs.
            dcm_fail_path = os.path.join(TRAINING_PATH, "process"+suffix+"DCfail")
            
            with open(dcm_fail_path, 'a') as F:
                F.write((str(v)+', '+str(w)+'\n').translate(None, '[]'))

        return False
    ## End try

    #############
    ## Begin magma process and output capture.

    try:
        magma_bias_param = '"None"' if (entropy_bias == None) else entropy_bias

        # This can be done in real-time, if we were interested in this kinda thing.
        comment_string = """
        magma_process = subprocess.Popen(['magma', '-b',
                                                'suffix:='+"par-run-" + str(suffix),
                                                'f0_vec:='+str(v),
                                                'f1_vec:='+str(w),
                                                'bias:='+str(magma_bias_param),
                                                "magma/attempt-edge-traverse-II.m"],
                                         cwd=SRC_ABS_PATH, stdout=subprocess.PIPE)

        realtime_printer = iter(magma_process.stdout.readline, "")

        # Ping the process as it runs.
        data_label = ""
        timings_data = ""
        while magma_process.poll() == None:
            for line in realtime_printer:
                print line
                if line[0:9] == "AIStream:":
                    timings_data += line.strip()
                elif line[0:10] == "DataLabel:":
                    data_label = line[10:]
                else:
                    print line

                print line, data_label

        retcode = magma_process.returncode
        """

        magma_output = subprocess.check_output(['magma', '-b',
                                                'suffix:='+"par-run-" + str(suffix),
                                                'f0_vec:='+str(v),
                                                'f1_vec:='+str(w),
                                                'bias:='+str(magma_bias_param),
                                                'timeout:='+str(PHASE_I_ALARM),
                                                "magma/attempt-edge-traverse-II.m"], cwd=SRC_ABS_PATH)

        # Add data label to the nn data
        timingsY, timingsX = parse_edge_traverse_output(magma_output)
        nn_data = attach_timings_data(nn_data, timingsX, timingsY)

        # write to success file.
        write_nn_data(suffix, nn_data, True)
        return True

    except subprocess.CalledProcessError as e:
        if not e.returncode == ALARM_CLOCK_CODE:
            raise e

        # Setup failure data write to failure file.
        timingsY, timingsX = parse_edge_traverse_output(e.output)
        timingsY = fail_data_string
        nn_data = attach_timings_data(nn_data, timingsX, timingsY)
        write_nn_data(suffix, nn_data, False)
        return False
        
