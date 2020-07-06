
import queue
import os
PERIOD_SUITE_SAFE_FLAG = ".PERIODSUITE-this-directory-is-safe-to-rm-fr"

def format_magma_args(args):
    return [k+':='+str(args[k]) for k in args]

## 
def construct_all_odes(**kwds):

    while True:
        
        # Check for the terminate signal.
        if quitEvent.is_set():
            return "Quit!"

        # Retrieve an available job from the queue.
        if job_queue.empty():
            return "Quit!"
        
        try:
            e = job_queue.get(timeout=TIMEOUT+1)
        except queue.Empty:
            return "Quit!"
            
        entropy_bias=None
        dirname = edge_ivp_label(e)
        abs_dirname = "{}ode-data/{}/".format(SRC_ABS_PATH, dirname)
        v = e[0].quartic()
        w = e[1].quartic()

        if entropy_bias == None:
            magma_bias_param = '"None"'
        else:
            magma_bias_param = entropy_bias

        # Check for the directory.
        if not os.path.exists(abs_dirname):
            os.mkdir(abs_dirname)
            with open(abs_dirname + PERIOD_SUITE_SAFE_FLAG, 'w') as F:
                F.write('')


        
        # Check if this job or something similar has been attempted.
        args = {'f0':str(v), 'f1':str(w), 'bias':str(magma_bias_param), 'timeout':PHASE_II_ALARM}

        try:
            args['only_first'] = kwds['only_first']
        except KeyError:
            args['only_first'] = False
            
        
        if attempt_already_made('construct_all_odes', abs_dirname, args):
            continue
        
        ## Launch magma to check if the edge should be added to the tree.
        ## Magma will write success data into a special file.
        ## Python writes failure data if the Magma alarm is triggered.
        import subprocess
        try:            
            magma_process = subprocess.Popen(['magma', '-b', 'name:='+dirname] + 
                                              format_magma_args(args) + 
                                              ["magma/transition-homotopy.m"],
                                             stdout=subprocess.PIPE)

            realtime_printer = iter(magma_process.stdout.readline, b'')
            
            # Ping the process as it runs. Also check if the quit event has
            # been set.
            while magma_process.poll() == None:

                for b_line in realtime_printer:
                    line = b_line.decode()
                    if line[0:9] == "AIStream:":
                        with open(abs_dirname + "timings", 'a') as F:
                            F.write(line)
                    else:
                        print(line, end='')
                    
                # Basically, like time.sleep(), but the sleep can be inturrupted.
                if quitEvent.wait(timeout=1):
                    magma_process.terminate()
                    return "Quit!"

            magma_output = magma_process.returncode

            # Write an indicator flag indicating that the file-write is clean.
            if magma_output == 0:
                with open(abs_dirname + "safe_write_flag", 'w') as F:
                    F.write("SAFE\n")

            if magma_output == ALARM_CLOCK_CODE:
                job_done_queue.put([False, e, dirname])
            else:
                job_done_queue.put([True, e, dirname])
        
        except subprocess.CalledProcessError as err:
            logging.error("ERROR: ")
            logging.error(err.output)
            return "ERROR!"

####
