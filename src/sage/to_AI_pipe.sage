
import os
def send_jobs_to_AI(E):
    """
    Sends the list of edges `E`, represented by pairs of 35-vectors, to the
    AI input file for *future* ranking. No output it returned, and the AI
    process is not run.
    """

    TIMEOUT = 5
    old_dir = magma.eval("GetCurrentDirectory()")
    
    # Use sage's builtin magma spawn with a timeout wrapper to
    # Send jobs to the AI.

    def initialize_magma():
        magma.eval('ChangeDirectory("{}")'.format(SRC_ABS_PATH))
        magma.load("magma/create-nn-data-II.m")

    initialize_magma()

    # Clean data out of the directory.
    for fname in ["DCM01-AI.csv", "DCM10-AI.csv", "edgesX-AI.csv"]:
        dir_plus_file = SRC_ABS_PATH + "edge-data-unlabelled/" + fname
        if os.path.exists(dir_plus_file):
            os.remove(dir_plus_file)
        
    for e in E:
        f0 = e[0]
        f1 = e[1]
        
        # DONT use signal handler to externally manage the alarm. Sage will crash.
        try:
            # One can launch/cancel the alarm here, and check for a crash.
            line1 = "Alarm({});".format(DCM_ALARM)
            line2 = 'DCMData("{}","{}","{}","\\"None\\"");'.format("AI",f0,f1)
            line3 = "Alarm(0);"
            output = magma.eval("{}\n{}\n{}\n".format(line1,line2,line3))

            # Detect a crash and restart. Skip this edge in the return data.
            if output == '':
                magma.eval()
                initialize_magma()
            else:
                nn_data = parse_magma_nn_output(output)
                write_unlabelled_nn_data("AI", nn_data)
            
        except RuntimeError as rexecption:
            print(rexecption)
            print('\n')
        finally:
            magma.eval("Alarm(0);")
        
    magma.eval('ChangeDirectory("{}")'.format(old_dir))
    return
