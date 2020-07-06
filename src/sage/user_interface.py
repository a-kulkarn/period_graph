

####
# USER INPUT MANAGEMENT.

import signal
def input_with_timeout():
    try:
        signal.alarm(TIMEOUT)        
        foo = input()
        signal.alarm(0)
        return foo
    except:
        # timeout
        return "TIMEOUT"

####
    
