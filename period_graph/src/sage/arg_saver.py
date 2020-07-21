
import os
def attempt_already_made(function_name, dirname, new_args):

    MAKE_ATTEMPT = False
    
    # Construct filename from function and dirname.
    filename = dirname + function_name + '_args.sobj'
    
    special_comparisons = {'construct_all_odes' : construct_all_odes_cmp}
    try:
        old_args = load(filename)
    except IOError:

        # Legacy code to suppose old data. Will be depreciated.
        # print ("\nACHTUNG! Bitte stellen dass alte Daten neu formatiert wurden. "
        #        + "Versuche es trotzdem nochmal...\n")
        
        save(new_args, filename)
        return MAKE_ATTEMPT

    if function_name in special_comparisons:
        comparison_function = special_comparisons[function_name]
    else:
        comparison_function = (lambda x,y : x['timeout'] > y['timeout'])


    # The comparison function should return True if an attempt should be made.
    if comparison_function(new_args, old_args):
        save(new_args, filename)
        return MAKE_ATTEMPT
    else:
        return not MAKE_ATTEMPT


def construct_all_odes_cmp(x,y):
    if x['only_first'] == False and y['only_first'] == True:
        return True
    else:
        return x['timeout'] > y['timeout']
