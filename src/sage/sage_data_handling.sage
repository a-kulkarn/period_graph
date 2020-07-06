###############################################################################################
# Reduced version of "neural_network/data_handling.py" meant to work with sage/python2
#

import numpy as np
from numpy import genfromtxt
import glob, os
import math
from sys import getsizeof

class DataGroup(dict):
    
    def __init__(self, *args, **kwds):

        filename_by_field = args[0]
        try:
            empty = kwds['empty']
        except KeyError:
            empty = False
            
        if empty == True:
            super(dict, self).__init__()
            for field in filename_by_field.keys():
                dataShape = filename_by_field.shape()
                self[field] = np.empty((0,dataShape[field]))
        else:
            super(dict, self).__init__()
            for field, val in filename_by_field.items():
                self[field] = val

    def concatenate(self, DG):
        for key in self:
            datum = DG[key]
            self[key] = np.concatenate((self[key], datum), axis=0)

    def append(self, DG):
        self.concatenate(DG)

    def data_size(self):
        return sum(A.nbytes for k,A in self.items())

    def truncate_size_to(self, new_size):
        if not self.data_size() <= new_size:
            first_row_size = sum(A[0].nbytes for k,A in self.items())
            num_keep = int(new_size / first_row_size)

            for key in self:
                self[key] = self[key][0:num_keep]


class FilenameByField(dict):
    def __init__(self, raw_fbf_dict, shape):

        self._shape = shape
        super(dict, self).__init__()
        for field, val in raw_fbf_dict.items():
                self[field] = val

    def shape(self):
        return self._shape


    
def read_file_data(filename_tuple, dataShape):
    """
    Read data from a single tuple of file. Each file in the tuple should be associated
    to the same output tuple.

    The parameter subset takes the number of items and outputs a set of indices to select.
    """
    
    fnames = filename_tuple.keys()
    data_gruppe = {field : np.empty((0,dataShape[field])) for field in fnames}

    
    for field in fnames:
        fieldDataFile = filename_tuple[field]
        datum = genfromtxt(fieldDataFile,  delimiter=',', dtype='str')
        
        # Catch misshapings form single line files.
        if len(datum.shape) == 0:
            continue # Empty file, so do nothing.
        elif len(datum.shape) == 1:
            datum = np.array([datum])
            
        data_gruppe[field] = datum

    return DataGroup(data_gruppe)


def BasicSampler(filename_by_field, max_size):
    """
    Read data from all the filenames inside 'filename_by_field', up to the limit specified
    by 'maxsize'.
    """
    fnames = filename_by_field.keys()
    file_list_len  = len(filename_by_field[list(fnames)[0]])
    data_gruppe = DataGroup(filename_by_field, empty=True)
    
    for i in range(file_list_len):

        # Read a single file of data.
        filename_tuple = {field : filename_by_field[field][i] for field in fnames}
        data_gruppe_item = read_file_data(filename_tuple, filename_by_field.shape())
        data_gruppe.concatenate(data_gruppe_item)
        
        # Check the total data size
        if data_gruppe.data_size() >= max_size:
            data_gruppe.truncate_size_to(max_size)
            break
    return data_gruppe

## Outputs for ReadData:
## X is Mx70, Y is Mx3, M01digmat is Mx21x21, M10digmat is Mx21x21.
def ReadData(folder,  dataShape,
             Sampler=BasicSampler, verbose=False):
             

    MAX_INPUT_DATA_SIZE = None
    if MAX_INPUT_DATA_SIZE == None:
        max_size = float('inf')
    else:
        max_size = size_str_to_num(MAX_INPUT_DATA_SIZE)
        print("MaxSize is:   ",max_size, "   bytes.")
        
    ####
    # Arrange filenames into an array
    #

    fnames = list(dataShape.keys())
    field_globs = {field : glob.glob(os.path.join(folder,field))
                   for field in fnames}
    filename_by_field = FilenameByField({field : np.sort(field_globs[field]) for field in fnames},
                                        dataShape)

    if verbose:
        head_globs = field_globs['edgesX-*.csv'][0:10]
        num_other_files  = len(field_globs['edgesX-*.csv']) - len(head_globs)
        print("Input files:")
        print('\n'.join(head_globs))

        if num_other_files > 0:
            msg = ''.join([" ...\n", 9*" ", "...and {} other files.\n".format(num_other_files)])
            print(msg)
        else:
            print("\n")
    
    # total length
    file_list_len  = len(filename_by_field[fnames[0]])
    Yfile_list_len = len(filename_by_field['timingsY-*.csv'])

    if file_list_len == 0:
        error_string = "Input data directory contains no data matching filename specification.\n"
        error_val1 = "INPUT_DIR: {}".format(folder)
        error_val2 = "fnames: {}".format(fnames)
        error_val3 = "glob example: {}".format(os.path.join(folder,"*"+fnames[0]))
        error_post = "Please check if the folder is correct."
        raise RuntimeError('\n'.join([error_string, error_val1, error_val2, error_val3, error_post]))

    # Check if the data is labelled
    is_data_labelled = (not Yfile_list_len == 0)

    if not is_data_labelled:
        fnames.remove('timingsY-*.csv')
        filename_by_field.pop('timingsY-*.csv', None)
        
    ####
    # Reading data
    #
    data_gruppe = Sampler(filename_by_field, max_size)
    
    ####
    # Check data validity. (Check that hashes match.)
    #
    hashes = np.asarray([data_gruppe[field][:,0] for field in fnames], dtype='int')
    if not np.all(np.equal.reduce(hashes)):
        raise RuntimeError("Possible data corruption: hashes do not match.\n"+
                           str(hashes[:,~np.equal.reduce(hashes)]))

    # Clip the hashes off and return.
    for a in data_gruppe:
        data_gruppe[a] = data_gruppe[a][:,1:]
        
    return data_gruppe, is_data_labelled
