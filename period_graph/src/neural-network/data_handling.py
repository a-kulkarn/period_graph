

from NNCONFIG import *

import scipy.linalg
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils import resample
from numpy import genfromtxt
from sklearn.decomposition import PCA
import glob, os
import pickle as pk
import matplotlib.pylab as plt
import math
from time import process_time
from pandas import read_csv
from sys import getsizeof

###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Data handling and preprocessing.

def size_str_to_num(s):
    """
    parses 'maxsize' input
    """
    suffix_one = {'M':10**6, 'G':10**9}
    suffix_two = { x + 'B': suffix_one[x] for x in suffix_one.keys()}

    if s[-1] in suffix_one:
        txtnum = s[0:-1]
        pwr = suffix_one[s[-1]]
        
    elif s[-2:] in suffix_two:
        txtnum = s[0:-2]
        pwr = suffix_two[s[-2:]]

    else:
        raise NotImplementedError
    return eval(txtnum) * pwr


def read_file_data(filename_tuple, dataShape, subsetsz=5000):
    """
    Read data from a single tuple of file. Each file in the tuple should be associated
    to the same output tuple.

    The parameter subset takes the number of items and outputs a set of indices to select.
    """
    
    ##GET A RANDOM SUBSET OF ROWS FROM EACH FILE: THESE ROWS WILL MATCH AMONG X,Y,M1,M2.
    def get_file_len(shortfname):
        datum = read_csv(shortfname, header=None).values
        return len(datum)

    shortfname = filename_tuple['edgesX-*.csv']
    slen = get_file_len(shortfname)
    subs = np.arange(slen)

    np.random.seed(0)
    np.random.shuffle(subs)

    # Debug info
    if False:
        print(filename_tuple)
        print(get_file_len(shortfname), subs)
    
    fnames = filename_tuple.keys()
    data_gruppe = {field : np.empty((0,dataShape[field])) for field in fnames}

    # Only use 'subsetsz' data points; note subsetsz is a number. 
    skipthese = lambda x: x not in subs[:subsetsz]

    
    for field in fnames:
        fieldDataFile = filename_tuple[field]

        # Select the data type depending on if we are reading DCM matrices
        # (with exact rational entries)
        #
        # TODO: (low priority): The 442 is a magic number. The data type could be specified
        #       via the "FilenameByField" object.
        #
        datatype = 'str' if dataShape[field]==442 else float 
        datum = read_csv(fieldDataFile, dtype=datatype, skiprows=skipthese, header=None).values

        # Catch misshapings form single line files.
        if len(datum.shape) == 0:
            continue # Empty file, so do nothing.
        elif len(datum.shape) == 1:
            datum = np.array([datum])
            
        data_gruppe[field] = datum
        # data_gruppe[field] = datum[subset(datum.shape[0])]

    return DataGroup(data_gruppe)


###############################################################################################
# Classes


class DataGroup(dict):
    
    def __init__(self, *args, empty=False):
        filename_by_field = args[0]            
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


class DataSet:
    """
    Class allowing for the manipulation of the dataset as an abstract list of
    filenames. Has methods to actually read in the data.
    """
    def __init__(self, folder, dataShape, ratio, verbose=False):


        ####
        # Arrange filenames into an array
        #
        fnames = list(dataShape.keys())
        field_globs = {field : glob.glob(os.path.join(folder,"**",field), recursive=True)
                       for field in fnames}

        filename_by_field = FilenameByField({field : np.sort(field_globs[field]) for field in fnames}, dataShape)

        if verbose:
            self._print_verbose_reading_info(field_globs)

        # total length
        file_list_len  = len(filename_by_field[fnames[0]])
        Yfile_list_len = len(filename_by_field['timingsY-*.csv'])

        if file_list_len == 0:
            self._raise_no_data_error(folder, fnames)

        # Check if the data is labelled
        is_data_labelled = (not Yfile_list_len == 0)

        if not is_data_labelled:
            fnames.remove('timingsY-*.csv')
            filename_by_field.pop('timingsY-*.csv', None)

        # Initialize the object variables.
        self.filename_by_field = filename_by_field
        self._is_labelled = is_data_labelled
        self._ratio = ratio  # train/test ratio
        #######

        
    def _print_reading_info(self, field_globs):
        head_globs = field_globs['edgesX-*.csv'][0:10]
        num_other_files  = len(field_globs['edgesX-*.csv']) - len(head_globs)
        print("Input files:")
        print('\n'.join(head_globs))

        if num_other_files > 0:
            print(" ...\n", 9*" ", "...and {} other files.\n".format(num_other_files))
        else:
            print("\n")
                
        
    def _raise_no_data_error(self, folder, fnames):
        error_string = "Input data directory contains no data matching filename specification.\n"
        error_val1 = "INPUT_DIR: {}".format(folder)
        error_val2 = "fnames: {}".format(fnames)
        error_val3 = "glob example: {}".format(os.path.join(folder,"*"+fnames[0]))
        error_post = "Please update the NNCONFIG.py file if the folder is incorrect."
        raise RuntimeError('\n'.join([error_string, error_val1,
                                      error_val2, error_val3, error_post]))


    #######
    # Basic attribute access
    #######
    
    def is_labelled(self):
        return self._is_labelled


    #######
    # Internal data reading
    #######
    
    def _read_data(self, filename_by_field, Sampler):
        """
        Main method to read in the data from the filenames specified by
        'filename_by_field', using the 'Sampler'.
        """
        fnames = list(filename_by_field.keys())
        
        if MAX_INPUT_DATA_SIZE == None:
            max_size = math.inf
        else:
            max_size = size_str_to_num(MAX_INPUT_DATA_SIZE)
            print("MaxSize is:   ", max_size, "   bytes.")

        # Reading data
        data_gruppe = Sampler(filename_by_field, max_size)
        self._check_data_validity(data_gruppe)

        # Clip the hashes off and return.
        for a in data_gruppe:
            data_gruppe[a] = data_gruppe[a][:,1:]

        return data_gruppe, self.is_labelled()


    def _check_data_validity(self, data_gruppe):
        fnames = data_gruppe.keys()
        hashes = np.asarray([data_gruppe[field][:,0] for field in fnames], dtype='int')
        if not np.all(np.equal.reduce(hashes)):
            hash_error_msg = str(hashes[:,~np.equal.reduce(hashes)])
            raise RuntimeError("Possible data corruption: hashes do not match.\n"+hash_error_msg)


    #######
    # Data access methods.
    #######
    
    def read_all(self):
        return self._read_data(self.filename_by_field, Sampler=BasicSampler)

    
    def sample_training(self, sampler):
        return self._read_data(self._training_files, Sampler=sampler)
    

    def sample_testing(self, sampler):
        return self._read_data(self._testing_files, Sampler=sampler)

    #######
    # Partitioning
    #######
    
    def partition(self, independent=True):
        """
        Partitions data folders into training and testing. Ratio specifies how large
        each of these are. The independent parameter loosely controls whether the success/fail
        ratio of the whole dataset is reflected in the partition.
        """
        filename_by_field = self.filename_by_field
        fnames = filename_by_field.keys()
        
        file_list_len  = len(self.filename_by_field[list(fnames)[0]])
        num_training   = int(self._ratio * file_list_len)
        
        ##THIS IS A NEW RANDOM SEED. Need this to match between train and test!
        np.random.seed(30)
        randomized_order = np.random.permutation(file_list_len)

        if num_training == 0 or num_training == file_list_len:
            self._raise_bad_partition_error(file_list_len, num_training)
        
        # TODO: Implement the independent selection.
        training_indices = randomized_order[0:num_training]
        testing_indices = randomized_order[num_training:]

        self._training_files = FilenameByField(
            {field : filename_by_field[field][training_indices] for field in list(fnames)},
            filename_by_field.shape())

        self._testing_files = FilenameByField(
            {field : filename_by_field[field][testing_indices] for field in list(fnames)},
            filename_by_field.shape())

        
    def _raise_bad_partition_error(self, file_list_len, num_training):
        msg_template = "Partition of {} files not possible with ratio {}. Results in {} with 0 files."
        bad_set = "training set" if num_training == 0 else "testing set"
        error_msg = msg_template.format(file_list_len, self._ratio, bad_set)
        raise RuntimeError(error_msg)
        
        
###############################################################################################
# Samplers

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


def RandomSampler(filename_by_field, max_size, verbose=False):
    """
    Select a random subset of data from all the filenames inside 'filename_by_field', up to the
    limit specified by 'maxsize'.
    """
    fnames = filename_by_field.keys()
    file_list_len  = len(filename_by_field[list(fnames)[0]])
    data_gruppe = DataGroup(filename_by_field, empty=True)
    
    for i in np.random.permutation(file_list_len):
        filename_tuple = {field : filename_by_field[field][i] for field in fnames}

        # This takes a long time...
        data_gruppe_item = read_file_data(filename_tuple, filename_by_field.shape()) 
        data_gruppe.concatenate(data_gruppe_item)

        # Check the total data size
        if verbose:
            print("before truncating, data_gruppe size was:   ",data_gruppe.data_size(), "   bytes.")
        if data_gruppe.data_size() >= max_size:
            data_gruppe.truncate_size_to(max_size)
            print("after truncating, data_gruppe size is now:   ",
                  data_gruppe.data_size(), "   bytes.")
                
            print("max size allowed was:   ",int(max_size), "   bytes.")
            break
    return data_gruppe


def RandomBalancedSampler(filename_by_field, max_size): # TODO: allow arg passing of ratio
    """
    Select a random subset of data from all the filenames inside 'filename_by_field', up to the
    limit specified by 'maxsize'.
    """
    fnames = filename_by_field.keys()

    # Hardcoded ratio constat
    success_ratio = 0.5
    
    # Divide successes and fails.
    import re
    spat = re.compile('edge-data')
    fpat = re.compile('failed-edges')

    successes = FilenameByField(
        {field : [a for a in filename_by_field[field] if spat.search(a)] for field in fnames},
        filename_by_field.shape())
    
    failures  = FilenameByField(
        {field : [a for a in filename_by_field[field] if fpat.search(a)] for field in fnames},
        filename_by_field.shape())
    
    suc_gruppe  = RandomSampler(successes, max_size * success_ratio, False)
    fail_gruppe = RandomSampler(failures,  max_size * (1-success_ratio), False)
    
    fail_gruppe.concatenate(suc_gruppe)
    
    return fail_gruppe


###############################################################################################
# Read Data

    
def ReformatData(data_gruppe, is_data_labelled, NumMats):

    Y = None
    DCM_stack_list = []
    for a in data_gruppe:
        start = process_time()
        if a[0:8] == "timingsY":
            Y = data_gruppe[a][:,0]
            Y = (Y<1000)*1
            Y = Y.ravel()
            
        elif a[0:3] == "DCM":
            datum = data_gruppe[a]
            mat = np.reshape(datum, (datum.shape[0],21,21))
            mat = np.asarray([MatrixComplexity(m) for m in mat])
            DCM_stack_list += [mat]

        else:
            data_gruppe[a] = np.asarray(data_gruppe[a], dtype='float64')


    mid = process_time()

    if DCM_stack_list != []:
        Ms = np.stack(DCM_stack_list, axis=3)

    # LEGACY: Supports the old 4-5-nomial datasets for backward compatibility.
    if NumMats==1:
        Mss = Ms[:,:,:,0]
        if Ms.shape[3]==1: #if you're only given one cohomology matrix in the file
            ds = int(np.sqrt(dataShape["DCM-*.csv"]-1))
        elif Ms.shape[3]==2: #if you're given two cohomology matrices but only want to use one
            ds = int(np.sqrt(dataShape["DCM01-*.csv"]-1))
        Ms = np.reshape(Mss,(len(Mss),ds,ds,NumMats))
    
    end = process_time()

    print("\n\nReformat time thresh/reshape: ",mid-start," time.")
    print("Reformat time process Ms: ",end-mid," time.\n\n")

    return data_gruppe["edgesX-*.csv"], Y, Ms
    
    
def _reformat_ys(y):
    pass


def _reformat_dcm_matrices(Ms):
    ds = int(np.sqrt(dataShape["DCM01-*.csv"]-1))
    test_M = np.reshape(test_M,(len(test_M),ds,ds,NumMats))
    if NumMats==1: test_M = test_M[:,:,:,0]
    return test_M
    
    
def MatrixComplexity2(M): #total number of digits per element in each matrix
    W = []
    for sample in M:
        Wi=[entry.replace("-","").replace(" ","").split('/') for entry in sample]
        W.append([sum([0 if u=='0' else len(u) for u in w]) for w in Wi])
    return np.array(W)


def ReadDataAndFormat(input_dir, dataShape, NumMats, data_part, ratio,
                      Sampler=BasicSampler, verbose=False):
    start = process_time()
    data_set = DataSet(input_dir, dataShape, ratio, verbose=verbose)
    if data_part == "training":
        data_set.partition() ## KH Moved this 7/19. I think we only want to call this once for the following.
        data_gruppe, is_data_labelled = data_set.sample_training(sampler=Sampler)
    elif data_part == "testing":
        data_set.partition() ## KH Moved this 7/19. I think we only want to call this once for the following.
        data_gruppe, is_data_labelled = data_set.sample_testing(sampler=Sampler)
    elif data_part == "all":
        data_gruppe, is_data_labelled = data_set.read_all()
    else:
        raise ValueError("Invalid value for data_part: {}".format(data_part))
    mid = process_time()
    outdat = ReformatData(data_gruppe, is_data_labelled, NumMats)
    end = process_time()
    print("\n\nReadData takes: ",mid-start," time.")
    print("ReformatData takes: ",end-mid,  " time.\n\n")

    return outdat


#def KH_circumvent(input_dir, dataShape, NumMats, data_part, ratio, Sampler=BasicSampler, verbose=False):
#    #this is just to do the 4nomial case, for KH. different file format.
#    #this is the alternative to ReadDataAndFormat that generated Table 9 in paper.
#
#    data_set = DataSet(input_dir, dataShape, ratio, verbose=verbose)
#    data_gruppe, is_data_labelled = data_set.read_all()
#    for a in data_gruppe:
#        num_samps = len(data_gruppe[a])
#
#    scrambler = KHpermutation()
#    stopat = int(ratio * len(scrambler))
#
#    if data_part == "training":
#        for a in data_gruppe:
#            data_gruppe[a] = data_gruppe[a][scrambler[:stopat]]
#    elif data_part == "testing":
#        for a in data_gruppe:
#            data_gruppe[a] = data_gruppe[a][scrambler[stopat:]]
#
#    outdat = ReformatData(data_gruppe, is_data_labelled, NumMats)
#    return outdat


def PerformPCA(PCAk, train_x):
    print("\n\nSTEP 1 (OPTIONAL): Doing PCA for dimension reduction...")
    pca = PCA(n_components=PCAk)
    pca.fit(train_x)
    print("...singular values of input dataset: \n", pca.singular_values_,"\n")
    #    plt.plot(pca.singular_values_)
    train_x_pca = pca.transform(train_x) #dimension reduced by PCA. First PCAk comp proj.
    return train_x_pca,pca


def UpSampleToBalance(X,y,M):
    print("\n\nSTEP 2 (OPTIONAL): Balancing Dataset...")
    y0 = y.ravel()
    y_succ,y_fail = y[y0==1],y[y0==0]
    X_succ,X_fail = X[y0==1],X[y0==0]
    M_succ,M_fail = M[y0==1],M[y0==0]
    nsamps = np.round(len(y[y0==0])).astype('int')
    # Upsample minority class
    X_succ_upsampled, M_succ_upsampled, y_succ_upsampled \
        = resample(X_succ, M_succ, y_succ, replace=True,\
                   n_samples=nsamps,\
                   random_state=0)
    # Combine majority class with upsampled minority class
    X_upsampled = np.concatenate((X_fail, X_succ_upsampled))
    M_upsampled = np.concatenate((M_fail, M_succ_upsampled))
    y_upsampled = np.concatenate((y_fail, y_succ_upsampled))
    print("***** # successes in BALANCED training set: ",
          np.sum(y_upsampled)," / ",y_upsampled.shape[0]," total training samples.")
    return X_upsampled, y_upsampled, M_upsampled


def MatrixComplexity(M,out="W"): #total number of digits in each matrix
    #out can be "W" or "ND"
    W,N,D = [],[],[]
    char2delete = "-"," ","}","{",'"'
    def replacechar(string):
        for char in char2delete:
            string = string.replace(char,"")
        return string
        
    if out=="W":
        for sample in M:
            Wi=[replacechar(entry).split('/') for entry in sample]
            Wj=[sum([len(u) for u in w]) for w in Wi]
            W.append(Wj)
        return np.array(W)
    else:
        for sample in M:
            Wi=[replacechar(entry).split('/') for entry in sample]
            for w in Wi:
                w = w.append(0) if len(w)==1 else w
            Wj=np.asarray([[int(u) for u in w] for w in Wi])
            N.append(Wj[:,0])
            D.append(Wj[:,1])
        return np.asarray(N),np.asarray(D)
        
        
def MatrixStats(Ns,Ds):
    def kl_divergence(p, q):
        return -np.sum(np.where(p*q != 0, p * np.log(p / q), 0))
    maxM,kldM,sumM,lenM,avgM,entM = [],[],[],[],[],[]
    for N,D in zip(Ns,Ds):
        NpD = N+D
        maxM.append(np.max(NpD))
        kldM.append(kl_divergence(N,D))
        sumM.append(np.sum(NpD))
        lenM.append(len(NpD[NpD>0]))
        avgM.append(np.sum(NpD)/len(NpD[NpD>0])) #average over nonzero elements
        entM.append(scipy.stats.entropy(NpD))
        titles = ["MAXIMUM", "KL DIVERGENCE", "SUM", "LENGTH OF NONZEROS", "AVERAGE OVER NONZEROS", "ENTROPY"]
    return np.asarray([maxM,kldM,sumM,lenM,avgM,entM]),titles
