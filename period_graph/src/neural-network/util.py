###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Utilities for:
#   (1) Analyzing results.
#   (2) Writing/Printing results.



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
from sys import getsizeof


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Analytics.

def clusteringResults(yGuess,yTrue,titleString):
    err = yGuess - yTrue;
    fps = np.count_nonzero(err == 1);
    trs = np.count_nonzero(err == 0);
    fns = np.count_nonzero(err == -1);
    print(
          "\n"+titleString+
          "\n     % false positives: "+str(100*fps/err.shape[0])+"%,"+
          "\n     % false negatives: "+str(100*fns/err.shape[0])+"%,"+
          "\n     % correct guesses: "+str(100*trs/err.shape[0])+"%.\n")
    return err


import collections
def OptimalROCSup(p_predict, y_true, ttl, zepplin=False):
    """
    Return the *supremum* of the T such that the ROC distance is optimal.
    Note that as the aforementioned T is never actually in this set.
    """
    fpr, tpr, thresholds = roc_curve(y_true,p_predict,pos_label=1)
    plt.plot(fpr,tpr,label=ttl)
    plt.title("ROC curves.")
    roc_dist = tpr**2+(1-fpr)**2
    T_best = np.argmax(roc_dist)
    
    if zepplin:
        print("Been dazed and confused for so long it's not true!")
        
    return thresholds[T_best]
    
    
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Saving to files.

def WritePlots(NN_PATH,uniquenum,pNN,pCN,pEN,test_y):
    fname = os.path.join(NN_PATH,"Plot"+uniquenum)
    
    
    plt.figure()
    _ = plt.hist(pNN, bins=10, alpha=0.7, label="MLP")  # arguments are passed to np.histogram
    _ = plt.hist(pCN, bins=10, alpha=0.7, label="CNN")  # arguments are passed to np.histogram
    plt.title("Histograms of SUCC/FAIL probabilities on Test")
    plt.legend(prop={'size': 10})
    plt.savefig(fname + "_1.png")

    fig=plt.figure()
    fig.suptitle("Histograms of SUCC/FAIL probabilities on TEST")
    ax1=plt.subplot(1, 2, 1)
    ax2=plt.subplot(1, 2, 2)
    _ = ax1.hist(pNN[test_y==1], bins=10, alpha=0.7, label="MLP")
    _ = ax1.hist(pCN[test_y==1], bins=10, alpha=0.7, label="CNN")
    plt.legend(prop={'size': 10})
    ax1.set_title('Test TRUE SUCCESSES')
    _ = ax2.hist(pNN[test_y==0], bins=10, alpha=0.7, label="MLP")
    _ = ax2.hist(pCN[test_y==0], bins=10, alpha=0.7, label="CNN")
    plt.legend(prop={'size': 10})
    ax2.set_title('Test TRUE FAILURES')
    plt.savefig(fname + "_2.png")

    fig = plt.figure()
    plt.plot(pEN, label = "Ensemble Prob")
    plt.plot(test_y, label = "True Prob")
    plt.legend(prop={'size': 10})
    plt.savefig(fname + "_3.png")
    
    fig = plt.figure()
    plt.plot(pEN[test_y==1], label = "Ensemble Prob")
    pdfT = test_y[test_y==1].astype(float)
    pdfT /= max(pdfT)
    plt.plot(pdfT, label = "True Prob")
    plt.legend(prop={'size': 10})
    plt.savefig(fname + "_4.png")

    # plt.show()
    return

def ThresholdProbs(vec,t):
    tvec = np.zeros((vec.shape))
    tvec[vec > t]  = 1
    return tvec

def WriteConfusion(NN_PATH,uniquenum,paramsNN,paramsCN,test_y,yNN,yCN,yEN):

    fname = os.path.join(NN_PATH, "EvalOutputs/ConfMats"+uniquenum+".txt")
    with open(fname,"w+") as f:
        #####
        f.write("\n\n*****************\n\nFilter 1 (ReLU MLP) Params:\n")
        if len(paramsNN)==1:
            #loading a pre-trained model
            B = "Loading from pre-trained MLP model: " + str(paramsNN)
        else:
            strg = ["\nWidth of each hidden layer:   ",
                    "\nRegularization penalty parameter:   ",
                    "\nNumber of training iterations:   "]
            
            B = [s+str(n) for s,n in list(zip(strg,paramsNN))]
        f.write(''.join(B))
        f.write("\n\nConfusion Matrix for MLP:\n")
        f.write(str(confusion_matrix(test_y,yNN)))

        #####
        f.write("\n\n*****************\n\nFilter 2 (CNN) Params:\n")
        if len(paramsCN)==1:
            #loading a pre-trained model
            B = "Loading from pre-trained CNN model: " + str(paramsCN)
        else:
            strg = ["\nBatch size for training:   ",
                    "\nEpoch length:   ",
                    "\nBatch size for testing:   "]
            B = [s+str(n) for s,n in list(zip(strg,paramsCN))]

            
        f.write(''.join(B))
        f.write("\n\nConfusion Matrix for CNN:\n")
        f.write(str(confusion_matrix(test_y,yCN)))
        f.write("\n\n*****************\n\nConfusion Matrix for Ensemble:\n")
        f.write(str(confusion_matrix(test_y,yEN)))
        f.write("\n\n*****************\n")
        
    print("\nConfusion matrices written to:   ", fname, "\n")
    return


def _WriteTable9Data(fname, MB, ttratio, confusion_mat):
    """
    Writes to a file specified by 'fname'. Each row of the file is of the form

        <model id>, <train-test ratio (alpha)>, [C.ravel()], TP+TN/(FP+FN)

    where 'C' is the confusion matrix of the ensemble network. Table 9 refers to the table
    in the article accompanying this software.
    """
    with open(fname, 'a') as F:
        A = confusion_mat
        conf_rat = (A[0,0] + A[1,1])/(A[0,1] + A[1,0])
        line = ', '.join([MB.name(), str(ttratio), str(A.ravel()), str(conf_rat)])
        F.write(line+'\n')
        
    return

def WritePredictions(NN_PATH, INPUT_DIR, rand_seed, readnum, uniquenum, datain):

    # Internally keep training transcripts, for future analysis.
    transcript_directory = os.path.join(NN_PATH, "EvalOutputs", '') #save files locally

    pCN, rCN, pNN, rNN, pEN, rEN = datain
    print(rNN.shape, rCN.shape)
    
    file_data  = [pCN, rCN, pNN, rNN, pEN, rEN]
    file_names = [
        "ProbabilitiesNN{}.txt", "RankedCoefsNN{}.txt",
        "ProbabilitiesCN{}.txt", "RankedCoefsCN{}.txt",
        "ProbabilitiesEN{}.txt", "RankedCoefsEN{}.txt"]
    
    true_file_names = [name.format(uniquenum) for name in file_names]

    for i in range(len(true_file_names)):
        np.savetxt(transcript_directory+true_file_names[i],file_data[i])
        
    # Save the name of the network to refer to later.
    with open(transcript_directory+"EvalParams{}.txt".format(uniquenum), 'w+') as F:
        F.write("\nModel identifier:  " + str(readnum) +"\n")
        F.write("Model folder:      " + str(NN_PATH)  +"\n")
        F.write("Eval Data folder:  " + str(INPUT_DIR)+"\n")
        F.write("Random seed:       " + str(rand_seed)+"\n")
        F.write("Eval identifier:   " + str(uniquenum)+"\n")
    
    # Print a status update:
    print("\nProbabilities & Rankings written to: ")
    [print(tfn) for tfn in true_file_names]
    print("\n")
    return
    
def WritePredictionsToSagePipe(NN_PATH, datain):
    
    output_directory = os.path.join(NN_PATH,"..",'') # for the ai_output file (global)
    pCN, rCN, pNN, rNN, pEN, rEN = datain
    print(rNN.shape, rCN.shape)
        
    temp = np.array2string(rEN, separator=',', threshold=1e32, max_line_width=1e32)[1:-1]
    ensemble_sorted_edges = temp.replace("],","]").replace(".","").replace(" [","[")

    
    # Write the output for reading by the ODE-computation process.
    #np.savetxt(output_directory+"ai_output", rEN.astype(int), fmt="%1i")
    with open(output_directory+"ai_output", "w+") as f:
        f.write(ensemble_sorted_edges)

    # Also write the probability vectors to output.
    np.savetxt(output_directory+"ai_probabilities", np.stack((pEN, pNN, pCN), axis=1))
    return

def ReadPredictions(NN_PATH, uniquenum):
    transcript_directory = os.path.join(NN_PATH, "EvalOutputs", '')
    file_names = [
        "ProbabilitiesNN{}.txt", "RankedCoefsNN{}.txt",
        "ProbabilitiesCN{}.txt", "RankedCoefsCN{}.txt",
        "ProbabilitiesEN{}.txt", "RankedCoefsEN{}.txt",
        ]
        
    with open(transcript_directory+"EvalParams{}.txt".format(uniquenum), 'r') as F:
        NetNum = F.read()

    true_file_names = [name.format(uniquenum) for name in file_names]
    file_data = [np.loadtxt(transcript_directory+tfn, dtype=float) for tfn in true_file_names]
    file_data += [NetNum]
    
    return file_data
    
def PrintConfusion(test_y, yNN, yCN, yEN, show_legend=True):
    if show_legend:
        print("\n*********************")
        print(" LEGEND:\n")
        print(" The entries of the confusion matrix C_{i,j} are the number of objects ")
        print(" with label 'i' assigned label 'j'. In this case, the first row corresponds ")
        print(" to 'failed' labels.")
        
    print("\n*********************")
    print("\n Confusion Matrix, Filter 1 (MLP):\n",    confusion_matrix(test_y,yNN))
    print("\n Confusion Matrix, Filter 2 (CNN):\n",    confusion_matrix(test_y,yCN))
    print("\n Confusion Matrix, given BOTH filters:\n",confusion_matrix(test_y,yEN))
    print("\n*********************\n")

def discrete_matshow(data):
#    #get discrete colormap
#    cmap = plt.get_cmap('YlGnBu', np.max(data)-np.min(data)+1)
#    # set limits .5 outside true range
#    mat = plt.imshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
#    #tell the colorbar to tick at integers
#    cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))
#    #plt.show()
    
    fig,a =  plt.subplots(2,1)
    for i in [0,1]:
        d = data[:,:,i]
        cmap = plt.get_cmap('Blues', np.max(d)-np.min(d)+1)
        mat = a[i].imshow(d,cmap=cmap,vmin=0,vmax=5)#vmin = np.min(d)-.5, vmax = np.max(d)+.5)
        a[i].axis('off')
#        cax = plt.colorbar(mat,ax=a[i])#, ticks=np.arange(0,7))#np.min(d),np.max(d)+1))

