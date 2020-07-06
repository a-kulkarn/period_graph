###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# Utilities for creating and fine-tuning neural networks in Keras.



import os, sys, scipy.io, scipy.linalg, time, random, pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #don't display warnings; only errors
import numpy as np, tensorflow as tf, matplotlib.pylab as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from numpy import genfromtxt
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from time import time, asctime
import pickle as pk

import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import optimizers

from util import *
from model_bundle import *
from data_handling import *


def generate_network_name(reference_network=None):
    # Old names str(int(time()))
    
    if reference_network == None:
        date_components = asctime().replace(':', ',').split()
        return '_'.join(date_components[1:4])
    else:
        return reference_network + '+' + generate_network_name() + '_FT'
    

def finetune_bundle(old_model_bundle, data, EpochNum, BatchSize, StepSizeMLP, StepSizeCNN, Balancing=True):

    train_x,train_y,train_M = data
    if Balancing:
        train_x,train_y,train_M = UpSampleToBalance(train_x,train_y,train_M)

    saved_pca, saved_NN, saved_CN = old_model_bundle.components()
    train_x = saved_pca.transform(train_x)
    
    # Freeze the layers except the last 2 layers
    for layer in saved_NN.layers[:-2]:
        layer.trainable = False
    for layer in saved_CN.layers[:-2]:
        layer.trainable = False

    #[print(layer, layer.trainable) for layer in saved_CN.layers]
    #[print(layer, layer.trainable) for layer in saved_NN.layers]

    bs,ep = BatchSize,EpochNum
    additional_layer,act = 1024,"relu"
    finetuned_NN = MLPFineTune(saved_NN,additional_layer,act,StepSizeMLP)
    finetuned_CN = CNNFineTune(saved_CN,additional_layer,act,StepSizeCNN)

    print("\n\nSTEP 3f: Fine-tuning Filter 1 (MLP using X,Y)... ")
    finetuned_NN.fit(train_x, train_y, batch_size=bs, epochs=ep, verbose=1) # Main MLP-fine-tuning.
    print("        ...done.\n")

    print("\n\nSTEP 4f: Fine-tuning Filter 2 (CNN using X,Y)... ")
    finetuned_CN.fit(train_M, train_y, batch_size=bs, epochs=ep, verbose=1) # Main CNN-fine-tuning
    print("        ...done.\n")

    return ModelBundle(generate_network_name(old_model_bundle.name()),
                       saved_pca, finetuned_NN, finetuned_CN,
                       base_network = old_model_bundle)
####


def train_model_bundle(data, NumMats,
                       DoPCA = True,
                       PCAk = 23,
                       BatchSize = 2000,
                       EpochNum = 100,
                       StepSizeMLP = 1e-5,
                       StepSizeCNN = 1e-5,
                       Balancing = True):

    train_x, train_y, train_M = data
    bs, ep = BatchSize, EpochNum
    emreon = False ##BETA. This is the vector2tensor thing.
    
    if Balancing:
        train_x,train_y,train_M = UpSampleToBalance(train_x,train_y,train_M)

    # Substantial data processing.
    if DoPCA and not emreon:
        train_x,pca = PerformPCA(PCAk, train_x)
    else:
        pca = None

    # ** SUPERVISED: MULTILAYER PERCEPTRON
    print("\n\nSTEP 3: Training Filter 1 (MLP using X,Y)... ")
    
    if emreon:
        print("\n\n********\n\n")
        train_x = np.asarray([vector2tensor(tx) for tx in train_x])
        print(train_x.shape)
    
        print("TESTING: We've replaced the MLP with an edge-CNN!!! This is in beta.")
        NN = CNNClassifier(1,5)
    else:
        hlsizes,numiters,act  = (100,1000,1000,1000,1000,100,100), 100, "relu"
        NN = MLPClassifier0(hlsizes,StepSizeMLP,act,train_x.shape[1])

    NN.fit(train_x, train_y, batch_size=bs, epochs=ep, verbose=1) # Main MLP-Training.
    print("        ...done.")

    # ** SUPERVISED: CONVNET
    # hyperparameters are contained in util.py
    print("\n\nSTEP 4: Training Filter 2 (CNN using M,Y)... ")
    CN = CNNClassifier(NumMats,train_M.shape[1],StepSizeCNN)
    CN.fit(train_M, train_y, batch_size=bs, epochs=ep, verbose=1) # Main CNN-Training
    print("        ...done.\n")

    # ** SAVE WEIGHTS & MODELS
    paramsNN,paramsCN    = [hlsizes,StepSizeMLP,StepSizeCNN,numiters],[bs,ep]

    return ModelBundle(generate_network_name(), pca, NN, CN), paramsNN, paramsCN




###############################################################################################
# Classifier constructors.

def CNNClassifier(k,l,ss):


    model = Sequential()
    # model.add(Conv2D(22, (3, 3), activation='relu',input_shape=(21, 21, k)))
    if l==5:
         model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(l, l, l)))
    elif l==21:
        model.add(Conv2D(64, kernel_size=3, activation='relu',input_shape=(l, l, k)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  #converts 2D feature maps to 1D feature vectors
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=ss, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def MLPClassifier0(hlsizes,ss,act,insz):
    model = Sequential()
    model.add(Dense(hlsizes[0], input_dim=insz, kernel_initializer="uniform", activation = act))
    for i in range(len(hlsizes)-1):
        model.add(Dense(hlsizes[i+1], kernel_initializer="uniform", activation=act))
    model.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))


    print("STEP SIZE IS:         ",ss)

    sgd = optimizers.SGD(lr=ss, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def CNNFineTune(oldmodel,numlay,act,ss):
    model = Sequential()
    model.add(oldmodel)
     
    # Add new layers
    model.add(Dense(numlay, activation=act))
    model.add(Dense(1, activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=ss, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def MLPFineTune(oldmodel,numlay,act,ss):
    model = Sequential()
    model.add(oldmodel)

    # Add new layers
    model.add(Dense(numlay, activation=act))
    model.add(Dense(1, activation='sigmoid'))
    
    sgd = optimizers.SGD(lr=ss, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
