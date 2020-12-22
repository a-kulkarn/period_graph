
import os
import pickle as pk
from keras.models import load_model

class trivialPCA:
    def __init__(self):
        pass
    
    def transform(self, x):
        return x

class ModelBundle:

    def __init__(self, *args, **kwds):

        if len(args) == 1:
            model_id = args[0]
            PCA, MLP, CNN = None, None, None
        elif len(args) == 4:
            model_id, PCA, MLP, CNN = args
        else:
            raise NotImplementedError

        try:
            self.base_network_name = kwds['base_network'].name()
        except KeyError:
            self.base_network_name = None
                
        self.model_id = model_id
        self.PCA = PCA
        self.MLP = MLP
        self.CNN = CNN

    def name(self):
        return self.model_id

    def components(self):
        return self.PCA, self.MLP, self.CNN
    
    def _load(self, path):
        spath = os.path.join(path, self.model_id, '')

        try:
            self.PCA = pk.load(open(spath+'PCs'+ self.model_id +'.pkl','rb'))
        except IOError:
            self.PCA = trivialPCA()
            
        self.MLP = load_model(spath+'MLP'+self.model_id+'.h5')
        self.CNN = load_model(spath+'CNN'+self.model_id+'.h5')

    
    def save(self, path, also_to_newest=False):
        
        if also_to_newest:
            names = [self.model_id, "_newest"]
        else:
            names = [self.model_id]

        for name in names:
            spath = os.path.join(path, name, '')

            try:
                os.mkdir(spath)
            except FileExistsError:
                pass
            
            pk.dump(self.PCA, open(spath+'PCs'+ name +'.pkl',"wb"))
            self.MLP.save(spath+'MLP'+name+'.h5')
            self.CNN.save(spath+'CNN'+name+'.h5')

    def save_parameters(self, path, setup_dic, params_dic, also_to_newest=False):

        # IDEA: Model bundles should perhaps have a header where a dictionary of
        # some of these params are kept (such as if it is a finetuned model).

        # IDEA: Model bundles could also pickle the parameter dictionary for later.
        
        if also_to_newest:
            names = [self.model_id, "_newest"]
        else:
            names = [self.model_id]

        for name in names:

            fname = os.path.join(path, name, "Params"+name+".txt")
            with open(fname,"w+") as f:
                f.write("\n*****************\n")
                f.write("Network Training Params for '{}'".format(self.model_id))
                f.write("\n\n")
                
                # Print key-value pairs according to special formatting instructions
                # Determined by dictionary keys.

                B = ["Base network (None if new): " + str(self.base_network_name),
                     "",
                     "Setup parameters:",
                     tall_dic_str(setup_dic),
                     "", 
                     "Network architecture hyperparameters:",
                     tall_dic_str(params_dic), "\n"]

                f.write('\n'.join(B))
                
                # strg = ["Network permanent name:             ",
                #         "Fresh network? Else finetuned:      ",
                #         "New network name:                   ",
                #         "Num cohomology matrices / pair:     ",
                #         "Balance the training set:           ",
                #         "PCA preprocessing with 23 PCs:      ",
                #         "Training set filename:              ",
                #         "Total time elapsed:                 ",
                #         "Reference network (if finetuning):  ",
                #         "Random seed:                        "]
                # B = [s+str(n) for s,n in list(zip(strg,paramsOther))]

            print("\nNetwork parameters written to:   ",fname,"\n")
        return

    
    def save_training_data_info(self, path, data_dic):
        """
        Writes information regarding the preparation of the training data to the model folder.
        """
        fname = os.path.join(path, self.name(), "TrainDataInfo" + self.name() + ".txt")

        notice_msg = ("NOTE: the random seed has no effect on RandomSampler, as there is a "
                      + "separate seed set in that sampler. Future improvements might remove "
                      + "this issue. For the time being, we will be untroubled by this "
                      + "non-critical loss of generality.")
        
        B = ["Data info:", tall_dic_str(data_dic), '', notice_msg, '\n']
        with open(fname, 'w') as f:
            f.write('\n'.join(B))
    
        return

            
    def evaluate_models(self, data):
        test_x,test_y,test_M = data

        print("PC PROJECTIONS STARTED")
        test_x0 = self.PCA.transform(test_x)
        print("PC PROJECTIONS COMPLETE")
        
        # batch size
        BSTEST = 10
        
        pNN = self.MLP.predict(test_x0).ravel()
        pCN = self.CNN.predict(test_M, batch_size=BSTEST, verbose=1).flatten()

        ## COMBINED: ENSEMBLE METHOD OF MLP + CNN
        # TODO: Is this equivalent after thresholding?
        pEN = pCN*pNN

        # ranking from highest prob to lowest prob.
        ranking = (lambda v : (test_x[(-v).argsort()]).astype(int)) 

        return pCN, ranking(pCN), pNN, ranking(pNN), pEN, ranking(pEN)


# Loader function to reconstruct object.
def load_model_bundle(path, model_id):
    B = ModelBundle(model_id)
    B._load(path)
    return B

def tall_dic_str(D):
    max_key_len = max(len(k) for k in D)
    format_string = "{0:" + str(max_key_len) + "} : {1},"
    s = "\n".join(format_string.format(str(k), str(v)) for k,v in D.items())
    return "\n".join(['{', s, '}'])


# Acquire the correct model given the parameters.
def fetch_model(NN_PATH, ReadNewest, UseModel):
    """
    Returns the model specified by the input parameters.
    """
    MODEL_DIRS = ["SpecialModels", "SavedModels"]
    if ReadNewest:
        model_path = 'SavedModels'
        fname_list = os.listdir(os.path.join(NN_PATH, model_path))

        if len(fname_list) == 0:
            error_msg = ("No models present in the 'SavedModels' directory. Please either train " 
                         + "A network using the provided utilities, or use one of the "
                         + "presupplied models in the 'SpecialModels' directory.")
            raise IOError(error_msg)
        else:
            key_func = lambda fname : os.path.getmtime(os.path.join(NN_PATH, model_path, fname))
            model_name = max(fname_list, key=key_func)

    else:
        model_name = UseModel

        for dir in MODEL_DIRS:
            if model_name in os.listdir(NN_PATH + dir):
                model_path = dir
                break
        else:
            error_msg = "No model corresponding to '{}' found.".format(UseModel)
            raise IOError(error_msg)
        
    return load_model_bundle(os.path.join(NN_PATH, model_path), model_name)

