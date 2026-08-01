"""
Georgia Doing 2025
SeqADAGE class

A class for an RNA-seq based ADAGE model. Inherits ADAGE class.

Usage
    constructor
    SeqADAGE.train_model()

"""

from adage import Adage as ad
from adage import LinkedAE as lae
from adage import AdageHyperModel as ahm
import pandas as pd
import numpy as np
from tensorflow.keras import optimizers, losses
import keras_tuner



class SeqAdage(ad.Adage): 
    """
    Adage instance with associated linked denoising autoencoder (LinkedAE), 
    training data, and hyperparameter settings. Weight matrix used to 
    assign Adage atributes, namely high weight genes and gene set 
    enrichment results.

    Attributes:
        autoencoder (LinkedAE): The tied-weights denoising autoencoder
        all_comp (pd.df): Pandas DataFrame of training data
        gene_num (int): Number of genes (deatures, dims) in input data
        enc_dim (int): Encoding dimension, number of nodes in hidden layer
        epochs (int): Number of epochs for training
        seed (int): Random seed
        batch_size (int): Number of samples per training batch
        mm (flt): Momentum for fitting, 0-1
        lr (flt): Learning rate, 0-1
        v (int): Verbose, 1-True, 0-False

    Methods:
        prep_data() : prepares data by introducing noise
        train_model(): fits autoencoder to training data
        tune_model(): hyperparameter search
        save_weights(out_dir, out_file): write weights, bias and loss to csvs

    To Do:
        * check autoencoder weights update after training/fitting
        * reconcile with Adage constructor, super().__init__()
        * prep E. coli KEGG, GO, operon and regulon gene sets
        * implement transfer models
    """
    
    def __init__(self, input_file, seed=100, enc_dim=10 ,kl1=0, kl2=0, 
                 act="tanh", act2="tanh", tied=True, epochs=50, 
                 init="glorot_uniform", batch_size=10, dropout=0, mm=0, 
                 lr=0.01, v=1):
        """Constructor for SeqAdage with hyperparameters

        This class inherits the Adage class, instantiating a linked
        denoising autoencoder with a set of hyperparameter values, fitting
        it to a training dataset and saving weight and bias matrices to files.

        Args:
            input_file (str): Name of csv with training data
            seed (int): Random seed
            enc_dim (int): Encoding dimension, number of nodes in hidden layer
            kl1 (flt): Amount of L1 regularizatotion, 0-1
            kl2 (flt): Amount of L2 regularizatotion, 0-1
            act (str): Activation function during encoding
            act2 (str): Activation function during decoding
            tied (bool): Whether ncoder and decoder weights are tied
            epochs (int): Number of epochs for training
            init (str): Inititalization function for random weight distribution
            batch_size (int): Number of samples per training batch
            dropout (flt): Proportion of input values randomly dropped
            mm (flt): Momentum for fitting, 0-1
            lr (flt): Learning rate, 0-1
            v (int): Verbose, 1-True, 0-False
        """
        self.autoencoder = lae.LinkedAE(enc_dim, act2, dropout,kl1,kl2,init)
        self.all_comp = pd.read_csv(input_file, index_col=0)
        self.gene_num = np.size(self.all_comp, 0)
        self.encoding_dim = enc_dim
        self.act = act2
        self.dropout = dropout
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.kl1 = kl1
        self.kl2 = kl2
        self.init=init
        self.mm = mm
        self.v = v
        self.history = None
        self.input_file = input_file
        #super().__init__(self.autoencoder, self.history, self.all_comp)

    def get_act(self):
        return self.act

    def prep_data(self):
        """Prepares training data by adding noise. Called by train_model()"""
        all_comp_np = self.all_comp.values.astype("float64")
        gene_num = np.size(all_comp_np, 0)
        x_train = all_comp_np.transpose()
        # set noise factor
        noise_factor = 0.1
        x_train_noisy = x_train + (noise_factor* np.random.normal(loc=0.0,scale=1.0, size=x_train.shape))
        # limit to range 0-1
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        return(x_train, x_train_noisy)



    def train_model_variable(self, ep):
        """Fits the autoencoder model to the training set."""
        np.random.seed(self.seed)
        x_train, x_train_noisy = self.prep_data()
        optim = optimizers.SGD(learning_rate=self.lr, momentum=self.mm) # lr=0.001, rho=0.95, epsilon=1e-07
        self.autoencoder.compile(optimizer=optim, 
                        loss=losses.MeanSquaredError()) #BinaryCrossentropy(from_logits=False)) 

        history = self.autoencoder.fit(x=x_train, 
                                  y=x_train_noisy, 
                                  epochs=ep,
                                  batch_size=self.batch_size,
                                  #shuffle=True,
                                  validation_split = 0.1,
                                  verbose=self.v
                                 )
        print("fitted")
        #self.autoencoder = autoencoder
        self.history = history
        return(self.autoencoder)

    def train_model(self):
        """Fits the autoencoder model to the training set."""
        self.train_model_variable(self.epochs)

    def pre_train_model(self):
        """Fits the autoencoder model to the training set for 1/10th epochs."""
        self.train_model_variable(self.epochs//10)
        

    def fine_tune_model(self, map_file, new_data, kl1=None, kl2=None, 
                 act=None, act2=None,  epochs=None, 
                  batch_size=None, dropout=None, mm=None, 
                 lr=None,init=None):
        """Fits the autoencoder model to new training set."""
        # update hyper-params
        if act is not None:
            self.act = act
        if act2 is not None:
            self.act2 = act2
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if dropout is not None:
            self.dropout = dropout
        if mm is not None:
            self.mm = mm
        if lr is not None:
            self.lr = lr
        if kl1 is not None:
            self.kl1 = kl1
        if kl2 is not None:
            self.kl2 = kl2
        if init is not None:
            self.init = init
        
        # get weights after pre-training
        weights_prev, b_weights_prev, d_weights_prev = self.autoencoder.get_weights()[0:3]

        # load mapping file between pre-training and fine-tuning domains
        mapper = pd.read_csv(map_file, index_col=0)
        new_weights = weights_prev.T.dot(mapper).T
        new_d_weights = d_weights_prev.T.dot(mapper).T

        # load new data
        np.random.seed(self.seed)
        self.all_comp = pd.read_csv(new_data, index_col=0)
        x_train, x_train_noisy = self.prep_data()

        # initialize a new data with new input shape
        autoencoder2 = lae.LinkedAE(self.encoding_dim, self.act, self.dropout,self.kl1,self.kl2,self.init)
        optim = optimizers.SGD(learning_rate=self.lr, momentum=self.mm) # lr=0.001, rho=0.95, epsilon=1e-07
        autoencoder2.compile(optimizer=optim, 
                        loss=losses.MeanSquaredError()) #BinaryCrossentropy(from_logits=False)) 
        
        history = autoencoder2.fit(x=x_train, 
                                  y=x_train_noisy, 
                                  epochs=1,
                                  batch_size=self.batch_size,
                                  #shuffle=True,
                                  validation_split = 0.1,
                                  verbose=self.v
                                 )
        # get initialized weights as background distribution
        weights_tmp, b_weights_tmp, d_weights_tmp = autoencoder2.get_weights()[0:3]
        new_weights[new_weights==0] = weights_tmp[new_weights==0]
        new_d_weights[new_d_weights==0] = d_weights_tmp[new_d_weights==0]

        # initialize to weights from mapping and background
        autoencoder2.set_weights([new_weights, b_weights_prev, new_d_weights])

        # fine-tune training
        history = autoencoder2.fit(x=x_train, 
                                  y=x_train_noisy, 
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  #shuffle=True,
                                  validation_split = 0.1,
                                  verbose=self.v
                                 )
        
        print("fine-tuned")
        self.autoencoder = autoencoder2
        self.history = history
        return(autoencoder2)


    def save_weights(self, out_dir, out_file):
        """
        Save logs and output for a model in an outputs foolder
        """
        weights, b_weights = self.autoencoder.get_weights()[0:2]
        np.savetxt(out_dir + '/weights/' + out_file + '_ew_da.csv',
                   np.matrix(weights), fmt = '%s', delimiter=',')
        np.savetxt(out_dir + '/bias/' + out_file + '_eb_da.csv',
                   np.matrix(b_weights), fmt = '%s', delimiter=',')
        np.savetxt(out_dir + '/loss/' + out_file + '_l_da.csv',
                   np.matrix(self.history.history['loss']), fmt = '%s', delimiter=',')
        np.savetxt(out_dir + '/val_loss/' + out_file + '_vl_da.csv',
                   np.matrix(self.history.history['val_loss']), fmt = '%s', delimiter=',')
        

    def tune_model(self, seed, u=[10,100,10],d=[0.0,0.9,0.1],
                  a1=["sigmoid","tanh","relu", "celu"], a2=["sigmoid","tanh","relu", "celu"],
                  i=["glorot_uniform","glorot_normal"],k1=[0,.15,.005],
                  k2=[0,.9,.05],l=[.001,.1,.01],b=[5,100,5],m=[0.0,.9,.1]):
        """
    	
    	"""
        hp = keras_tuner.HyperParameters()
    	# defining the number of neurons dynamically
        units = hp.Int(name="units", min_value=u[0], max_value=u[1], step=u[2])   
        # defining the dropout rate
        dropout = hp.Float(name="dropout", min_value=d[0], max_value=d[1], step = d[2])
        # Automatically assign True/False values.
        act1 = hp.Choice('act1', a1)
        act2 = hp.Choice('act2', a2)
        shuffle = hp.Boolean("shuffle", default=False)
        init = hp.Choice('init', i) 
        kl1 = hp.Float('kl1', min_value = k1[0], max_value = k1[1], step = k1[2])
        kl2 = hp.Float('kl2', min_value = k2[0], max_value = k2[1], step = k2[2])
        #al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1)
        lr = hp.Float('lr', min_value = l[0], max_value = l[1], step = l[2]) 
        bs = hp.Int('bs', min_value = b[0], max_value = b[1], step = b[2]) 
        mm = hp.Float('mm', min_value = m[0], max_value = m[1], step = m[2])
    
        #all_comp = pd.read_csv(input_file, index_col=0)
        gene_num = np.size(self.all_comp, 0)
    
        tuner = keras_tuner.Hyperband(
                           hypermodel=ahm.AdageHyperModel(input_shape=gene_num),#
    		               hyperparameters = hp,
                           objective = "val_loss", #optimize val acc
                           max_epochs=50, #for each candidate model
                           overwrite=True,  #overwrite previous results
                           directory='/work/gd134/hyperband_search_dir', #Saving dir
                           project_name=self.input_file.removesuffix(".csv").replace("../data_files/","")+"_" + str(seed))
        
        
        #x_train, x_train_noisy = prep_data(all_comp, seed)
        x_train, x_train_noisy = self.prep_data()
        
        np.random.seed(seed)
        train_idxs = np.random.choice(x_train.shape[0],
    							      int(x_train.shape[0]*0.9), replace=False)
    	#print(train_idxs[1:5])
        x_train_train = x_train[train_idxs,:]
        x_train_test = x_train[~np.in1d(range(x_train.shape[0]),train_idxs),:]
    
        x_train_noise_train = x_train_noisy[train_idxs,:]
        x_train_noise_test = x_train_noisy[~np.in1d(range(x_train.shape[0]),
    		                                              train_idxs),:]
        
        #print(len(ss.space))
        
        tuner.search(x_train_noise_train, x_train_train,
                 #max_trials=50,  # Max num of candidates to try
    			 #batch_size=batch_size,
                 validation_data=(x_train_noise_test,x_train_test))
        tuner.results_summary() 
        num_trials = len(tuner.oracle.trials.values())
        best_hps = tuner.get_best_hyperparameters(num_trials)
        best_models = tuner.get_best_models(num_trials)
        #ss = tuner.search_space_summary(extended=True)
        
        model = tuner.hypermodel.build(best_hps[0])
        hist  = tuner.hypermodel.fit(
    	    best_hps[0], 
    		model,
    	    x = x_train_noise_train, 
    		y = x_train_train,) 
        
        return(best_hps, tuner)


        
