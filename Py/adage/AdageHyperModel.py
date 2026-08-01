"""
Georgia Doing 2026
ADAGE HyperModel class

A class for a tied-weight autoencoder that can be used with Keras Tuners.

Usage
	constructor

"""


#import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras as keras

import tensorflow as tf
from tensorflow import keras

import argparse
import numpy as np
import csv
import pandas as pd


from tensorflow.keras import optimizers, regularizers, layers, initializers, models
from tensorflow.keras.layers import Input, Dense, Dropout
#from keras.models import Model, Sequential
#from tensorflow.keras import initializers
#import TiedWeightsEncoder as tw
#import Adage as ad
from adage import Adage as ad
from adage import DenseTranspose
from adage import LinkedAE as lae

#from autoencoders import DenseLayerAutoencoder

#from DenseTranspose import DenseTranspose


import keras_tuner as kt
import matplotlib.pyplot as plt
import os


class AdageHyperModel(kt.HyperModel):

    def __init__(self, input_shape, preT=False, init_weights=[]):
        self.input_shape = input_shape
        self.preT = preT
        self.init_weights = init_weights
        

    def build(self, hp):
        enc_dim= hp.Choice('units', [8, 16, 32, 64])
        act1 = hp.Choice('act1', ["sigmoid","tanh","relu", "celu"])
        tied = hp.Choice('tied', [True, False])
        act2 = hp.Choice('act2', ["sigmoid","tanh","relu", "celu"])
        init = hp.Choice('init', ["glorot_uniform","glorot_normal"])
        kl1 = hp.Float('kl1', min_value = 0, max_value = 1, step = 0.1)
        kl2 = hp.Float('kl2', min_value = 0, max_value = 1, step = 0.1)
        #al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1) 
        lr = hp.Float('lr', min_value = 0.001, max_value = 0.5, step = 0.001) 
        mm = hp.Float('mm', min_value = 0.0, max_value = 1, step = 0.01)
        dropout = hp.Float(name="dropout", min_value=0.0, max_value=0.9, step = 0.1)
        
        #inputs = Input(shape=(self.input_shape,))
        #d = Dropout(dropout)
        #x = DenseLayerAutoencoder([encoding_dim], activation=act1, dropout=0.0,l1=kl1,l2=kl2, preT=self.preT, init_fun = init, act_fun = act1)(d)
        #model = models.Model(inputs=inputs, outputs=x)

        #model = keras.models.Sequential()
        model = lae.LinkedAE(enc_dim, act2, dropout,kl1,kl2,init)

        '''
        dense_1 = keras.layers.Dense(encoding_dim, activation=act1,
		    kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
		    #activity_regularizer = regularizers.l1(0),
		    kernel_initializer = init, name = "dense_1")
        #dense_2 = keras.layers.Dense(30, activation=act1)

        model.add(inputs)
        model.add(d)
        model.add(dense_1)
        #model.add(dense_2)

        #model.add(DenseTranspose(dense_2, activation=act1))
        model.add(DenseTranspose(dense_1, activation=act2))

        if not tied:
        	dense_1 = keras.layers.Dense(encoding_dim, activation=act1,
        	    kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
        	    #activity_regularizer = regularizers.l1(0),
        	    kernel_initializer = init, name = "dense_1")
        	dense_2 = keras.layers.Dense(self.input_shape, activation=act2,
        	    kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
        	    #activity_regularizer = regularizers.l1(0),
        	    kernel_initializer = init, name = "dense_2")	
        	model = keras.models.Sequential()
        	model.add(inputs)
        	model.add(d)
        	model.add(dense_1)
        	model.add(dense_2)



        if self.preT:
        	print("preT ae ae")
        	#model.set_weights(self.init_weights)
'''
        optim = optimizers.SGD(learning_rate=lr, momentum=mm) # lr=0.001, rho=0.95, epsilon=1e-07
        model.compile(optimizer=optim,loss=tf.keras.losses.MeanSquaredError()) #BinaryCrossentropy(from_logits=False)) 
        return(model)

    
    def fit(self,hp, model, *args, **kwargs):
        bs = hp.Int('bs', min_value = 10, max_value = 50, step = 10) 
        return model.fit(
              *args,
              batch_size = bs,
              #epochs = 10,
              shuffle = hp.Boolean("shuffle", default=False),
              **kwargs)
         