"""
Georgia Doing 2023
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
from tensorflow.keras.layers import Input, Dense
#from keras.models import Model, Sequential
#from tensorflow.keras import initializers
import TiedWeightsEncoder as tw
import Adage as ad

import keras_tuner as kt
import matplotlib.pyplot as plt
import os


class AdageHyperModel(kt.HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        

    def build(self, hp):
        encoding_dim= hp.Choice('units', [8, 16, 32, 64])
        act1 = hp.Choice('act1', ["sigmoid","tanh","relu"])
        #act2 = hp.Choice('act2', ["sigmoid","tanh","relu"])
        init = hp.Choice('init', ["glorot_uniform","glorot_normal"])
        kl1 = hp.Float('kl1', min_value = 0, max_value = 1, step = 0.1)
        kl2 = hp.Float('kl2', min_value = 0, max_value = 1, step = 0.1)
        al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1) 
        lr = hp.Float('lr', min_value = 0.001, max_value = 0.1, step = 0.01) 
        
        encoded = layers.Dense(
                        encoding_dim,
                        input_shape=(self.input_shape, ), 
                        activation=act1,
					    kernel_initializer = init,
					    kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				    activity_regularizer = regularizers.l1(al1))


        decoder = tw.TiedWeightsEncoder(
                        input_shape=(encoding_dim,),
					    output_dim=self.input_shape,   
                        encoded=encoded, 
                        activation="sigmoid")

        autoencoder = keras.Sequential()
        autoencoder.add(encoded)
        autoencoder.add(decoder)
        optim = optimizers.SGD(learning_rate=lr, momentum=.9) # lr=0.001, rho=0.95, epsilon=1e-07
        autoencoder.compile(optimizer=optim, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)) # "mse" tf.keras.losses.BinaryCrossentropy(from_logits=False) mse

        return autoencoder
    
    def fit(self,hp, model, *args, **kwargs):
        bs = hp.Int('bs', min_value = 10, max_value = 50, step = 10) 
        return model.fit(
              *args,
              batch_size = bs,
              #epochs = 10,
              shuffle = hp.Boolean("shuffle", default=False),
              **kwargs)
         