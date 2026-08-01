"""
Georgia Doing 2025
LinkedAE class

A class for a tied-weight autoencoder.
Inherits keras Sequential class and uses DenseTranspose

Usage
    constructor

"""

import keras
from adage import DenseTranspose as dt
from tensorflow.keras import regularizers

class LinkedAE(keras.models.Sequential):
    """
    LinkedAE is a class for a linked, tied-weights autoencoder. It inherits
    the keras Sequential class.

    Attributes:
        dense_1 (keras.layers.Dense): The encoding layer
        d (keras.layers.Dropout): A dropout layer
        act2 (str): The decoder activation function 
        decoder (DenseTranspose): The decoding layer

        
    Methods:
        call(inputs) : Initialized the LinkedAE with an input dataset

    To Do:
        * inherit keras.Model for more versatile architectures
        * implement pre-training option

    """

    def __init__(self, encoding_dim = 10, act2="tanh", d=0,kl1=0,kl2=0,init="tanh" ):
        """
        LinkedAE is a class for a linked, tied-weights autoencoder. It inherits
        the keras Sequential class.
    
        Args:
            encoding_dim (int): Encoding dimensions, nodes in hidden layer
            act2 (str): The decoder activation function 
            d (flt): The proportion of input features to drop in dropout layer
        """
        super().__init__()
        self.dense_1 = keras.layers.Dense(encoding_dim, kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
                                         kernel_initializer = init)
        self.d = keras.layers.Dropout(d)
        self.act2 = act2
        self.decoder = dt.DenseTranspose(self.dense_1, activation=self.act2)
        self.add(self.d)
        self.add(self.dense_1)
        self.add(self.decoder)



    def call(self,inputs):
        """Initialized the LinkedAE with an input dataset"""
        return super().call(inputs)