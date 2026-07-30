"""
Georgia Doing 2025
DenseTranspose class

A class for dense layer transposed from another dense layer. Used for 
tied-weights autoencoder. Inherits keras Layer class.

Usage
	constructor

"""

import tensorflow as tf
import keras

class DenseTranspose(keras.layers.Layer):
    """
    DenseTranspose is a class for linked decoders. It inherits
    the keras Sequential class.

    Attributes:
        dense (keras.layers.Dense): The encoding layer to transpose
        activation (str): The decoder activation function 
        
    Methods:
        build(batch_input_shape): sets biases and shape
        call(inputs) : Transpose encoding weights, returns activations

    To Do:
        * get_config, from_config
        * implement **kwargs for other params from Layer class
    """
    
    def __init__(self, dense,  activation="tanh"): # , **kwargs
        """Constructor for DenseTranspose class

        Class for creating a decoding layer from an encoding layer

        Args:
            dense (keras.layers.Dense): The encoding layer to transpose
            activation (str): The decoder activation function
        """
        super().__init__() #**kwargs
        self.dense = dense
        self.activation = keras.activations.get(activation)
        
    
    def build(self, batch_input_shape):
        """Builds decoder for batch, with bias"""
        self.biases = self.add_weight(name="bias", initializer="zeros",
            shape=[self.dense.input.shape[-1]]) 
        super().build(batch_input_shape)
    
    def call(self, inputs, training=None):
        """Calculates weights transpose and activation vals with bias"""
        z = tf.matmul(inputs, self.dense.kernel, transpose_b=True)
        return self.activation(z + self.biases)


    def get_config(self):
        config = super().get_config()
        config.update({
            "dense": self.dense,
            "activation": self.activation
        })
        return config
