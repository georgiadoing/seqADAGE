"""
Mukundan Thanigaivelan 
UltraRegularizer class

October 18, 2026

A class to define a custom loss function penalty using ultrametric p-adic topology. Inherits
the Keras Regularizer class.

:author: Mukundan Thanigaivelan
"""

import tensorflow as tf
from keras.regularizers import Regularizer

class UltraRegularizer(Regularizer):
    def __init__(self, lambda_ultra=0.01, num_triplets=256):
        self.lambda_ultra = tf.Variable(lambda_ultra, dtype=tf.float32, trainable=False)
        self.num_triplets = num_triplets

    def __call__(self, weight_matrix):
        signatures = tf.transpose(weight_matrix)
        num_signatures = tf.shape(signatures)[0]
        
        idx_A = tf.random.uniform((self.num_triplets,), minval=0, maxval=num_signatures, dtype=tf.int32)
        idx_B = tf.random.uniform((self.num_triplets,), minval=0, maxval=num_signatures, dtype=tf.int32)
        idx_C = tf.random.uniform((self.num_triplets,), minval=0, maxval=num_signatures, dtype=tf.int32)

        A = tf.gather(signatures, idx_A)
        B = tf.gather(signatures, idx_B)
        C = tf.gather(signatures, idx_C)

        d_AB = tf.sqrt(tf.reduce_sum(tf.square(A - B), axis=1) + 1e-8)
        d_BC = tf.sqrt(tf.reduce_sum(tf.square(B - C), axis=1) + 1e-8)
        d_AC = tf.sqrt(tf.reduce_sum(tf.square(A - C), axis=1) + 1e-8)

        distances = tf.stack([d_AB, d_BC, d_AC], axis=1)

        sorted_distances = tf.sort(distances, axis=1, direction='DESCENDING')
        d1 = sorted_distances[:, 0] # Longest side
        d2 = sorted_distances[:, 1]

        penalty = tf.reduce_mean(tf.square(d1 - d2))
        return self.lambda_ultra * penalty

    def get_config(self):
        return {'lambda_ultra': float(self.lambda_ultra.numpy())}
