"""

Georgia Doing 2020
Denoising Autoencoder with P.a. RNAseq

Usage:

	run_model.py <dataset> [--seed=<SEED>]

Options:
	-h --help			Show this screen
	<dataset>			File patht to RNAseq compnedium
	--seed = <SEED>		Random seed for training [default: 1]

Output:

	**Weights, Loss and validation loss saved as files
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
from AdageHyperModel import AdageHyperModel

import keras_tuner
import matplotlib.pyplot as plt
import os


#from keras import initializers

def tune_model(input_file, seed):
    """
	
	"""
    hp = keras_tuner.HyperParameters()
	# defining the number of neurons dynamically
    units = hp.Int(name="units", min_value=10, max_value=100, step=10)   
    # defining the dropout rate
    #dropout = hp.Int(name="dropout", min_value=0.0, max_value=0.3)
    # Automatically assign True/False values.
    act1 = hp.Choice('act1', ["sigmoid","tanh","relu"])
    #act2 = hp.Choice('act2', ["sigmoid","tanh","relu"])
    shuffle = hp.Boolean("shuffle", default=False)
    init = hp.Choice('init', ["glorot_uniform","glorot_normal"]) 
    kl1 = hp.Float('kl1', min_value = 0, max_value = 1, step = 0.1)
    kl2 = hp.Float('kl2', min_value = 0, max_value = 1, step = 0.1)
    al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1)
    lr = hp.Float('lr', min_value = 0.001, max_value = 0.1, step = 0.01) 
    bs = hp.Int('bs', min_value = 10, max_value = 50, step = 10) 

    all_comp = pd.read_csv(input_file, index_col=0)
    gene_num = np.size(all_comp, 0)

    tuner = keras_tuner.Hyperband(
                       hypermodel=AdageHyperModel(gene_num),#
		               hyperparameters = hp,
                       objective = "val_loss", #optimize val acc
                       max_epochs=50, #for each candidate model
                       overwrite=True,  #overwrite previous results
                       directory='hyperband_search_dir', #Saving dir
                       project_name='adage_tuner')
    
    
    x_train, x_train_noisy = prep_data(all_comp, seed)
    
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


    

def run_model(input_file, post_data = '', seed=123, enc_dim = 300, epochs=50, kl1=0, kl2=0, lr = 0.01,
			  act='sigmoid', init='glorot_uniform', tied=True, batch_size=10,
			  v=1, pre_w = ''):
	"""

	"""

	if(post_data == ''):
		post_data = input_file

	print("updated22")
	#print(keras.backend.backend())
	#all_comp = np.loadtxt(open(input_file, "rb"), delimiter=',', skiprows = 1)
	all_comp = pd.read_csv(input_file, index_col=0)
	 # this is the size of our input
	gene_num = np.size(all_comp, 0)
	encoding_dim = enc_dim


	x_train, x_train_noisy = prep_data(all_comp, seed)

	if(tied):
		autoencoder = linked_lstmae(encoding_dim, gene_num, act, init,
								seed, kl1, kl2)
	else:
		autoencoder = unlinked_ae(encoding_dim, gene_num, act, init,
								  seed, kl1, kl2, pre_w)
	#autoencoder.summary()
	autoencoder, history = train_model(autoencoder, x_train,
		                               x_train_noisy, epochs,
									   seed, batch_size, lr, v)



    # save second model before fine-tuning
	weights_tmp, b_weights_tmp = autoencoder.get_weights()[0:2]
	file_desc = ( input_file[:-4]
	             + post_data[16:-4]
	             + '_seed:' + str(seed)
				 + '_nodes:' + str(encoding_dim)
				 + "_kl1:" + str(kl1)
				 + "_kl2:" + str(kl2)
				 +  "_act:" + act
				 + '_init:' + init
				 + '_ep:' + str(epochs)
				 + '_tied:' + str(tied)
				 + '_batch:' + str(batch_size)
				 + '_lr:' + str(lr)
				 + '_prew:' + pre_w
				 + '_postPT')
	write_data(file_desc, weights_tmp, b_weights_tmp, history)


	#autoencoder.trainable = False
	base_weights = autoencoder.get_weights()
	#autoencoder.save('/tmp/base_model')
	print(np.shape(autoencoder.get_weights()))
	#base_model = tf.keras.models.load_model('/tmp/base_model')


	all_comp_post = pd.read_csv(post_data, index_col=0)
	x_train2, x_train2_noisy = prep_data(all_comp_post, seed)

	#base_model = tf.keras.models.load_model('/tmp/base_model')
	#base_weights = base_model.get_weights()

	# autoencoder2 = linked_ae(encoding_dim, gene_num, act, init, seed, kl1, kl2)
	# print(np.shape(autoencoder2.get_weights()))
	# print(np.shape(base_weights))
	# autoencoder2.set_weights(base_weights)
	# #print(np.shape(autoencoder2.get_weights()))

	# for  layer in autoencoder2.layers[:-1]:
	# 	layer.trainable = False
	# autoencoder2, history2 = train_model(autoencoder2, x_train2,x_train2_noisy, epochs, seed, batch_size, lr, v)

	#autoencoder2.save('/tmp/preT_model')

	#for  layer in autoencoder2.layers[:-1]:
	#	layer.trainable = True


    # save second model before fine-tuning
	# weights_tmp, b_weights_tmp = autoencoder2.get_weights()[0:2]
	# file_desc = (input_file[:-4]
	#              + post_data[16:-4]
	#              + '_seed:' + str(seed)
	# 			 + '_nodes:' + str(encoding_dim)
	# 			 + "_kl1:" + str(kl1)
	# 			 + "_kl2:" + str(kl2)
	# 			 +  "_act:" + act
	# 			 + '_init:' + init
	# 			 + '_ep:' + str(epochs)
	# 			 + '_tied:' + str(tied)
	# 			 + '_batch:' + str(batch_size)
	# 			 + '_lr:' + str(lr)
	# 			 + '_prew:' + pre_w
	# 			 + '_preFT')
	# write_data(file_desc, weights_tmp, b_weights_tmp, history2)

	# for fine tuning, epochs and lr could be 1/5 
	#autoencoder2, history2 = train_model(autoencoder2, x_train2, x_train2_noisy, int(epochs*10), seed, batch_size, lr, v)

	#autoencoder2.save('/tmp/tuned_model')

    # save post fine-tuned model
	#weights, b_weights = autoencoder2.get_weights()[0:2]
	#file_desc = (input_file[:-4]
	#              + post_data[16:-4]
	#              + '_seed:' + str(seed)
	# 			 + '_nodes:' + str(encoding_dim)
	# 			 + "_kl1:" + str(kl1)
	# 			 + "_kl2:" + str(kl2)
	# 			 +  "_act:" + act
	# 			 + '_init:' + init
	# 			 + '_ep:' + str(epochs)
	# 			 + '_tied:' + str(tied)
	# 			 + '_batch:' + str(batch_size)
	# 			 + '_lr:' + str(lr)
	# 			 + '_prew:' + pre_w
	# 			 + '_postFT')
	# write_data(file_desc, weights, b_weights, history)


	adage = ad.Adage(autoencoder, history, all_comp)
	#adage2 = ad.Adage(autoencoder2, history2, all_comp)
	return adage #, adage2


def unlinked_ae(encoding_dim, gene_num, act, init,seed, kl1, kl2, pre_w):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.Dense(encoding_dim, input_shape=(gene_num, ),
					activation=act, #sigmoid
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))(input_img)




	decoded = layers.Dense(gene_num, activation='sigmoid',
    				activity_regularizer = regularizers.l1(0))(encoded)

	# this model maps an input to its reconstruction
	autoencoder = models.Model(input_img, decoded)
	return(autoencoder)

def linked_ae_preT(encoding_dim, gene_num, act, init,seed, kl1, kl2):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.Dense(encoding_dim, input_shape=(gene_num, ), activation=act,
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))


	decoder = tw.TiedWeightsEncoder(input_shape=(encoding_dim,),
					output_dim=gene_num,encoded=encoded, activation="sigmoid")

	autoencoder = keras.Sequential()
	autoencoder.add(pre_w)
	autoencoder.add(decoder)
	return(autoencoder)


def linked_ae(encoding_dim, gene_num, act, init,seed, kl1, kl2):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.Dense(encoding_dim, input_shape=(gene_num, ), activation=act,
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))


	decoder = tw.TiedWeightsEncoder(input_shape=(encoding_dim,),
					output_dim=gene_num,encoded=encoded, activation="sigmoid")

	autoencoder = keras.Sequential()
	autoencoder.add(encoded)
	autoencoder.add(decoder)
	return(autoencoder)

def linked_lstmae(encoding_dim, gene_num, act, init,seed, kl1, kl2):
	input_img = layers.Input(shape=(gene_num,1))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.LSTM(encoding_dim)(input_img)
	
	decoded = layers.RepeatVector(gene_num)(encoded)
	decoded = layers.LSTM(1, return_sequences=True, activation="tanh")(decoded)
	
	autoencoder = keras.Model(input_img, decoded)
	return(autoencoder)


def prep_data(all_comp, seed):

	all_comp_np = all_comp.values.astype("float64")
	#print(np.shape(all_comp_np))
	# this is the size of our input
	gene_num = np.size(all_comp_np, 0)

	x_train = all_comp_np.transpose()
	#x_train = x_train.reshape((len(x_train),
	#						   np.prod(x_train.shape[1:])))
	noise_factor = 0.1
	x_train_noisy = x_train + (noise_factor
		            * np.random.normal(loc=0.0,scale=1.0, size=x_train.shape))
	x_train_noisy = np.clip(x_train_noisy, 0., 1.)

	return(x_train, x_train_noisy)

def train_model(autoencoder, x_train, x_train_noisy, epochs, seed, batch_size, lr, v):

	np.random.seed(seed)
	train_idxs = np.random.choice(x_train.shape[0],
							      int(x_train.shape[0]*0.9), replace=False)
	#print(train_idxs[1:5])
	x_train_train = x_train[train_idxs,:]
	x_train_test = x_train[~np.in1d(range(x_train.shape[0]),train_idxs),:]

	x_train_noise_train = x_train_noisy[train_idxs,:]
	x_train_noise_test = x_train_noisy[~np.in1d(range(x_train.shape[0]),
		                                              train_idxs),:]


	#optim = optimizers.Adadelta(learning_rate=lr) # lr=0.001, rho=0.95, epsilon=1e-07
	optim = optimizers.SGD(learning_rate=lr, momentum=.9) # lr=0.001, rho=0.95, epsilon=1e-07
	autoencoder.compile(optimizer=optim, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)) # "mse" tf.keras.losses.BinaryCrossentropy(from_logits=False) mse

	history = autoencoder.fit(x_train_noisy, x_train,
	              	epochs=epochs,
	                batch_size=batch_size,
	                shuffle=True,
	                validation_split = 0.1,
	                #verbose=1,
	                #validation_data=(np.array(x_train_noise_test),
	                #				 np.array(x_train_test)),
	                verbose=v
	                )
	return(autoencoder, history)


def write_data(file_desc, weights, b_weights, history):
	"""
	Save logs and output for a model in an outputs foolder
	"""
	#print(np.shape(weights))
	np.savetxt('../outputs/weights/data_files/' + file_desc + '_en_weights_da.csv',
		np.matrix(weights), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/bias/data_files/' + file_desc + '_en_bias_da.csv',
		np.matrix(b_weights), fmt = '%s', delimiter=',')
	#np.savetxt('../outputs/' + file_desc + '_de_weights.csv',
	#	np.matrix(weights[2]), fmt = '%s', delimiter=',')
	#np.savetxt('../outputs/' + file_desc + '_de_bias.csv',
#		np.matrix(weights[3]), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/loss/data_files//' + file_desc + '_loss_da.csv',
		np.matrix(history.history['loss']), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/val_loss/data_files/' + file_desc + '_val_loss_da.csv',
		np.matrix(history.history['val_loss']), fmt = '%s', delimiter=',')


if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='Get training set.')
		parser.add_argument('filename',type=str, nargs=1,
			help='filpath to training set.')
		args=parser.parse_args()
		#print(args.filename[0])
		run_model(args.filename[0])
