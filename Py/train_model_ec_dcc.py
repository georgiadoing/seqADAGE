"""
This script 

Usage:
    train_model_ec_dcc.py seed

    seed: random seed

"""

# for adage models
from adage import Adage as ad
from adage import SeqAdage

# for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# misc
#from scipy.stats import hypergeom
#import csv
#import random
import time
import tensorflow as tf

import sys




if __name__ == '__main__':

    seed = int(sys.argv[1])
    all_comp = pd.read_csv('../data_files/ecmg_mgsp_lcn01.csv', index_col = 0)
    gene_num = np.size(all_comp, 0)
    samp_num = np.size(all_comp, 1)

    comp = "ecmg_mgsp_lcn01.csv"
    
    
    #model_dict_post = {}
    
    stime = time.time()
    ltime = 0
    c = 0

    name = 'ad_' + comp[:-4] + '_2026_07_27_' + str(seed+571)
    print(name)
    ttime = time.time()
    sa = SeqAdage.SeqAdage('../data_files/' + comp,
                                              seed=seed+571,
                                              enc_dim = 450,
                                              kl1=0,
                                              kl2=0,
                                              act = "tanh",
                                              act2="tanh",
                                              tied = True,
                                              epochs=100,
                                              init="glorot_uniform",
                                              batch_size=15,
                                              dropout = 0,
                                              mm = 0.6,
                                              lr = 0.5) 
    mseq = sa.train_model()
    sa.save_weights('/work/gd134/sA_out',name)
    temp = ad.Adage(sa.autoencoder, sa.history, sa.all_comp)
    #model_dict_post[name] = temp
    ltime = ((time.time() - ttime) + ltime)
    c+=1
    
    rtime = time.time() - stime
    print(rtime)
    print(c)
    print(ltime / c)
    print(rtime / 60)
