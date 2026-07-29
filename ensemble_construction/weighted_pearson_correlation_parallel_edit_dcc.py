"""
This script calculates the weighted Pearson correlation between each weight
vector pair in 100 individual models.

Usage:
    weighted_pearson_correlation_parallel.py netfolder outfile start_end
    process_n gene_n

    netfolder: the folder that stores individual adage models
    outfile: the output file
    start_end: the start and end of 100 random seeds, separated by comma
    process_n: the number of cores to use in parallel
    gene_n: the number of genes in the input layer of each model

"""

import numpy
import statsmodels.stats.weightstats as st
import sys
import multiprocessing as mp
import fnmatch
import os


def read_weight_matrix(network_file, feature_size):
    '''
    This function reads weight matrix from network structure file.
    It only reads in the first feature_size number of rows, which
    correspond to the places that stores weight matrix.
    '''
    W = []
    input_count = 0

    with open(network_file, 'r') as network_fh:
        #network_fh.next()  # skip the layer count line
        #network_fh.next()  # skip 'weight matrix' line
        for line in network_fh:
            line = line.strip().split(',')
            W.append(line)
            input_count += 1
            if input_count == feature_size:
                break

    W = numpy.array(W, dtype='float')
    return W


def w_cor(i, j):
    '''
    this function calculates the weighted Pearson correlation between two
    columns i,j in the combo_weight matrix.
    '''
    # the correlation weight is the average weight of two weight vectors
    weight_vector = (abs(combo_weight[:, i])+abs(combo_weight[:, j]))/2
    weighted_obj = st.DescrStatsW(combo_weight[:, [i, j]], weight_vector)
    w_cor_coef = weighted_obj.corrcoef[0, 1]
    return w_cor_coef


def w_cor_helper(args):
    '''
    helper function to pass multiple parameters to pool.map
    reference from http://www.erogol.com/passing-multiple-arguments-
    python-multiprocessing-pool/
    '''
    return w_cor(*args)


if __name__ == '__main__':

    netfolder = sys.argv[1]
    outfile = sys.argv[2]
    start_end = sys.argv[3]
    process_n = int(sys.argv[4])
    gene_n = int(sys.argv[5])
    # the first and last of 100 random seeds
    start, end = map(int, start_end.split(','))

    # read in 100 individual ADAGE models specified by a random seed and
    # combine them into one combo matrix
    combo_weight = []
    for seed in range(start, end+1):
        print(seed)
        for file in os.listdir(netfolder):
            print(file)
            if fnmatch.fnmatch(file, '*_'+str(seed)+ # '*_s'+str(seed)+
                               '*.csv'):
                netfile = os.path.join(netfolder, file)
                print("reading " + netfile)
                weight_matrix = read_weight_matrix(netfile, gene_n)
                combo_weight.extend(weight_matrix.T)

    combo_weight = numpy.array(combo_weight, dtype='float')
    combo_weight = combo_weight.T
    ncol = combo_weight.shape[1]

    # calculate the weighted Pearson correlation between two weight vectors
    # to avoid duplication, only the upper triangle in the correlation matrix
    # is calculated
    pool = mp.Pool(processes=process_n)  # initialize processes
    combo_weighted_corr = numpy.empty([ncol, ncol])
    upper_tri_index = numpy.triu_indices(ncol)  # index to upper triangle
    upper_tri = []
    for i in range(ncol):
        print("calculating correlation for column "+str(i))
        list_b = range(i, ncol)
        list_a = [i]*len(list_b)
        job_args = [(item_a, list_b[i]) for i, item_a in enumerate(list_a)]
        results = pool.map(w_cor_helper, job_args)
        upper_tri.extend(results)

    combo_weighted_corr[upper_tri_index] = upper_tri
    # copy the upper triangle part to lower triangle
    combo_weighted_corr[(upper_tri_index[1], upper_tri_index[0])] = \
        combo_weighted_corr[upper_tri_index]
    # write the correlation matrix to output file
    numpy.savetxt(outfile, combo_weighted_corr, delimiter='\t')
