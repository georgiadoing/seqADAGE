"""
Georgia Doing 2020
ADAGE class

A class for a tied-weight autoencoder.

Usage
    constructor

"""

import os
from keras.models import Model, Sequential
import numpy as np
import pandas
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

class Adage(object):
    """
    Adage (Analysis using Denoising Autoencoder of Gene Expression) class
    with the DAE, training data and biological enrichments of 'high weight'
    gene sets extracted from the hidden layer. Designed to facilitate 
    biological assessment as done in Tan and Doing et al, 2017, Cell Systems
    (<https://www.cell.com/cell-systems/fulltext/S2405-4712(17)30231-4>)

    Attributes:
        autoencoder (keras.Model): The tied-weights denoising autoencoder
        history (keras.History): Training loss over epochs
        weights (np.array): Encoding weight matrix
        loss (array): Training loss from history
        val_loss (array): Validation loss from history
        hw_genes (array): Arrays of high weight gene sets
        hw_genes_all (array): Arrays of high and low weight gene sets
        hw_genes_low (array): Arrays of low weight gene sets
        compendium (np.array): Training compendium
        activities (array): Encoded training compendium
        kegg_ps (list): KEGG enrichment p-values
        go_ps (list): GO term enrichment p-values
        regs_ps (list): Regulon enrichment p-values
        ops_ps (list): Operon enrichment p-values
        
    Methods:
        set_hwg_cutoff(x) : Re-assignes high weight gene lists based on x, a
            cutoff of standard deviations from the mean
        calc_enrich(path_file): Re-calculates enrichment p-vals from a gene
            set file, called by set_kegg(), set_go(), set_reg(), set_op()

    To Do:
        * prep E. coli KEGG, GO, operon and regulon gene sets
        * test with autoencoder as LinkedAE
        * calc_act vs set_act
    """

    def __init__(self, autoencoder, history, train_comp):
        """Constructor for Adage class

        Class for calculating enrichment of biological gene sets in high
        and low weight gene sets from a denoising autoencer trained on 
        gene expression data

        Args:
            autoencoder (keras.Model): The tied-weights denoising autoencoder
            history (keras.History): Training loss over epochs
            train_comp (np.array): Training compendium
        """
        self.autoencoder = autoencoder
        self.history = history
        self.weights = autoencoder.get_weights()[0] #np.zeros((train_comp.shape[0],train_comp.shape[0])) #
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        self.hwg_cutoff = 2.5
        self.hw_genes = self.weights > (np.std(self.weights, axis=0) * 2.5)
        self.hw_genes_all = np.concatenate((self.weights 
            > (np.std(self.weights, axis=0) * 2.5), self.weights 
            < (np.std(self.weights, axis=0) * -2.5)), axis=1)
        self.hw_genes_down = self.weights < (np.std(self.weights, axis=0) 
                                             * -2.5)
        self.compendium = train_comp
        self.activities = np.dot( self.compendium.T, self.weights)
        self.kegg_ps = []
        self.go_ps = []
        self.regs_ps = []
        self.ops_ps = []
        



    def set_hwg_cutoff(self, x):
        """Uses standard deviation cutoff to make high/low weight gene sets"""
        self.weights = self.autoencoder.get_weights()[0]
        self.hwg_cutoff = x
        self.hw_genes = self.weights > (np.std(self.weights, axis=0) * x)
        self.hw_genes_down = self.weights < (np.std(self.weights, axis=0) * -x)
        self.hw_genes_all = np.concatenate((self.hw_genes, self.hw_genes_down), axis=1)
        return self.hw_genes_all

    def calc_enrich(self, path_file, all_sigs = True):
        """Calculates enrichments of high/low weight genes for a gene set"""
        self.weights = self.autoencoder.get_weights()[0]
        hw_temp = self.hw_genes_all
        if(not all_sigs):
            hw_temp = self.hw_genes_all
        kegg = pandas.read_csv(path_file, 
                                  header = None, sep = '\t')
        temp_kegg_en = [1]*hw_temp.shape[1]
        for i in range(hw_temp.shape[1]):
            path_en = []
            sig_genes = self.compendium.index[np.where(hw_temp[:,i])]
            for j in range(kegg.shape[0]):
                path_genes = kegg[2][j].split(';')
                x = len(list(set(sig_genes) & set(path_genes))) - 1
                n = self.weights.shape[0]
                k = len(sig_genes)
                m = len(path_genes)
                p = hypergeom.logsf(x,n,k,m)
                path_en.append(-p)
            temp_kegg_en[i] = path_en
        kegg_df = pandas.DataFrame(temp_kegg_en, columns = kegg[0])
        return(kegg_df.replace([np.inf, -np.inf, 'nan'], 0))    

    def set_kegg(self, path_file, all = True):
        """Sets KEGG enrichments of gene sets lists from a file"""
        self.kegg_ps = self.calc_enrich(path_file)
        return(self.kegg_ps)

    def set_go(self, path_file, all = True):
        """Sets GO enrichments of gene sets lists from a file"""
        self.go_ps = self.calc_enrich(path_file)
        return(self.go_ps)

    def set_reg(self, path_file, all = True):
        """Sets Regulon enrichments of gene sets lists from a file"""
        self.regs_ps = self.calc_enrich(path_file)
        return(self.regs_ps)

    def set_op(self, path_file, all = True):
        """Sets Operon enrichments of gene sets lists from a file"""
        self.ops_ps = self.calc_enrich(path_file)
        return(self.ops_ps)

    def set_act(self, comp):
        """Sets encoded activities of the training data"""
        self.weights = self.autoencoder.get_weights()[0]
        self.activities = np.dot(comp, self.weights)
        return self.activities

    def calc_act(self, gene_exp):
        """Returns encoded activities of the training data"""
        self.weights = self.autoencoder.get_weights()[0]
        return np.dot(gene_exp, self.weights)