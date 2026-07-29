###########################################################
# This script builds an ensemble ADAGE model from many individual ADAGE models.
#
# Usage:
#     Rscript ensemble_construction.R data_file netfolder scratch_folder k
#             begin end cluster_seed method
#
#     data_file: file path to gene expression input file, mainly for
#                extracting gene identifiers from it
#     netfolder: file path to a folder that stores 100 ADAGE models to
#                ensemble with
#     scratch_folder: file path to a folder that stores some large tmp files
#     k: cluster size, usually set equal to the network size
#     begin: the first seed
#     end: the last seed
#     cluster_seed: random seed used in clustering
#     method: method to be used in measure node distance, either weight
#             or weighted. "weight" means corADAGE, and "weighted" means eADAGE.
#     outfolder: file path to folder to store the output ensemble model
###########################################################
#library(ppcor)
library(ClusterR)
#pacman::p_load("cluster", "ff",  "readr") #"sprint",
library(cluster)
library(readr)
library(ff)
#ptest()
source("ConsensusClusterPlus_modified2.R")
#source("ppam.R")
#source("pcor.R")
########### load command arguments

data_file <- commandArgs(trailingOnly = T)[1]
netfolder <- commandArgs(trailingOnly = T)[2]
scratch_folder <- commandArgs(trailingOnly = T)[3]
k <- as.integer(commandArgs(trailingOnly = T)[4])
begin <- as.integer(commandArgs(trailingOnly = T)[5])
end <- as.integer(commandArgs(trailingOnly = T)[6])
cluster_seed <- as.integer(commandArgs(trailingOnly = T)[7])
method <- commandArgs(trailingOnly = T)[8]
outfolder <- commandArgs(trailingOnly = T)[9]


########### load functions

ppamHook <- function(dist, k) {
  # this function hooks ppam function in the sprint package to
  # ConsensusClusterPlus.M
  dist <- ff(as.matrix(dist), vmode = "double")
  ppam_result <- ppam(dist, k, do.swap = FALSE)
  assignment <- ppam_result$clustering
  return(assignment)
}

pcmHook <- function(dist, k) {
  # this function hooks ppam function in the sprint package to
  # ConsensusClusterPlus.M
  #dist <- ff(as.matrix(dist), vmode = "double")
  print("ppam_begin")
  ppam_result <- Cluster_Medoids(dist, k, swap_phase= FALSE, threads=6, verbose=TRUE)
  assignment <- ppam_result$clusters
  print("ppam_end")
  return(assignment)
}
########### check method input

if (method == "weight") {
  weighted <- FALSE
} else if (method == "weighted") {
  weighted <- TRUE
  weighted_cor_file <- commandArgs(trailingOnly = T)[10]
} else {
  print("Unknown method! Method can either be 'weight' or 'weighted'.")
  pterminate()
  quit()
}

########### read in data

# set the temporary file folder path
options(fftempdir = scratch_folder)
# read in gene names from data_file
col_n <- count.fields(data_file, sep = ",")[1]
geneID <- read.table(data_file, sep = ",", header = T,
                     colClasses = c("character", rep("NULL", col_n - 1)))
geneN <- nrow(geneID)
print(geneN)
# read in weight matrix from network files and combine them together
combo_weight <- c()
model_count <- 0
for (seed in begin:end) {
  seed <- as.character(seed)
  print(seed)
  netfile <- paste("_", seed,
                   sep = "")
  print(netfile)
  netfile <- list.files(netfolder, pattern = netfile)
  print(netfile)
  netfile <- file.path(netfolder, netfile)
  #print(netfile)
  weight <- read_delim(netfile, delim = ",", col_names = F, n_max = geneN,
                       skip = 0)
  weight_matrix <- data.matrix(weight[1:geneN, ])
  nodeN <- ncol(weight_matrix)
  combo_weight <- cbind(combo_weight, weight_matrix)
  colnames(combo_weight)[(ncol(combo_weight) - nodeN + 1):ncol(combo_weight)] <-
    paste(rep("seed", nodeN), seed, "_node", seq(1, nodeN), sep = "")
  model_count <- model_count + 1
}
print("finish reading network files")

if (weighted) {
  print(weighted_cor_file)
  combo_weight_cor <- data.matrix(read_delim(weighted_cor_file, delim = "\t", col_names = F))
  print(dim((combo_weight_cor )))
  colnames(combo_weight_cor) <- rownames(combo_weight_cor) <- colnames(combo_weight)
  print(dim((combo_weight_cor )))
} else {
  # calculate pearson correlation between every two weight vectors
  combo_weight_cor_file <- tempfile(pattern = "pcor", tmpdir = scratch_folder,
                                    fileext = "")
  combo_weight_cor <- pcor(combo_weight, filename_ = combo_weight_cor_file)
}
print("to dist")
print(dim(as.dist(combo_weight_cor)))
#combo_weight_cor_round <- apply(combo_weight_cor, c(1,2), FUN = function(x) round(x, digits=10))
combo_weight_cor_dist <- ((1 - combo_weight_cor)/2)
print(combo_weight_cor_dist[1])
print("combo_weight_cor_dist dim")
print(dim(combo_weight_cor_dist))
print(class(combo_weight_cor_dist))
print("finish reading/calculating correlation matrix")

########### build ensemble model

# consensus clustering nodes from 100 models based on their weight vectors
print("consensus clustering starts")
# ConsensusClusterPlus.M is a simplified version of ConsensusClusterPlus
print("omg")
print(dim(combo_weight_cor_dist))
print("k")
print(k)
res <- ConsensusClusterPlus.M(combo_weight_cor_dist, oneK = k, reps = 10,
                              pItem = 0.5, pFeature = 1, distance = "pearson",
                              #pItem = 0.8, pFeature = 1, distance = "pearson",
                              clusterAlg = "pcmHook", # "pam"
                              seed = cluster_seed, verbose = TRUE)
print("consensus clustering finished")

# merge nodes by averaging weight vectors in the same cluster
print("merging node weights in the same cluster")
cl_assignment <- as.data.frame(res[[2]][["consensusClass"]])
colnames(cl_assignment) <- "cluster"
combo_weight_t <- t(combo_weight)
combo_weight_t_cl <- merge(combo_weight_t, cl_assignment, by = 0)

merged_node_weight <- aggregate(combo_weight_t_cl[, 2:(geneN + 1)],
                                list(combo_weight_t_cl$cluster), mean)
merged_node_weight <- t(merged_node_weight)
merged_node_weight <- merged_node_weight[-1, ]
merged_node_weight <- round(merged_node_weight, 8)
rownames(merged_node_weight) <- geneID$Gene_symbol

########### write outputs

# write the merged weight matrix to file, bias vectors are set to 0
dir.create(outfolder)
weight_file <- file.path(outfolder, paste0("net", nodeN, "_", model_count,
                         "models_", begin, "_", end, "_k=", k, "_seed=",
                         cluster_seed, "_ClusterBy", method,
                         "_avgweight_network_ADAGE.txt"))
write("layer1", weight_file)
write("weight matrix", weight_file, append = T)
write.table(merged_node_weight, weight_file, append = T,
            col.names = F, row.names = F, sep = "\t")
write("hidden bias vector", weight_file, append = T)
for (i in 1:nodeN) {
  write("0", weight_file, append = T)
}
write("visible bias vector", weight_file, append = T)
for (i in 1:geneN) {
  write("0", weight_file, append = T)
}
print("ensemble construction finished!")

# cleaning up temp files
if (exists("combo_weight_cor_file")) {
  file.remove(combo_weight_cor_file)
}

#pterminate()
quit()
