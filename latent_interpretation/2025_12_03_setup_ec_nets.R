################################################################################
# Title:  setup_ec_nets.R
# Author: G. Doing, georgia.doing@duke.edu
# Date:   2025-12-03
#
# Description: An r script that loads data for the visualization of gene-gene
# networks for the E. coli y-ome project. A helper script sourced by
# notebook: 2025_12_03_ecoli_y-ome_nets_mg1655plus.Rmd

################################################################################
# Load data and models
# compendium
#ec_comp <- read.csv('../data_files/ecoli_MG165plus_genome_part2_log_counts_norm_01.csv')
ec_comp <- read.csv('../data_files/ecmg_mgsp_lcn01.csv')
# gene precense-absence matrix
gpa <- read.csv('ecoli_pan_genomev3_gene_presence_absence_ann.csv', 
                stringsAsFactors = F)
rownames(gpa) <- gpa$first_name_comp

gpa$Gene2 <- gpa$Gene

gpa['e0ba967783ad45e29471d6259beaaa64_7630', 'Gene2'] <- 'eloD41'
gpa['e0ba967783ad45e29471d6259beaaa64_7632', 'Gene2'] <- 'eloE'
gpa['e0ba967783ad45e29471d6259beaaa64_7634', 'Gene2'] <- 'eloT'

# combine gene loci names from chosen strains
name_cat <- paste(paste(paste(paste(gpa$MG1655, gpa$O157_H7_EHEC, sep = "-"),
                        gpa$BL21_DE3, sep = "-"), gpa$Gene, sep = "-"), gpa$Gene2, sep = "-")
names(name_cat) <- gpa$first_name_comp
rownames(ec_comp) <- make.unique(name_cat[ec_comp$X])

# names with mg1655 numbers for operon analysis
name_cat_mg1655 <- name_cat[! is.na(gpa$MG1655)]
names(name_cat_mg1655) <- gpa$MG1655[! is.na(gpa$MG1655)]

# ensemble model(s)
ec_model_files <- list.files('/work/gd134/adage_models', pattern = "net450*")
ec_models <- lapply(ec_model_files, function(x){
  m <- read.csv(paste0('/work/gd134/adage_models/',x),
                stringsAsFactors = F,
                sep = "\t", skip = 2, header = F, nrows  = 4272) #6390
  rownames(m) <- rownames(ec_comp)
  m
})

################################################################################
# Calculate gene-gene correlations
corr_method <- 'pearson' # pearson, kendall, spearman
# compenium correlations
ec_comp_corr <- cor(t(ec_comp[,-1]), method = corr_method) 
rownames(ec_comp_corr)  <- rownames(ec_comp)
colnames(ec_comp_corr)  <- rownames(ec_comp)

# model correlations
ec_model_corrs <- lapply(ec_models, function(x){
  comp_corr <- cor(t(x), method = corr_method)
  rownames(comp_corr)  <- rownames(ec_comp)
  colnames(comp_corr)  <- rownames(ec_comp)
  comp_corr
})

################################################################################
# Load annotations and KEGG pathways
# MG1655 annotations
ec_anns_mg1655 <- read.csv('Jessica--(MG1655)-All-genes-of-MG1655.txt',
                           header = T,
                           stringsAsFactors = F,
                           sep = '\t')
unchar_genes_mg1655 <- ec_anns_mg1655$Accession.1[ec_anns_mg1655$Characterization == "UNCHARACTERIZED"]
rownames(ec_anns_mg1655 ) <- ec_anns_mg1655$Accession.1

# EHEC 0157_H7 annotations
ec_anns_ehec <- read.csv('Jessica--(O157_H7-EHEC)-All-genes-of-E.-coli-serovar-O157_H7-str.-Sak.txt',
                         header = T,
                         stringsAsFactors = F,
                         sep = '\t')
unchar_genes_ehec <- ec_anns_ehec$Accession.1[ec_anns_ehec$Characterization == "UNCHARACTERIZED"]

# BL21-DE3 annotations
ec_anns_bl21de3 <- read.csv('Jessica--(BL21-DE3)-All-genes-of-E.-coli-BL21(DE3).txt',
                            header = T,
                            stringsAsFactors = F,
                            sep = '\t')
unchar_genes_bl21de3 <- ec_anns_bl21de3$Accession.1[ec_anns_bl21de3$Characterization == "UNCHARACTERIZED"]

# all genes by any of their names
comp_genes_pool <- unique(unlist(lapply(rownames(ec_comp), 
                                        function(x) (str_split(x,'-' )))))
# genes not in MG1655
no_mg1655 <- rownames(ec_comp)[sapply(rownames(ec_comp), 
                                      function(x) unlist(str_split(x,'-' ))[1] == "NA")]
# names of uncharacterized genes from MG1655
name_cat_unchar <- rownames(ec_comp)[sapply(rownames(ec_comp), function(x){
  unlist(str_split(x,'-' ))[1] %in% unchar_genes_mg1655})]
unchar_genes <- rownames(ec_comp)[rownames(ec_comp) %in% c(name_cat_unchar)]
unchar_genes_all <- rownames(ec_comp)[rownames(ec_comp) %in% 
                                        c(name_cat_unchar, no_mg1655)]

# gene characterization status
ec_anns <- data.frame(
  row.names = rownames(ec_comp_corr),
  'Characterization' = sapply(rownames(ec_comp_corr), function(x){
    n = unlist(str_split(x,'-' ))[1]
    if( n %in% rownames(ec_anns_mg1655)){
      ec_anns_mg1655[n, 'Characterization']
    } 
    else{
      "NoMG1655"
    }
  })
)
ec_anns[ec_anns == ""] <- "NoNote"
################################################################################
# setup net
# edges from correlation values
edges <- melt(ec_model_corrs[[1]])
colnames(edges) <- c('from','to','weight')
edges <- edges[edges$weight > 0.1,]
edges <- edges[edges$weight < 1,]
de_genes <- unchar_genes[c(1:30)]
# 
de_edges <- edges[edges$weight > 0.175 & (edges$from %in% de_genes | edges$to %in% de_genes),]

net_genes <- union(union(de_genes, de_edges$from), de_edges$to)
net_edges <- de_edges
net_edges <- net_edges[!is.na(net_edges$from),]

net_edges <- net_edges[!net_edges$from == net_edges$to,]

################################################################
# setup operons

ops <- unique(unlist(lapply(ec_anns_mg1655$Transcription.Units, 
                            function(x) str_split(x, " // "))))
ops <- ops[! ops == ""]
opes_list <- lapply(ops, function(x){
  genes <- unlist(sapply(c(1:nrow(ec_anns_mg1655)), function(g){
    if(str_detect(ec_anns_mg1655$Transcription.Units[g], x)){
      TRUE
    }else{FALSE}
  }))
  gene_acc <- sapply(ec_anns_mg1655$Accession.1[genes], function(g) unlist(str_split(g, " // ")))
  gene_acc
})
names(opes_list) <- ops

opes_corrs_model <- lapply(opes_list, function(x){
  genes_temp <- unlist(x)
  genes_temp <- name_cat_mg1655[genes_temp]
  genes_temp <- genes_temp[genes_temp %in% rownames(ec_model_corrs[[1]])]
  
  corr_temp <- ec_model_corrs[[1]][genes_temp,genes_temp]
  corr_temp[lower.tri(corr_temp)] <- 0
  corr_temp[abs(corr_temp) > 0 & corr_temp < 1]
})

opes_corrs_comp <- lapply(opes_list, function(x){
  genes_temp <- unlist(x)
  genes_temp <- name_cat_mg1655[genes_temp]
  genes_temp <- genes_temp[genes_temp %in% rownames(ec_comp_corr)]
  
  corr_temp <- ec_comp_corr[genes_temp,genes_temp]
  corr_temp[lower.tri(corr_temp)] <- 0
  corr_temp[abs(corr_temp) > 0 & corr_temp < 1]
})

nopes_corrs_both <- lapply(opes_list, function(x){
  genes_temp <- unlist(x)
  genes_temp <- name_cat_mg1655[genes_temp]
  genes_temp <- genes_temp[genes_temp %in% rownames(ec_model_corrs[[1]])]
  rand_genes <- sample(setdiff(rownames(ec_model_corrs[[1]]),genes_temp), length(genes_temp))
  
  corr_temp <- ec_model_corrs[[1]][rand_genes,rand_genes]
  corr_temp[lower.tri(corr_temp)] <- 0
  
  
  corr_temp2 <- ec_comp_corr[rand_genes,rand_genes]
  corr_temp2[lower.tri(corr_temp2)] <- 0
  
  list(corr_temp2[(abs(corr_temp2) > 0) & (corr_temp2 < 1)], 
       corr_temp[(abs(corr_temp) > 0) & (corr_temp < 1)])
})

all_corrs <- c(opes_corrs_comp,lapply(nopes_corrs_both, function(x) x[[1]]),
               opes_corrs_model,lapply(nopes_corrs_both, function(x) x[[2]]))
op_corr_df <- data.frame("operon" = rep(ops,4),
                         "corrs_med" = sapply(all_corrs, function(x) median(x)),
                         "corrs_mu" = sapply(all_corrs, function(x) mean(x)),
                         "opnop" = rep(rep(c("operon", "random"), each=length(ops)),2),
                         "df" = c(rep("compendium", length(ops)*2), rep("model", length(ops)*2))
)