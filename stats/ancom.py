import pyviennacl as pv
import pyopencl as cl
import numpy as np
import pandas as pd
from time import time
import copy

import os, sys, site
import rpy2.robjects as robj
import pandas.rpy.common as com
import numpy as np
from numpy import random, array
from pandas import DataFrame, Series


from math import log
from statsmodels.sandbox.stats.multicomp import multipletests 
from permutation import (_cl_mean_permutation_test,
                         _np_mean_permutation_test,
                         _naive_mean_permutation_test)

def _log_compare(mat,cats,permutations=1000):
    """
    Calculates pairwise log ratios between all otus
    and performs a permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest
    
    otu_table: numpy 2d matrix
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array
    Returns:
    --------
    log ratio pvalue matrix
    """    
    r,c = mat.shape
    log_mat = np.log(mat+(1./r))
    log_ratio = np.zeros((r,r))
    for i in range(r-1):
        ratio =  np.array(np.matrix(log_mat[i+1:,:]) - np.matrix(log_mat[i,:]))
        m, p = _np_mean_permutation_test(ratio,cats,permutations)
        log_ratio[i,i+1:] = np.matrix(p).transpose()        
    return log_ratio

def ancom_cl(otu_table,cats,alpha,permutations=1000):
    """
    Calculates pairwise log ratios between all otus
    and performs permutation tests to determine if there is a
    significant difference in OTU ratios with respect to the
    variable of interest
    
    otu_table: pandas.core.DataFrame
    rows = samples
    cols = otus
    
    cat: numpy array float32
    Binary categorical array

    permutations: int
    Number of permutations to use in permutation test
    
    Returns:
    --------
    log ratio pvalue matrix
    """

    mat = otu_table.as_matrix().transpose()
    _logratio_mat = _log_compare(mat,cats,permutations)
    logratio_mat = _logratio_mat + _logratio_mat.transpose()
    n_otu,n_samp = mat.shape
    ##Multiple comparisons
    for i in range(n_otu):
         _,pvalues,_,_ = multipletests(logratio_mat[i,:])
         logratio_mat[i,:] = pvalues
    W = np.zeros(n_otu)
    for i in range(n_otu):
        W[i] = sum(logratio_mat[i,:] < alpha)
    par = n_otu-1 #cutoff

    c_start = max(W)/par
    cutoff = c_start - np.linspace(0.05,0.25,5)
    D = 0.02 # Some arbituary constant
    dels = np.zeros(len(cutoff))
    prop_cut = np.zeros(len(cutoff),dtype=np.float32)
    for cut in range(len(cutoff)):
        prop_cut[cut] = sum(W > par*cutoff[cut])/float(len(W))
    for i in range(len(cutoff)-1):
        dels[i] = abs(prop_cut[i]-prop_cut[i+1])
        
    if (dels[1]<D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[1]
    elif (dels[1]>=D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[2]
    elif (dels[2]>=D) and (dels[3]<D) and (dels[4]<D):
        nu=cutoff[3]
    else:
        nu=cutoff[4]
    up_point = min(W[W>nu*par])
    results = otu_table.columns[W>=nu*par]
    return results

def ancom_R(otu_table,sig,multcorr,wilcox):
    """
    otu_table: pandas.DataFrame
        rows = samples
        cols = otus
        A table of OTU abundances
        Last column will be group ids (e.g. Urban/Rural)
    sig: float
        significance value
    multcorr: int
        multiple corrections (e.g. 1, 2, 3)
        1: Very strict
        2: Not as strict
        3: No multiple hypotheses correction (very bad idea)
    wilcox: bool
       perform an exact wilcox test or not

    Returns:
    --------
    Names of signficantly correlated OTUs
    """
    rcode = """
    ancom.detect <- function(otu_data,n_otu,alpha,multcorr,wilcox=FALSE){
      logratio.mat=matrix(NA,nr=n_otu,nc=n_otu)

      for(i in 1:(n_otu-1)){
          for(j in (i+1):n_otu){
              data.pair=otu_data[,c(i,j,n_otu+1)]
              lr=log((0.001+as.numeric(data.pair[,1]))/(0.001+as.numeric(data.pair[,2])))
              logratio.mat[i,j]=wilcox.test(lr[data.pair$grp==unique(data.pair$grp)[1]],
                              lr[data.pair$grp==unique(data.pair$grp)[2]],
                              exact=wilcox)$p.value
          }
      }

      ind <- lower.tri(logratio.mat)
      logratio.mat[ind] <- t(logratio.mat)[ind]
      logratio.mat[which(is.finite(logratio.mat)==FALSE)]=1
      mc.pval=t(apply(logratio.mat,1,function(x){
        s=p.adjust(x, method = "BH")
        return(s)
      }))

      a=logratio.mat[upper.tri(logratio.mat,diag=F)==T]

      b=matrix(0,nc=n_otu,nr=n_otu)
      b[upper.tri(b)==T]=p.adjust(a, method = "BH")
      diag(b)=NA
      ind.1 <- lower.tri(b)
      b[ind.1] <- t(b)[ind.1]

      if(multcorr==2){
        W=apply(mc.pval,1,function(x){
          subp=length(which(x<alpha))
        })
      }else if(multcorr==1){
        W=apply(b,1,function(x){
          subp=length(which(x<alpha))
        })
      }else if(multcorr==3){
        W=apply(logratio.mat,1,function(x){
          subp=length(which(x<alpha))
        })
      }

      return(W)
    }

    ANCOM <- function(real.data,sig,multcorr_type,wilcox){

      ####real.data <- read.delim(filepath,header=TRUE)
      colnames(real.data)[ ncol(real.data) ] <- "grp"
      real.data <- data.frame(real.data[which(is.na(real.data$grp)==FALSE),],row.names=NULL)
      par1_new=dim(real.data)[2]-1

      W.detected <- ancom.detect(real.data,par1_new,sig,multcorr_type,wilcox)
      if( ncol(real.data) < 10 ){

        ### Detected using arbitrary cutoff
        results <- colnames(real.data)[which(W.detected > par1_new-1 )]    
      } else{
        ### Detected using a stepwise mode detection
        if(max(W.detected)/par1_new >= 0.10){
          c.start <- max(W.detected)/par1_new
          cutoff  <- c.start-c(0.05,0.10,0.15,0.20,0.25)
          prop_cut<- rep(0,length(cutoff))
          for(cut in 1:length(cutoff)){
            prop_cut[cut] <- length(which(W.detected>=par1_new*cutoff[cut]))/length(W.detected)
          } 
          del=rep(0,length(cutoff)-1)
          for(i in 1:(length(cutoff)-1)){
            del[i]=abs(prop_cut[i]-prop_cut[i+1])
          }

          if(del[1]<0.02&del[2]<0.02&del[3]<0.02){nu=cutoff[1]
          }else if(del[1]>=0.02&del[2]<0.02&del[3]<0.02){nu=cutoff[2]
          }else if(del[2]>=0.02&del[3]<0.02&del[4]<0.02){nu=cutoff[3]                                
          }else{nu=cutoff[4]}

          up_point <- min(W.detected[which(W.detected>=nu*par1_new)])

          W.detected[W.detected>=up_point]=99999
          W.detected[W.detected<up_point]=0
          W.detected[W.detected==99999]=1

          results=colnames(real.data)[which(W.detected==1)]

        } else{
          W.detected=0
          results <- "No significant OTUs detected"
        }
      }

      results <- as.data.frame( results , ncol=1 )
      colnames(results)= paste0("OTU Significant at FDR = ", sig )
      return(results)

    }
    """
    otu_Rtable = com.convert_to_r_dataframe(otu_table)
    Rfunc = robj.r(rcode)
    sig_Rotus = Rfunc(otu_Rtable,sig,multcorr,wilcox)
    sig_otus = com.convert_robj(sig_Rotus)
    return sig_otus


