#########################################
#### Code for ANCOM
## Main function: ANCOM(...)
## Subsidiary function: ancom.detect(...)

############################################################
#### Data Format
# #########################################
# 1.Enter data with first p columns representing OTUs/ Taxa abundance counts 
# and last column representing the grouping variable. 
# Example:
# OTU1 OTU2 ... OTUp grp
# c11 c12 ... c1p Grp1
# c21 c22 ... c2p Grp2
# ....
# In case you have genus level summarized data the column names would be
# Genus1 Genus2 ... GenusK grp
# Note that the last column name must be "grp",
# but the source code can be changed to modify that.

### Taxonomy matrix if needed
# 2.Enter taxonomy matrix for the OTUs with each row for a single OTU
# in same order as columns in the real data. Example below:
# OTU_ID Taxonomy
# OTU1 String1
# OTU2 String2
# ...
###########################################

############################################################
############################################################
## The function to detect OTUs/Taxa (Used within the main ANCOM function)
## Arguments: 
## otu_data: Data (as described earlier)
## n_otu: Number of OTUs/Taxa (essentially number of columns - 1)
## alpha: significance level 
## multcorr: takes value 1 (Stringent correction: As described in the paper) 
## OR 2 (Less stringent: Multiplicity corrections are made within OTU/taxa)
## OR 3 (no multiple testing correction used).

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
############################################################
############################################################


############################################################
### How to implement ? 
############################################################
## The main ANCOM function
## Arguments: 
## filepath: Character string of filepath (for data as described earlier)
## Example: "~/Data Repository/data.txt"
## sig: significance level (usually FDR chosen as 0.05)
## multcorr_type: multcorr (defined above).
############################################################
############################################################

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
