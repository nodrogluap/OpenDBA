#!/usr/bin/env Rscript
library(diptest)

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=1) {
  stop("Usage: diptest.R <opendba-means output.txt>\n", call.=FALSE)
} 

file <- args[1]
con <- file(description=file, open="r")
 
com <- paste("wc -l ", file, " | awk '{ print $1 }'", sep="")
n <- system(command=com, intern=TRUE)
 
pvals <- c()
poss <- c()
for(i in 1:n) {
  tmp <- readLines(con=con, n=1); 
  vals <- unlist(strsplit(tmp, '\t')); 
  poss <- append(poss, vals[1]); 
  pvals <- append(pvals, dip.test(as.numeric(unlist(strsplit(as.character(vals[2]),','))))$p.value)
}
qvals <- p.adjust(pvals, method="fdr")
cat("Hartigan Dip Test (low FP):\n")
which(qvals < .05) + as.numeric(poss[1])
