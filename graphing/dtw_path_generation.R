#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)

library(dtwclust)

# Specify consensus sequence to compare to
centroid_norm <- scale(args[1])
centroid_norm <- as.numeric(centroid_norm)

dtwpaths <- function(query) {
  read <- read.table(query, header = FALSE)
  read <- t(read)
  read <- as.numeric(read[-1,])
  read <- as.numeric(scale(read))
  maxOR <- length(read)
  centroid_length <- length(centroid_norm)
  if (centroid_length >= maxOR) {
    dtwcompare <- dtw2(centroid_norm, read, step.pattern = symmetric2, open.end = TRUE, open.begin = FALSE, keep.internals = FALSE)
    direction <- c("NIL", dtwcompare$stepsTaken) 
    direction <- replace(direction, direction==1, "DIAG")
    direction <- replace(direction, direction==2, "UP")
    direction <- replace(direction, direction==3, "RIGHT")
    pa_query <- read[dtwcompare$index2]
    pa_centroid <- centroid_norm[dtwcompare$index1]
    paths <- data.frame(dtwcompare$index2, pa_query, dtwcompare$index1, pa_centroid, direction)
    paths$direction[paths$dtwcompare.index2 == maxOR & paths$direction == "RIGHT"] <- "OPEN_RIGHT"
    colnames(paths) <- c("Query_Pos", "Query_PA", "Centroid_Pos", "Centroid_PA", "Direction")
  } else {
    dtwcompare <- dtw2(read, centroid_norm, step.pattern = symmetric2, open.end = TRUE, open.begin = FALSE, keep.internals = FALSE)
    direction <- c("NIL", dtwcompare$stepsTaken) 
    direction <- replace(direction, direction==1, "DIAG")
    direction <- replace(direction, direction==2, "UP")
    direction <- replace(direction, direction==3, "RIGHT")
    pa_query <- read[dtwcompare$index1]
    pa_centroid <- centroid_norm[dtwcompare$index2]
    paths <- data.frame(dtwcompare$index1, pa_query, dtwcompare$index2, pa_centroid, direction)
    paths$direction[paths$dtwcompare.index2 == maxOR & paths$direction == "RIGHT"] <- "OPEN_RIGHT"
    colnames(paths) <- c("Query_Pos", "Query_PA", "Centroid_Pos", "Centroid_PA", "Direction")
  }
  write.table(paths, paste("paths.", query, sep = ""), row.names = FALSE, sep = "\t")
}

files <- list.files(path = ".", pattern = "*.txt", full.names = FALSE)

lapply(files, dtwpaths)