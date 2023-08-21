#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly=TRUE)

true_diagonals <- function(covtxt, diagtxt, prop, range) {
	covtxt <- args[1]
	diagtxt <- args[2]
	prop <- as.numeric(args[3])
	range <- as.numeric(args[4])
	cov <- read.table(covtxt, header = FALSE, fill = TRUE)
	seq_num <- cov[1,]
	seq_num <- seq_num$V1
	cov <- cov[-1,]
	diag <- read.table(diagtxt, header = FALSE)
	med <- numeric(0)
	for (i in 1:nrow(diag)) {
		position <- diag[i, 2]
		count <- diag[i, 1]
		half <- (range-1)/2
		range_15 <- seq(from = position - half, to = position + half, by = 1)
		range_15 <- range_15[range_15 >= 0]
		cov.wc <- numeric(0)
		for (j in range_15) {
			cov.wc <- c(cov.wc, cov$V1[cov$V2 == j])
		}
		cov.wc.median <- median(cov.wc)
		cov.wc.median[cov.wc.median >= seq_num] <- seq_num
		med <- c(med, cov.wc.median)
	}
	diag$Median <- med
	diag$Proportion <- diag$V1 / diag$Median
	diag_filtered <- subset(diag, Proportion >= prop)
	colnames(diag_filtered)[colnames(diag_filtered) == 'V1'] <- 'Count'
	colnames(diag_filtered)[colnames(diag_filtered) == 'V2'] <- 'Position'
	diag_filtered
}

true_diagonals(args[1], args[2], args[3], args[4])
