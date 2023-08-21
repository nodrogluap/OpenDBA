#!/usr/bin/env
args = commandArgs(trailingOnly=TRUE)

cent <- read.table(args[1], header = FALSE)
cent <- t(cent)
cent <- as.data.frame(cent)
cent_ID <- cent[1,]
cent <- cent[-1,]

pdf(file = paste(cent_ID, '.pdf', sep =''),
    width = 8, height = 5,
    bg = 'white')

plot(c(1:length(cent)), cent, type = 'l', lty = 1,
     xlab = cent_ID, ylab = 'Current intensity')

dev.off()
