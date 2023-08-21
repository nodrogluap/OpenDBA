#!/bin/bash

# Path file input for a specific cluster
grep 'DIAG' $1 | awk '{print $3}' | sort -g | uniq -c > diagonal.wc.txt
grep -v 'OPEN_RIGHT' $1 | awk '{print $3}' | sort -g | uniq -c > coverage.wc.txt

Rscript diag_median_proportion.R coverage.wc.txt diagonal.wc.txt 0.8 15
