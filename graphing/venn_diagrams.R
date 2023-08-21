# Venn diagram with a range of common positions in different test conditions
VIC_all_KS <- read.table("VIC_All_KS.txt", header = TRUE)
VIC_all_Dip <- read.table("VIC_All_Dip.txt", header = TRUE)
VIC_all_KDE <- read.table("VIC_All_KDE.txt", header = TRUE)
VIC_bc_KS <- read.table("VIC_bc_KS.txt", header = TRUE)
VIC_bc_Dip <- read.table("VIC_bc_Dip.txt", header = TRUE)
VIC_bc_KDE <- read.table("VIC_bc_KDE.txt", header = TRUE)

KCDC_all_KS <- read.table("KCDC_All_KS.txt", header = TRUE)
KCDC_all_Dip <- read.table("KCDC_All_Dip.txt", header = TRUE)
KCDC_all_KDE <- read.table("KCDC_All_KDE.txt", header = TRUE)
KCDC_bc_KS <- read.table("KCDC_bc_KS.txt", header = TRUE)
KCDC_bc_Dip <- read.table("KCDC_bc_Dip.txt", header = TRUE)
KCDC_bc_KDE <- read.table("KCDC_bc_KDE.txt", header = TRUE)

VIC_all_KS <- as.vector(VIC_all_KS$KS)
VIC_all_Dip <- as.vector(VIC_all_Dip$Dip)
VIC_all_KDE <- as.vector(VIC_all_KDE$KDE)
VIC_bc_KS <- as.vector(VIC_bc_KS$KS)
VIC_bc_Dip <- as.vector(VIC_bc_Dip$Dip)
VIC_bc_KDE <- as.vector(VIC_bc_KDE$KDE)

KCDC_all_KS <- as.vector(KCDC_all_KS$KS)
KCDC_all_Dip <- as.vector(KCDC_all_Dip$Dip)
KCDC_all_KDE <- as.vector(KCDC_all_KDE$KDE)
KCDC_bc_KS <- as.vector(KCDC_bc_KS$KS)
KCDC_bc_Dip <- as.vector(KCDC_bc_Dip$Dip)
KCDC_bc_KDE <- as.vector(KCDC_bc_KDE$KDE)

VIC_lit <- read.table("VIC_Literature.txt", header = FALSE)
VIC_lit <- as.vector(VIC_lit$V1)
KCDC_lit <- read.table("KCDC_Literature.txt", header = FALSE)
KCDC_lit <- as.vector(KCDC_lit$V1)


library(VennDiagram)
library(scales)
library(gridExtra)

# Matching with a range of +5 to -5
match_within_range <- function(x, vec) {
  return(any(abs(x - vec) <= 4))
}

# Indices of matched elements
get_matched_indices <- function(x, vec) {
  return(which(abs(x - vec) <= 4))
}

vcount <- function(test, lit) {
  overlap <- sapply(lit, match_within_range, test)
  tov <- table(overlap)
  count_true <- as.numeric(tov["TRUE"])
  count_false <- as.numeric(tov["FALSE"])
  count_lit <- length(lit)
  count_test <- length(test)
  veclit <- 1:count_lit
  vectest <- (count_lit-count_true+1):(count_lit-count_true+count_test)
  venn.diagram(x = list(veclit, vectest), category.names = rep("", 2), cex = 5,
               filename = "Venn.png",
               col = c("#0000ff", "#00ff00"), 
               fill = c(alpha("#0000ff", 0.3), alpha("#00ff00", 0.3)))
}
vcount(KCDC_bc_KS, VIC_lit)
vcount(KCDC_bc_KDE, VIC_lit)
vcount(KCDC_bc_Dip, VIC_lit)

vcount(KCDC_bc_KS, KCDC_lit)
vcount(KCDC_bc_KDE, KCDC_lit)
vcount(KCDC_bc_Dip, KCDC_lit)

vcount(VIC_all_KS, VIC_lit)
vcount(VIC_all_KDE, VIC_lit)
vcount(VIC_all_Dip, VIC_lit)

vcount(VIC_all_KS, KCDC_lit)
vcount(VIC_all_KDE, KCDC_lit)
vcount(VIC_all_Dip, KCDC_lit)