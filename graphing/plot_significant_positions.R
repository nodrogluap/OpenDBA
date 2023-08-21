# Input lists of significant positions found in each test
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

library(ggplot2)
library(dplyr)

VIC_all_KS <- c(VIC_all_KS, rep(NA, 525 - length(VIC_all_KS)))
VIC_all_Dip <- c(VIC_all_Dip, rep(NA, 525 - length(VIC_all_Dip)))
VIC_all_KDE <- c(VIC_all_KDE, rep(NA, 525 - length(VIC_all_KDE)))
VIC_bc_KS <- c(VIC_bc_KS, rep(NA, 525 - length(VIC_bc_KS)))
VIC_bc_Dip <- c(VIC_bc_Dip, rep(NA, 525 - length(VIC_bc_Dip)))
VIC_bc_KDE <- c(VIC_bc_KDE, rep(NA, 525 - length(VIC_bc_KDE)))

KCDC_all_KS <- c(KCDC_all_KS, rep(NA, 525 - length(KCDC_all_KS)))
KCDC_all_Dip <- c(KCDC_all_Dip, rep(NA, 525 - length(KCDC_all_Dip)))
KCDC_all_KDE <- c(KCDC_all_KDE, rep(NA, 525 - length(KCDC_all_KDE)))
KCDC_bc_KS <- c(KCDC_bc_KS, rep(NA, 525 - length(KCDC_bc_KS)))
KCDC_bc_Dip <- c(KCDC_bc_Dip, rep(NA, 525 - length(KCDC_bc_Dip)))
KCDC_bc_KDE <- c(KCDC_bc_KDE, rep(NA, 525 - length(KCDC_bc_KDE)))

pointsdf <- data.frame(x = 1:525, 
                       VIC_all_KS = VIC_all_KS,
                       VIC_all_Dip = VIC_all_Dip,
                       VIC_all_KDE = VIC_all_KDE, 
                       VIC_bc_KS = VIC_bc_KS,
                       VIC_bc_Dip = VIC_bc_Dip,
                       VIC_bc_KDE = VIC_bc_KDE, 
                       KCDC_all_KS = KCDC_all_KS,
                       KCDC_all_Dip = KCDC_all_Dip,
                       KCDC_all_KDE = KCDC_all_KDE, 
                       KCDC_bc_KS = KCDC_bc_KS,
                       KCDC_bc_Dip = KCDC_bc_Dip,
                       KCDC_bc_KDE = KCDC_bc_KDE)

pointsdf.long <- reshape2::melt(pointsdf, id.vars = "x", 
                                variable.name = "group", 
                                value.name = "y")
pointsdf.long <- na.omit(pointsdf.long)
pointsdf.long <- select(pointsdf.long, -x)

write.table(pointsdf.long, file = "long_dataframe_vectors.txt", sep = "\t", row.names = FALSE)

sig_values <- read.table("long_dataframe_vectors.txt", sep = '\t', header = TRUE)

library(ggpattern)

sig_values$direction <- ifelse(sig_values$group %in% c("VIC_all_KS",
                                                         "VIC_all_Dip",
                                                         "VIC_all_KDE",
                                                         "KCDC_all_KS",
                                                         "KCDC_all_Dip",
                                                         "KCDC_all_KDE"), 1, -1)
colors_hist <- c("VIC_all_Dip" = "#ff0000", 
                 "VIC_all_KDE" = "#ff7b7b", 
                 "VIC_all_KS" = "#ffbaba", 
                 "KCDC_all_Dip" = "#0000cd", 
                 "KCDC_all_KDE" = "#0066FF", 
                 "KCDC_all_KS" = "#77CCFF",
                 "VIC_bc_Dip" = "#ff0000", 
                 "VIC_bc_KDE" = "#ff7b7b", 
                 "VIC_bc_KS" = "#ffbaba", 
                 "KCDC_bc_Dip" = "#0000cd", 
                 "KCDC_bc_KDE" = "#0066FF", 
                 "KCDC_bc_KS" = "#77CCFF") 
                 
lit_vec <- read.table("literature_long_vector.txt", header = TRUE)
legend_order <- c("VIC_all_Dip", "VIC_all_KDE", "VIC_all_KS",
                  "KCDC_all_Dip", "KCDC_all_KDE", "KCDC_all_KS",
                  "VIC_bc_Dip", "VIC_bc_KDE", "VIC_bc_KS",
                  "KCDC_bc_Dip", "KCDC_bc_KDE", "KCDC_bc_KS")

# Plot histogram of significant positions
ggplot(sig_values, aes(x = y, y = direction, fill = group)) +
  geom_bar(stat = "identity", position = "stack") + 
  scale_fill_manual(values = colors_hist, breaks = legend_order) + 
  geom_point(data = lit_vec, aes(x = y, y = 0, shape = as.factor(group)), 
             size = 1.5, fill = "white") +
  scale_shape_manual(values = c(15, 23)) +
  geom_hline(yintercept = 0, color = "#5A5A5A", linetype = "solid", size = 0.1) +
  labs(shape = "Literature") + xlab("Reference Position 28555-28654") + ylab("Group") +
  xlim(c(28555, 28654))