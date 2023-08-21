# Violins of picoamperage distributions
Violin.156 <- read.table("156_Nref_pas_for_violin.txt", header = TRUE)
Violin.43bc <- read.table("bc43_Nref_pas_for_violin.txt", header = TRUE)
KS.156 <- read.table("156.KS.Total.txt", header = TRUE)
Dip.156 <- read.table("156.Dip.Total.txt", header = TRUE)
KDE.156 <- read.table("156.KDE.Total.txt", header = TRUE)
KS.43 <- read.table("43bc.KS.Total.txt", header = TRUE)
Dip.43 <- read.table("43bc.Dip.Total.txt", header = TRUE)
KDE.43 <- read.table("43bc.KDE.Total.txt", header = TRUE)
Tombo <- read.table("Tombo.txt", header = TRUE)

Dip.156 <- as.numeric(Dip.156$Dip.Total.Unique)
V.156.Dip <- Violin.156[Violin.156$key %in% Dip.156, ]

NtoVICref2 <- function(i) {
  ifelse (i >= 1630 & i <= 1698, 1699-i, 29889-i)
}

# Convert values to N:C.Reference position
V.156.Dip$NewKey <- NtoVICref2(V.156.Dip$key)
V.156.Dip$NewKey <- paste0("G.", V.156.Dip$NewKey)

Tombo$TomboNew <- NtoVICref2(Tombo$Tombo)
Tombo$TomboNew <- paste0("G.", Tombo$TomboNew)

library(ggplot2)
##########################################################
# Step 1: Diptest
ggplot(V.156.Dip, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "156 Dip") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

###########################################################
# Step 2: KDE minus Diptest
KDE.156 <- as.numeric(KDE.156$KDE.Total.Unique)
KDE.156.nodip <- KDE.156[!(KDE.156 %in% Dip.156)]
V.156.KDE <- Violin.156[Violin.156$key %in% KDE.156.nodip, ]
# Convert values to N:C.Reference position
V.156.KDE$NewKey <- NtoVICref2(V.156.KDE$key)
V.156.KDE$NewKey <- paste0("G.", V.156.KDE$NewKey)

p <- ggplot(V.156.KDE, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "156 KDE without Dip") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

KDE.key.nodip <- sort(unique(V.156.KDE$NewKey))

p2 <- p + scale_x_discrete(limits = as.character(KDE.key.nodip[1:35]))
p3 <- p + scale_x_discrete(limits = as.character(KDE.key.nodip[36:70]))
p4 <- p + scale_x_discrete(limits = as.character(KDE.key.nodip[71:104]))


###########################################################
# Step 3: KS minus Diptest and KDE
KS.156 <- as.numeric(KS.156$KS.Total.Unique)
Dip.KDE.156 <- c(KDE.156.nodip, Dip.156)
Dip.KDE.156 <- sort(Dip.KDE.156)
KS.156.noDipKDE <- KS.156[!(KS.156 %in% Dip.KDE.156)]
V.156.KS <- Violin.156[Violin.156$key %in% KS.156.noDipKDE, ]
V.156.KS$NewKey <- NtoVICref2(V.156.KS$key)
V.156.KS$NewKey <- paste0("G.", V.156.KS$NewKey)

p <- ggplot(V.156.KS, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "156 KS without Dip or KDE") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

KS.key.noDipKDE <- sort(unique(V.156.KS$NewKey))
vals <- as.numeric(gsub("G.","", KS.key.noDipKDE))
KS.key.noDipKDE <- KS.key.noDipKDE[order(vals)]

p2 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[1:35]))
p3 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[36:70]))
p4 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[71:105]))
p5 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[106:140]))
p6 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[141:175]))
p7 <- p + scale_x_discrete(limits = as.character(KS.key.noDipKDE[176:217]))

###########################################################
# Step 4: Tombo minus Diptest and KDE and KS
Tombo0 <- Tombo$Tombo
Dip.KDE.KS.156 <- sort(unique(c(KDE.156, KS.156, Dip.156)))
Tombo0_unique <- Tombo0[!(Tombo0 %in% Dip.KDE.KS.156)]
V.156.Tombo <- Violin.156[Violin.156$key %in% Tombo0_unique, ]
V.156.Tombo$NewKey <- NtoVICref2(V.156.Tombo$key)
V.156.Tombo$NewKey <- paste0("G.", V.156.Tombo$NewKey)

Tombo0_unique <- sort(unique(V.156.Tombo$NewKey))

p <- ggplot(V.156.Tombo, aes(x = factor(NewKey), y = value)) +
  geom_violin(fill = "blue") +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "156 Tombo") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

p2 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[1:35]))
p3 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[36:70]))
p4 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[71:105]))
p5 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[106:118]))

# Basecalled
rm(list=ls())
KS.43 <- read.table("43bc.KS.Total.txt", header = TRUE)
Dip.43 <- read.table("43bc.Dip.Total.txt", header = TRUE)
KDE.43 <- read.table("43bc.KDE.Total.txt", header = TRUE)
Tombo <- read.table("Tombo.txt", header = TRUE)
Violin.43 <- read.table("bc43_Nref_pas_for_violin.txt", header = TRUE)

##########################################################
# Step 1: Diptest
Dip.43 <- Dip.43$Dip.Total.Unique
V.43.Dip <- Violin.43[Violin.43$key %in% Dip.43, ]

V.43.Dip$NewKey <- NtoVICref2(V.43.Dip$key)
V.43.Dip$NewKey <- paste0("G.", V.43.Dip$NewKey)

Tombo$TomboNew <- NtoVICref2(Tombo$Tombo)
Tombo$TomboNew <- paste0("G.", Tombo$TomboNew)

ggplot(V.43.Dip, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "43BC Dip") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")


###########################################################
# Step 2: KDE minus Diptest
KDE.43 <- as.numeric(KDE.43$KDE.Total.Unique)
KDE.43.nodip <- KDE.43[!(KDE.43 %in% Dip.43)]
V.43.KDE <- Violin.43[Violin.43$key %in% KDE.43.nodip, ]
V.43.KDE$NewKey <- NtoVICref2(V.43.KDE$key)
V.43.KDE$NewKey <- paste0("G.", V.43.KDE$NewKey)

p <- ggplot(V.43.KDE, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "43BC KDE without Dip") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

KDE.key.noDip <- sort(unique(V.43.KDE$NewKey))
vals <- as.numeric(gsub("G.","", KDE.key.noDip))
KDE.key.noDip <- KDE.key.noDip[order(vals)]

p2 <- p + scale_x_discrete(limits = as.character(KDE.key.noDip[1:42]))
p3 <- p + scale_x_discrete(limits = as.character(KDE.key.noDip[43:83]))


###########################################################
# Step 3: KS minus Diptest and KDE
KS.43 <- as.numeric(KS.43$KS.Total.Unique)

Dip.KDE.43 <- sort(unique(c(KDE.43, Dip.43)))
KS.43.noDipKDE <- KS.43[!(KS.43 %in% Dip.KDE.43)]
V.43.KS <- Violin.43[Violin.43$key %in% KS.43.noDipKDE, ]
V.43.KS$NewKey <- NtoVICref2(V.43.KS$key)
V.43.KS$NewKey <- paste0("G.", V.43.KS$NewKey)

p <- ggplot(V.43.KS, aes(x = factor(NewKey), y = value, fill = factor(NewKey) %in% Tombo$TomboNew)) +
  geom_violin() +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "43BC KS without Dip or KDE") +
  scale_fill_manual(values = c("grey", "blue")) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

###########################################################
# Step 4: Tombo minus Diptest and KDE and KS
Tombo0 <- Tombo$Tombo
Dip.KDE.KS.43 <- sort(unique(c(KDE.43, KS.43, Dip.43)))
Tombo0_unique <- Tombo0[!(Tombo0 %in% Dip.KDE.KS.43)]
V.43.Tombo <- Violin.43[Violin.43$key %in% Tombo0_unique, ]
V.43.Tombo$NewKey <- NtoVICref2(V.43.Tombo$key)
V.43.Tombo$NewKey <- paste0("G.", V.43.Tombo$NewKey)

p <- ggplot(V.43.Tombo, aes(x = factor(NewKey), y = value)) +
  geom_violin(fill = "blue") +
  geom_point(aes(x = factor(NewKey), y = Nref), color = "red") +
  labs(x = "Reference Position", y = "Z-Normalized Picoamps", title = "43BC Tombo") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1), legend.position = "none")

Tombo0_unique <- sort(unique(V.43.Tombo$NewKey))


p2 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[1:35]))
p3 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[36:70]))
p4 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[71:105]))
p5 <- p + scale_x_discrete(limits = as.character(Tombo0_unique[106:133]))

### Single violinplot of non-matching positions

tests.156 <- sort(unique(c(KDE.156, KS.156, Dip.156, Tombo0)))
V.156.nonpos <- Violin.156[!(Violin.156$key %in% tests.156), ]
V.156.nonpos$valueshifted <- V.156.nonpos$value - V.156.nonpos$Nref

tests.43 <- sort(unique(c(KDE.43, KS.43, Dip.43, Tombo0)))
V.43.nonpos <- Violin.43[!(Violin.43$key %in% tests.43), ]
V.43.nonpos$valueshifted <- V.43.nonpos$value - V.43.nonpos$Nref

V.156.nonpos$group <- "Mapped and Unmapped"
V.43.nonpos$group <- "Mapped Only"

V.156.nonpost <- V.156.nonpos[c("valueshifted", "group")]
V.43.nonpost <- V.43.nonpos[c("valueshifted", "group")]

combined <- rbind(V.156.nonpost, V.43.nonpost)
ggplot(combined, aes(x = group, y = valueshifted)) + 
  geom_violin() + labs(x = "Group", y = "Z-normalized Picoamps")


