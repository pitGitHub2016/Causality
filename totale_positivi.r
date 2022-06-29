## Step 1: ozone.data.r loads the data, standardizes it (so each variable has mean 0 and variance 1), and saves it for matlab
## Step 2: ozone.m pre-whitens each variable
## --> Step 3: ozone.r runs the PC algorithm on the original data and the pre-whitened data and makes plots

library(pcalg)
source("utils.r")

# in the paper (Figure 7) we compare the PC algorithm run on the ozone data without prewhitening
# to the PC algorithm run on the ozone data with prewhitening.

# original dataset (well, the variables are actually standardized to have mean 0 and variance 1)
data = read.csv("totale_positivi.csv",header=F,col.names=c('Abruzzo','Molise','Puglia','Valle dAosta','Lazio','Sardegna','Umbria','Emilia-Romagna','Sicilia',
'Basilicata','Lombardia','P.A. Trento','Veneto','Liguria','Calabria','Marche','Campania','Toscana','Piemonte','P.A. Bolzano','Friuli Venezia Giulia'))
alpha = .05
fit = pc(list(data=data), indepTest = check.cond.indep.PC, p = ncol(data), alpha=alpha, conservative=FALSE)
labels = names(data)
save(fit, alpha, labels, file="totale_positivi_original_results.rdata")

pdf("totale_positivi.pdf")
plot.graph(fit@graph, names(data))
dev.off()

# pre-whitened dataset (see ozone.m)
data = read.csv("totale_positivi_prewhitened.csv",header=F,col.names=c('Abruzzo','Molise','Puglia','Valle dAosta','Lazio','Sardegna','Umbria','Emilia-Romagna',
'Sicilia','Basilicata','Lombardia','P.A. Trento','Veneto','Liguria','Calabria','Marche','Campania','Toscana','Piemonte','P.A. Bolzano','Friuli Venezia Giulia'))

fit = pc(list(data=data), indepTest = check.cond.indep.PC, p = ncol(data), alpha=alpha, conservative=FALSE)
labels = names(data)
save(fit, alpha, labels, file="totale_positivi_prewhitened_results.rdata")

pdf("totale_positivi-prewhitened.pdf")
plot.graph(fit@graph, names(data))
dev.off()
