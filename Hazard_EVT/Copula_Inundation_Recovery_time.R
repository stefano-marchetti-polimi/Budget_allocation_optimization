# Copula Code
# By Santiago Taguado & Thomas Coscia
# October 31st, 2024 

# Libraries
########################################################################

library(dplyr)
library(lubridate)
library(copula)
library(corrplot)
library(VineCopula)
library(CDVineCopulaConditional) 
library(NSM3)

###################################################################################
# Assess data is stationary. In order to assess you can perform tests  such as Petit test.
####################################################################################

ranks = pobs(Moderate_Inundation_Time_Duration$`Maximum Elevation (Meters)`)
ranks_1 = pobs(Moderate_Inundation_Time_Duration$`Duration (Hours)`+CC_impact)

B <- cbind(ranks,ranks_1)
plot(ranks,ranks_1)

# First Plot that must go
# Sample code to create the plot
plot(ranks, ranks_1, 
     xlab = expression("Rank of ~ Maximum Elevation"),  # Label showing rank of U[R]
     ylab = expression("Rank of ~ Flood Duration"), # Label showing rank of T[dew]
     pch = 19, col = "blue")

# Adding a perfect correlation line
# This assumes that ranks and ranks_1 are on the same scale
abline(0, 1, col = "black", lty = 2, lwd = 2)

##########################################################################################
# Step 2: Understanding the Dependence

spearman = cor(ranks,ranks_1, method = "spearman")
Kendall = cor(ranks,ranks_1,, method = "kendall")
Pearson = cor(ranks,ranks_1, method = "pearson")

c(spearman, Kendall, Pearson)
##########################################################################################
# Step 3: Assess confidence interval

stat_sig = kendall.ci(ranks,ranks_1, alpha=0.05, type="t", bootstrap=T, B=1000, example=F) 

##########################################################################################
# Step 4: Copula Assessment

# Depending on the type of dependence, that is positive, for example, then use copulas
# for this. 

# Gumbel Copula
gumbel_cop <- fitCopula(gumbelCopula(dim = 2), cbind(ranks, ranks_1), method = "ml")

# Gaussian copula
gaussian_cop <- fitCopula(normalCopula(dim = 2), cbind(ranks, ranks_1), method = "ml",optim.method = "BFGS")

# Clayton copula
clayton_cop <- fitCopula(claytonCopula(dim = 2), cbind(ranks, ranks_1), method = "ml")

# Frank copula
frank_cop <- fitCopula(frankCopula(dim = 2), cbind(ranks, ranks_1), method = "ml")

# Joy Copula
joe_copula <- fitCopula(joeCopula(dim = 2), cbind(ranks, ranks_1), method = "ml")

# t copula
t_copula <- fitCopula(tCopula(dispstr = "un", df.fixed = TRUE), cbind(ranks, ranks_1), method = "ml")

##########################################################################################
# Step 5: Goodness of Fit

# Formal statistical test such as the Cramers-Von Mises or the Smirnov-Kolmogorov test
# must be performed afterwards.

fit.c <- gofCopula(claytonCopula(dim = 2), 
                   cbind(ranks, ranks_1), 
                   simulation = c("mult"),
                   estim.method = c("mpl"))

fit.g <-gofCopula(gumbelCopula(dim = 2), 
                  cbind(ranks, ranks_1), 
                  simulation = c("mult"),
                  estim.method = c("mpl"))

fit.n <-gofCopula(normalCopula(dim = 2), 
                  cbind(ranks, ranks_1), 
                  simulation = c("mult"),
                  estim.method = c("mpl"),
                  optim.method = "BFGS")

fit.f <-gofCopula(frankCopula(dim = 2), 
                  cbind(ranks, ranks_1), 
                  simulation = c("mult"),
                  estim.method = c("mpl"))

fit.t <- gofCopula(tCopula(dispstr = "un", 
                           df.fixed = TRUE), 
                   cbind(ranks, ranks_1), 
                   simulation = c("mult"),
                   estim.method = c("mpl"))

fit.j <-gofCopula(joeCopula(dim = 2), 
                  cbind(ranks, ranks_1), 
                  simulation = c("pb"),
                  estim.method = c("mpl"),
                  ties = FALSE)# What are the repurcussions of setting this to pb instead of mult?

log_likelihoods = c(clayton_cop@loglik, gaussian_cop@loglik, gumbel_cop@loglik,  frank_cop@loglik, t_copula@loglik,joe_copula@loglik)
p_values = c(fit.c$p.value,fit.n$p.value, fit.g$p.value, fit.f$p.value, fit.t$p.value,fit.j$p.value) 

# Create a data frame
results_table <- data.frame(
  Copula = c("Clayton", "Gaussian", "Gumbel", "Frank", "T", "Joe"),
  LogLikelihood = log_likelihoods,
  PValue = p_values
)

##########################################################################################

# Step 6: Empirical Copula versus Model Comparison
EGC = empCopula(cbind(ranks,ranks_1), smoothing = c("none"), offset = 0, ties.method ="random" )

# Empirical Copula
dev.new(width = 5, height= 5)
wireframe2(EGC, FUN=pCopula, n.grid = 58, draw.4.pCoplines = TRUE)

# Model Copula
gumbel_cop_mod <- gumbelCopula(param = 3.81629, dim = 2)  # Adjust param based on your fit

# Adjust the plotting area to increase the right margin
par(mar = c(5, 4, 4, 8))  # Increase the right margin for the legend

# Plot Contours of Empirical vs. Gumbel Copula
dev.new(width = 5, height = 5)
plot(cbind(ranks, ranks_1), main = "Empirical vs Gumbel Copula Baseline", ylab = expression(u[2]), xlab = expression(u[1]), ylim = c(0, 1), xlim = c(0, 1), xaxs = "i", yaxs = "i",     pch = 19,         # Use solid circles for points
     col = "darkblue", # Set point color to dark blue for contrast
     cex = 1.2)

# Increase point size for better visibility

contour(EGC, 
        FUN = pCopula, 
        n.grid = 58, 
        delta = 0, 
        add = TRUE, 
        drawlabels = FALSE,        # Removes numbers from the empirical copula contour
        lty = 2,                   # Dashed line type for empirical copula
        lwd = 2)                   # Thicker lines for visibility

# Contour for Gumbel Copula (solid line with larger labels)
contour(gumbel_cop_mod, 
        FUN = pCopula, 
        n.grid = 58, 
        add = TRUE, 
        lty = 1,                   # Solid line type for Gumbel copula
        lwd = 2,                   # Thicker lines for visibility
        labcex = 1.5)              # Increase the size of contour labels
