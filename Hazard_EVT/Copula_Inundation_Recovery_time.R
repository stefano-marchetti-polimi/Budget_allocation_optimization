# Copula Fitting
# By Santiago Taguado, Stefano Marchetti & Thomas Matteo Coscia
# October 10th, 2025 

# Libraries
########################################################################

library(dplyr)
library(lubridate)
library(copula)
library(corrplot)
library(VineCopula)
library(CDVineCopulaConditional) 
library(NSM3)
library(truncnorm)
library(copula)

###################################################################################
# Assess data is stationary. In order to assess you can perform tests  such as Petit test.
####################################################################################

# Step 1
ranks = pobs(Moderate_Inundation_Time_Duration$`Maximum Elevation (Meters)`)
ranks_1 = pobs(Moderate_Inundation_Time_Duration$`Duration (Hours)`)

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

###################################################################################
# Step 7: Climate Change Impact
set.seed(42)

# Use first 24 rows (adjust as needed)
idx <- seq_len(24)
dat <- Moderate_Inundation_Time_Duration[idx, ]

fit_for_mean <- function(mu, sd = 0.01) {
  # 1) CC impact (truncated normal > 0), same length as dat
  CC_impact <- rtruncnorm(n = nrow(dat), a = 0, mean = mu, sd = sd)
  
  # 2) Adjusted inundation heights: Elevation + impact (NOT duration)
  CC_height <- dat$`Maximum Elevation (Meters)` + CC_impact
  
  # 3) Pseudo-observations
  ranks   <- pobs(CC_height)
  ranks_1 <- pobs(dat$`Duration (Hours)`)
  U <- cbind(ranks, ranks_1)
  
  # 4) Gumbel copula fit (θ >= 1)
  gumbel_cop <- fitCopula(gumbelCopula(dim = 2), U, method = "ml")
  
  # 5) Coefficients from summary()  <-- use $coef (list)
  sm <- summary(gumbel_cop)
  coef_mat <- sm$coef  # matrix with columns: Estimate, Std. Error, z value, Pr(>|z|)
  
  # Tidy coefficients table (add mu and parameter name)
  coef_df <- data.frame(
    mean      = mu,
    param     = rownames(coef_mat),
    Estimate  = coef_mat[, "Estimate"],
    Std.Error = coef_mat[, "Std. Error"],
    z.value   = if ("z value" %in% colnames(coef_mat)) coef_mat[, "z value"] else NA_real_,
    Pr_gt_z   = if ("Pr(>|z|)" %in% colnames(coef_mat)) coef_mat[, "Pr(>|z|)"] else NA_real_,
    row.names = NULL
  )
  
  # Estimate & variance (variance = SE^2)
  theta_hat <- coef_df$Estimate[1]
  var_hat   <- coef_df$Std.Error[1]^2
  
  # Print per run
  cat(sprintf("mu = %.2f -> estimate = %.6f, variance = %.6f\n",
              mu, theta_hat, var_hat))
  print(coef_df); cat("\n")
  
  list(
    mean     = mu,
    estimate = theta_hat,
    variance = var_hat,
    coef_df  = coef_df
  )
}

means <- c(0.05, 0.10, 0.15, 0.20, 0.25)

# Run and collect
results_list <- lapply(means, fit_for_mean)

# Summary tables
gumbel_est_var <- do.call(rbind, lapply(results_list, \(x)
                                        data.frame(mean = x$mean, estimate = x$estimate, variance = x$variance)))

gumbel_coefs <- do.call(rbind, lapply(results_list, `[[`, "coef_df"))

# Final prints
cat("=== Estimate & Variance per mu ===\n")
print(gumbel_est_var, row.names = FALSE)

cat("\n=== Full Coefficients (from summary) ===\n")
print(gumbel_coefs, row.names = FALSE)
