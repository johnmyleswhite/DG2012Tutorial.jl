# It takes longer for R for read this data set into memory than it takes Julia to run the SGD on the data.
# R also uses > 1.5 GB to represent the data set in memory, which ends at 750 MB later

df <- read.csv("data/training.csv", header = FALSE)
y <- df[, 1]
x <- df[, 2:91]
mean(residuals(lm(y ~ x[, 1]))^2)

lm(y ~ as.matrix(x))

mean(residuals(lm(y ~ as.matrix(x)))^2)

lm.fit <- lm(y ~ as.matrix(x))
