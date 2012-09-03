df <- read.csv("data/training.csv", header = FALSE)
y <- df[, 1]
x <- df[, 2:91]

lm.fit <- lm(y ~ as.matrix(x))

sqrt(mean(residuals(lm.fit)^2))

