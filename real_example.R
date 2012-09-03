# Load the data into memory.
df <- read.csv("data/training.csv", header = FALSE)

# Split the data into inputs and outputs.
y <- df[, 1]
x <- df[, 2:91]

# Make a generic prediction using lm().
lm.fit <- lm(y ~ as.matrix(x))

# Assess the RMSE of the fitted model.
sqrt(mean(residuals(lm.fit)^2))
