# Head the data into memory.
df <- read.csv("data/toy.csv", header = FALSE)
names(df) <- c("y", "x")

# Fit a simple linear model.
lm(y ~ x, data = df)
