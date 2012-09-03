df <- read.csv("data/toy.csv", header = FALSE)
names(df) <- c("y", "x")

lm(y ~ x, data = df)
