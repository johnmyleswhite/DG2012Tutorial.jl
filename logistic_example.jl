# Load libraries.
load("sgd.jl")

# Set up a linear model.
lm = LogisticModel(zeros(2))

# Fit a simple linear model with one-pass through the data.
fit(lm, "data/logistic.csv", false, 2, 1, 0.01, :constant, true, true, false, 10)

# Look at the results a bit.
lm

# Try again with larger mini-batches.
lm = LogisticModel(zeros(2))
fit(lm, "data/logistic.csv", false, 2, 1, 0.001, :constant, true, true, false, 100)

predict(lm, "data/logistic.csv")
